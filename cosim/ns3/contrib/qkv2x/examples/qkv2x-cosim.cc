/*
 * qkv2x-cosim — SUMO <-> ns-3 co-simulation orchestrator.
 *
 * Couples a real SUMO run (mobility + attacker injection) to a real ns-3
 * 802.11p network (channel + reception), logging the three raw streams the
 * cross-layer feature pipeline consumes (tx.csv, rx.csv, truth.csv).
 *
 * Build (inside an ns-3.42 tree with SUMO's libtraci available):
 *   ./ns3 run "qkv2x-cosim -- --help"
 *
 * Every CLI scalar is set by the Python sweep driver (qkv2x.cosim.run_matrix),
 * which constructs one command line per sweep point for full reproducibility.
 */
#include "ns3/applications-module.h"
#include "ns3/attack-engine.h"
#include "ns3/bsm-application.h"
#include "ns3/bsm-header.h"
#include "ns3/core-module.h"
#include "ns3/cross-layer-logger.h"
#include "ns3/llc-snap-header.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-module.h"
#include "ns3/wifi-module.h"

#include <libsumo/libtraci.h>

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;
using libtraci::Simulation;
using libtraci::Vehicle;

namespace
{
constexpr double PI = 3.14159265358979323846;

uint32_t
Fnv32(const std::string& s)
{
    uint32_t h = 2166136261u;
    for (char c : s)
    {
        h ^= static_cast<uint8_t>(c);
        h *= 16777619u;
    }
    return h ? h : 1u;
}

std::vector<std::string>
Split(const std::string& s, char delim)
{
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(s);
    while (std::getline(ss, item, delim))
    {
        if (!item.empty())
        {
            out.push_back(item);
        }
    }
    return out;
}

// Reception sniffer: peels the 802.11p MAC header + BsmHeader and logs an rx row
// with the PHY-measured received power. Bound args carry the receiver identity.
void
RxSniffer(CrossLayerLogger* logger,
          uint32_t rxNode,
          bool rxIsRsu,
          Ptr<MobilityModel> rxMob,
          Ptr<const Packet> packet,
          uint16_t /*channelFreqMhz*/,
          WifiTxVector /*txVector*/,
          MpduInfo /*aMpdu*/,
          SignalNoiseDbm signalNoise,
          uint16_t /*staId*/)
{
    Ptr<Packet> p = packet->Copy();
    WifiMacHeader mac;
    if (p->GetSize() < mac.GetSerializedSize())
    {
        return;
    }
    p->RemoveHeader(mac);
    if (!mac.IsData())
    {
        return; // ignore ACK/control frames
    }
    // The BSM is sent through a PacketSocket on the WifiNetDevice (EtherType
    // QKV2X_PROTO == 0x88dc), so the device prepends an LLC/SNAP header before the
    // BsmHeader. Strip it here, or the BsmHeader would deserialise from the SNAP
    // bytes (bsm_id == 0xAAAA0300000088DC) and every field would be garbage.
    constexpr uint16_t QKV2X_PROTO = 0x88dc; // must match BsmApplication's EtherType
    LlcSnapHeader llc;
    if (p->GetSize() < llc.GetSerializedSize())
    {
        return;
    }
    p->RemoveHeader(llc);
    if (llc.GetType() != QKV2X_PROTO)
    {
        return; // not one of our BSM frames
    }
    BsmHeader bsm;
    if (p->GetSize() < bsm.GetSerializedSize())
    {
        return;
    }
    p->RemoveHeader(bsm);

    const Vector pos = rxMob->GetPosition();
    logger->LogRx(bsm.m_bsmId,
                  Simulator::Now().GetSeconds(),
                  rxNode,
                  rxIsRsu,
                  pos.x,
                  pos.y,
                  signalNoise.signal,
                  signalNoise.noise,
                  bsm.m_claimedId,
                  bsm.m_claimedX,
                  bsm.m_claimedY,
                  bsm.m_claimedSpeed,
                  bsm.m_claimedHeading,
                  bsm.m_claimedAccel,
                  static_cast<double>(bsm.m_genTimeNs) * 1e-9,
                  bsm.m_txNodeId,
                  bsm.m_attackType,
                  bsm.m_isAttack);
}

/** Owns all co-sim state and is re-scheduled every step from the ns-3 timeline. */
class CoSimOrchestrator
{
  public:
    CoSimOrchestrator(double stepLen,
                      double endTime,
                      uint32_t truthEvery,
                      AttackEngine* engine,
                      CrossLayerLogger* logger,
                      std::vector<Ptr<BsmApplication>> apps,
                      std::vector<Ptr<ConstantVelocityMobilityModel>> mobs)
        : m_stepLen(stepLen),
          m_endTime(endTime),
          m_truthEvery(truthEvery),
          m_engine(engine),
          m_logger(logger),
          m_apps(std::move(apps)),
          m_mobs(std::move(mobs))
    {
        for (uint32_t i = 0; i < m_apps.size(); ++i)
        {
            m_free.push_back(i);
        }
    }

    void Start()
    {
        Simulator::Schedule(Seconds(m_stepLen), &CoSimOrchestrator::Step, this);
    }

    uint32_t Peak() const
    {
        return m_peakConcurrent;
    }

    uint64_t Activations() const
    {
        return m_activations;
    }

  private:
    void Step()
    {
        const double now = Simulator::Now().GetSeconds();
        if (now > m_endTime)
        {
            return;
        }

        Simulation::step(now); // advance SUMO to the current ns-3 time
        if (Simulation::getMinExpectedNumber() <= 0 && m_active.empty() && now > 1.0)
        {
            Simulator::Stop();
            return;
        }

        // Vehicles that left the network this step.
        for (const std::string& id : Simulation::getArrivedIDList())
        {
            auto it = m_active.find(id);
            if (it != m_active.end())
            {
                const uint32_t slot = it->second;
                m_apps[slot]->Deactivate();
                ParkNode(slot);
                m_free.push_back(slot);
                m_active.erase(it);
            }
        }

        // Vehicles that entered the network this step.
        for (const std::string& id : Simulation::getDepartedIDList())
        {
            if (m_active.count(id))
            {
                continue;
            }
            if (m_free.empty())
            {
                NS_FATAL_ERROR("Node pool exhausted ("
                               << m_apps.size()
                               << " slots). Increase node_pool in the sim config.");
            }
            const uint32_t slot = m_free.back();
            m_free.pop_back();
            m_active[id] = slot;

            AttackProfile prof = m_engine->AssignProfile(id, now);
            m_apps[slot]->Activate(id, Fnv32(id), prof);
            ++m_activations;
        }

        m_peakConcurrent = std::max<uint32_t>(m_peakConcurrent, m_active.size());

        // Push ground truth into ns-3 mobility + the transmitting apps.
        const bool logTruth = (m_stepCount % m_truthEvery) == 0;
        for (const auto& kv : m_active)
        {
            const std::string& id = kv.first;
            const uint32_t slot = kv.second;
            const libsumo::TraCIPosition pos = Vehicle::getPosition(id);
            const double spd = Vehicle::getSpeed(id);
            const double ang = Vehicle::getAngle(id);
            const double acc = Vehicle::getAcceleration(id);

            const double a = ang * PI / 180.0;
            m_mobs[slot]->SetPosition(Vector(pos.x, pos.y, 1.5));
            m_mobs[slot]->SetVelocity(Vector(spd * std::sin(a), spd * std::cos(a), 0.0));

            TrueState ts;
            ts.t = now;
            ts.x = pos.x;
            ts.y = pos.y;
            ts.speed = spd;
            ts.headingDeg = ang;
            ts.accel = acc;
            m_apps[slot]->UpdateTrueState(ts);

            if (logTruth)
            {
                m_logger->LogTruth(now, id, slot, pos.x, pos.y, spd, ang, acc);
            }
        }

        ++m_stepCount;
        if (now + m_stepLen <= m_endTime)
        {
            Simulator::Schedule(Seconds(m_stepLen), &CoSimOrchestrator::Step, this);
        }
    }

    void ParkNode(uint32_t slot)
    {
        // Park each idle slot at a UNIQUE, far-apart coordinate (5 km spacing, >> radio range).
        // BUGFIX: previously every idle slot was parked at the SAME (1e7,1e7). A BSM scheduled for
        // a vehicle's arrival/slot-reuse step could still fire after the node was parked; with all
        // idle nodes co-located there, that stray send was mass-received by ~1000 nodes at spurious
        // d0 (~ -27 dBm) loss, corrupting ~57% of rx rows. Unique far positions guarantee a stray
        // park-position send reaches no receiver (real nodes and other parked slots are out of range).
        m_mobs[slot]->SetPosition(Vector(1e7 + slot * 5000.0, 1e7, 0.0));
        m_mobs[slot]->SetVelocity(Vector(0.0, 0.0, 0.0));
    }

    double m_stepLen;
    double m_endTime;
    uint32_t m_truthEvery;
    AttackEngine* m_engine;
    CrossLayerLogger* m_logger;
    std::vector<Ptr<BsmApplication>> m_apps;
    std::vector<Ptr<ConstantVelocityMobilityModel>> m_mobs;

    std::map<std::string, uint32_t> m_active;
    std::vector<uint32_t> m_free;
    uint64_t m_stepCount{0};
    uint64_t m_activations{0};
    uint32_t m_peakConcurrent{0};
};

std::map<QkAttackType, double>
ParseMix(const std::string& spec)
{
    std::map<QkAttackType, double> mix;
    for (const std::string& tok : Split(spec, ','))
    {
        const auto pos = tok.find(':');
        if (pos == std::string::npos)
        {
            continue;
        }
        const QkAttackType t = AttackTypeFromName(tok.substr(0, pos));
        const double w = std::stod(tok.substr(pos + 1));
        if (t != QkAttackType::NONE)
        {
            mix[t] = w;
        }
    }
    return mix;
}
} // namespace

int
main(int argc, char* argv[])
{
    // ---- parameters (all overridable from the Python sweep driver) ----------
    std::string sumoBinary = "sumo";
    std::string sumoCfg;
    std::string sumoExtraArgs; // space-separated, appended verbatim
    double stepLen = 0.1;
    double warmup = 30.0;
    double duration = 300.0;
    double beginTime = 0.0;
    uint32_t nodePool = 400;
    uint32_t truthEvery = 1;

    uint32_t rsuCount = 4;
    double rsuMargin = 100.0;
    double rsuZ = 6.0;

    double txPowerDbm = 20.0;
    double plExp = 2.75;
    double refLossDb = 47.86;
    double refDist = 1.0;
    double shadowSigmaDb = 3.0;
    double shadowTauS = 1.0;
    bool nakagami = false;
    double rxSensDbm = -94.0;
    double ccaThreshDbm = -84.0;
    std::string dataRate = "OfdmRate6MbpsBW10MHz";

    double bsmRateHz = 10.0;
    uint32_t bsmSizeBytes = 300;

    double attackFraction = 0.15;
    double onsetMin = 10.0;
    double onsetMax = 60.0;
    std::string mixSpec =
        "const_pos:1,const_offset:1,random_pos:1,random_offset:1,eventual_stop:1,drift:1,replay:1,"
        "data_falsification:1,sybil:1,flooding:1";
    double constOffsetMin = 50.0;
    double constOffsetMax = 200.0;
    double randomOffsetRadius = 100.0;
    double driftSpeed = 2.0;
    double replayDelay = 5.0;
    double dataFalsSpeedMin = 0.0;
    double dataFalsSpeedMax = 2.5;
    double dataFalsHeadingJitter = 60.0;
    double eventualStopDelay = 10.0;
    uint32_t sybilGhosts = 3;
    double sybilSpacing = 8.0;
    uint32_t sybilPowerMode = 0;
    double sybilPowerOffsetDb = 3.0;
    double floodingFactor = 10.0;

    uint32_t seed = 1;
    std::string outDir = ".";

    CommandLine cmd(__FILE__);
    cmd.AddValue("sumoBinary", "SUMO binary (sumo or sumo-gui)", sumoBinary);
    cmd.AddValue("sumoCfg", "Path to the .sumocfg", sumoCfg);
    cmd.AddValue("sumoExtraArgs", "Extra SUMO args (space-separated)", sumoExtraArgs);
    cmd.AddValue("stepLen", "Sync/step length [s]", stepLen);
    cmd.AddValue("warmup", "Warmup [s] before logging", warmup);
    cmd.AddValue("duration", "Logged duration [s]", duration);
    cmd.AddValue("beginTime", "SUMO --begin [s]", beginTime);
    cmd.AddValue("nodePool", "Pre-allocated vehicle node slots", nodePool);
    cmd.AddValue("truthEvery", "Log truth every N steps", truthEvery);
    cmd.AddValue("rsuCount", "Number of static RSU receivers", rsuCount);
    cmd.AddValue("rsuMargin", "RSU inset from net bbox [m]", rsuMargin);
    cmd.AddValue("rsuZ", "RSU antenna height [m]", rsuZ);
    cmd.AddValue("txPowerDbm", "TX power [dBm]", txPowerDbm);
    cmd.AddValue("plExp", "Path-loss exponent eta", plExp);
    cmd.AddValue("refLossDb", "PL0 [dB] at d0", refLossDb);
    cmd.AddValue("refDist", "d0 [m]", refDist);
    cmd.AddValue("shadowSigmaDb", "Shadowing sigma [dB]", shadowSigmaDb);
    cmd.AddValue("shadowTauS", "Shadowing correlation tau [s]", shadowTauS);
    cmd.AddValue("nakagami", "Append Nakagami fast fading", nakagami);
    cmd.AddValue("rxSensDbm", "RX sensitivity [dBm]", rxSensDbm);
    cmd.AddValue("ccaThreshDbm", "CCA-ED threshold [dBm]", ccaThreshDbm);
    cmd.AddValue("dataRate", "802.11p OFDM mode", dataRate);
    cmd.AddValue("bsmRateHz", "BSM rate [Hz]", bsmRateHz);
    cmd.AddValue("bsmSizeBytes", "BSM on-air size [B]", bsmSizeBytes);
    cmd.AddValue("attackFraction", "Fraction of vehicles that attack", attackFraction);
    cmd.AddValue("onsetMin", "Attack onset min [s after depart]", onsetMin);
    cmd.AddValue("onsetMax", "Attack onset max [s after depart]", onsetMax);
    cmd.AddValue("mixSpec", "Attack mix weights k:v,...", mixSpec);
    cmd.AddValue("constOffsetMin", "Const-offset min [m]", constOffsetMin);
    cmd.AddValue("constOffsetMax", "Const-offset max [m]", constOffsetMax);
    cmd.AddValue("randomOffsetRadius", "Random-offset radius [m]", randomOffsetRadius);
    cmd.AddValue("driftSpeed", "Drift speed [m/s]", driftSpeed);
    cmd.AddValue("replayDelay", "Replay delay [s]", replayDelay);
    cmd.AddValue("dataFalsSpeedMin", "Data-falsification speed scale min", dataFalsSpeedMin);
    cmd.AddValue("dataFalsSpeedMax", "Data-falsification speed scale max", dataFalsSpeedMax);
    cmd.AddValue("dataFalsHeadingJitter", "Data-falsification heading jitter [deg]",
                 dataFalsHeadingJitter);
    cmd.AddValue("eventualStopDelay", "Eventual-stop lead-in [s]", eventualStopDelay);
    cmd.AddValue("sybilGhosts", "Sybil ghost identities", sybilGhosts);
    cmd.AddValue("sybilSpacing", "Sybil ghost trail spacing [m]", sybilSpacing);
    cmd.AddValue("sybilPowerMode", "Sybil power mode (0 same,1 offset,2 random)", sybilPowerMode);
    cmd.AddValue("sybilPowerOffsetDb", "Sybil power offset [dB]", sybilPowerOffsetDb);
    cmd.AddValue("floodingFactor", "Flooding rate multiplier", floodingFactor);
    cmd.AddValue("seed", "Master RNG seed", seed);
    cmd.AddValue("outDir", "Output directory for CSVs + meta.json", outDir);
    cmd.Parse(argc, argv);

    if (sumoCfg.empty())
    {
        NS_FATAL_ERROR("--sumoCfg is required");
    }

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(1);

    // ---- start SUMO via libtraci -------------------------------------------
    std::vector<std::string> sumoCmd;
    sumoCmd.push_back(sumoBinary);
    sumoCmd.push_back("-c");
    sumoCmd.push_back(sumoCfg);
    sumoCmd.push_back("--step-length");
    sumoCmd.push_back(std::to_string(stepLen));
    sumoCmd.push_back("--seed");
    sumoCmd.push_back(std::to_string(seed));
    sumoCmd.push_back("--no-step-log");
    sumoCmd.push_back("true");
    sumoCmd.push_back("--no-warnings");
    sumoCmd.push_back("true");
    if (beginTime > 0.0)
    {
        sumoCmd.push_back("--begin");
        sumoCmd.push_back(std::to_string(beginTime));
    }
    for (const std::string& a : Split(sumoExtraArgs, ' '))
    {
        sumoCmd.push_back(a);
    }
    Simulation::start(sumoCmd);

    // Net boundary for the random-position attack + RSU placement.
    const libsumo::TraCIPositionVector b = Simulation::getNetBoundary();
    double xmin = 1e18, ymin = 1e18, xmax = -1e18, ymax = -1e18;
    for (const auto& pt : b.value)
    {
        xmin = std::min(xmin, pt.x);
        ymin = std::min(ymin, pt.y);
        xmax = std::max(xmax, pt.x);
        ymax = std::max(ymax, pt.y);
    }

    const double endTime = warmup + duration;

    // ---- logger ------------------------------------------------------------
    CrossLayerLogger logger;
    logger.Open(outDir, warmup);

    // ---- attack engine -----------------------------------------------------
    AttackEngine::Config ac;
    ac.seed = seed;
    ac.fraction = attackFraction;
    ac.onsetMin = onsetMin;
    ac.onsetMax = onsetMax;
    ac.mix = ParseMix(mixSpec);
    ac.constOffsetMin = constOffsetMin;
    ac.constOffsetMax = constOffsetMax;
    ac.randomOffsetRadius = randomOffsetRadius;
    ac.driftSpeed = driftSpeed;
    ac.replayDelay = replayDelay;
    ac.dataFalsSpeedMin = dataFalsSpeedMin;
    ac.dataFalsSpeedMax = dataFalsSpeedMax;
    ac.dataFalsHeadingJitter = dataFalsHeadingJitter;
    ac.eventualStopDelay = eventualStopDelay;
    ac.sybilGhosts = static_cast<int>(sybilGhosts);
    ac.sybilSpacing = sybilSpacing;
    ac.sybilPowerMode = static_cast<int>(sybilPowerMode);
    ac.sybilPowerOffsetDb = sybilPowerOffsetDb;
    ac.floodingFactor = floodingFactor;
    ac.xmin = xmin;
    ac.ymin = ymin;
    ac.xmax = xmax;
    ac.ymax = ymax;
    AttackEngine engine(ac);

    // ---- nodes -------------------------------------------------------------
    NodeContainer vehicleNodes;
    vehicleNodes.Create(nodePool);
    NodeContainer rsuNodes;
    rsuNodes.Create(rsuCount);

    // ---- 802.11p PHY/MAC + custom channel ----------------------------------
    YansWifiChannelHelper channelHelper;
    channelHelper.AddPropagationLoss("ns3::CorrelatedShadowingLossModel",
                                     "Exponent", DoubleValue(plExp),
                                     "ReferenceLoss", DoubleValue(refLossDb),
                                     "ReferenceDistance", DoubleValue(refDist),
                                     "ShadowingSigma", DoubleValue(shadowSigmaDb),
                                     "ShadowingTau", DoubleValue(shadowTauS));
    if (nakagami)
    {
        channelHelper.AddPropagationLoss("ns3::NakagamiPropagationLossModel");
    }
    channelHelper.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    Ptr<YansWifiChannel> channel = channelHelper.Create();

    YansWifiPhyHelper phy;
    phy.SetChannel(channel);
    phy.Set("TxPowerStart", DoubleValue(txPowerDbm));
    phy.Set("TxPowerEnd", DoubleValue(txPowerDbm));
    phy.Set("RxSensitivity", DoubleValue(rxSensDbm));
    phy.Set("CcaEdThreshold", DoubleValue(ccaThreshDbm));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue(dataRate),
                                 "ControlMode", StringValue(dataRate));
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    NetDeviceContainer vehDevices = wifi.Install(phy, mac, vehicleNodes);
    NetDeviceContainer rsuDevices = wifi.Install(phy, mac, rsuNodes);

    // ---- mobility ----------------------------------------------------------
    MobilityHelper vehMob;
    vehMob.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    vehMob.Install(vehicleNodes);

    MobilityHelper rsuMob;
    rsuMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    rsuMob.Install(rsuNodes);

    // Place RSUs on a grid inside the inset bounding box.
    const uint32_t gridN = static_cast<uint32_t>(std::ceil(std::sqrt((double)rsuCount)));
    const double gx0 = xmin + rsuMargin;
    const double gy0 = ymin + rsuMargin;
    const double gw = std::max(1.0, (xmax - rsuMargin) - gx0);
    const double gh = std::max(1.0, (ymax - rsuMargin) - gy0);
    for (uint32_t i = 0; i < rsuCount; ++i)
    {
        const uint32_t r = i / gridN;
        const uint32_t c = i % gridN;
        const double rx = gx0 + (gridN > 1 ? gw * c / (gridN - 1) : gw * 0.5);
        const double ry = gy0 + (gridN > 1 ? gh * r / (gridN - 1) : gh * 0.5);
        rsuNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(rx, ry, rsuZ));
    }

    // ---- packet sockets + BSM apps -----------------------------------------
    PacketSocketHelper packetSocket;
    packetSocket.Install(vehicleNodes);
    packetSocket.Install(rsuNodes);

    std::vector<Ptr<BsmApplication>> apps;
    std::vector<Ptr<ConstantVelocityMobilityModel>> mobs;
    apps.reserve(nodePool);
    mobs.reserve(nodePool);
    for (uint32_t i = 0; i < nodePool; ++i)
    {
        Ptr<Node> node = vehicleNodes.Get(i);
        Ptr<ConstantVelocityMobilityModel> m =
            node->GetObject<ConstantVelocityMobilityModel>();
        m->SetPosition(Vector(1e7 + i * 5000.0, 1e7, 0.0)); // parked at a unique far slot until activated
        mobs.push_back(m);

        Ptr<BsmApplication> app = CreateObject<BsmApplication>();
        app->Setup(i, vehDevices.Get(i), &logger, bsmRateHz, bsmSizeBytes, replayDelay + 5.0);
        node->AddApplication(app);
        app->SetStartTime(Seconds(0.0));
        app->SetStopTime(Seconds(endTime));
        apps.push_back(app);
    }

    // ---- reception sniffers on every PHY (vehicles + RSUs) -----------------
    for (uint32_t i = 0; i < nodePool; ++i)
    {
        Ptr<WifiNetDevice> wnd = DynamicCast<WifiNetDevice>(vehDevices.Get(i));
        Ptr<MobilityModel> m = vehicleNodes.Get(i)->GetObject<MobilityModel>();
        wnd->GetPhy()->TraceConnectWithoutContext(
            "MonitorSnifferRx", MakeBoundCallback(&RxSniffer, &logger, i, false, m));
    }
    for (uint32_t i = 0; i < rsuCount; ++i)
    {
        Ptr<WifiNetDevice> wnd = DynamicCast<WifiNetDevice>(rsuDevices.Get(i));
        Ptr<MobilityModel> m = rsuNodes.Get(i)->GetObject<MobilityModel>();
        const uint32_t rxId = nodePool + i; // RSU ids continue after the vehicle pool
        wnd->GetPhy()->TraceConnectWithoutContext(
            "MonitorSnifferRx", MakeBoundCallback(&RxSniffer, &logger, rxId, true, m));
    }

    // ---- run ---------------------------------------------------------------
    CoSimOrchestrator orch(stepLen, endTime, truthEvery, &engine, &logger, apps, mobs);
    orch.Start();

    Simulator::Stop(Seconds(endTime + stepLen));
    Simulator::Run();

    Simulation::close();
    logger.Close();
    Simulator::Destroy();

    // ---- meta.json ---------------------------------------------------------
    std::ofstream meta(outDir + "/meta.json");
    meta << "{\n"
         << "  \"sumo_cfg\": \"" << sumoCfg << "\",\n"
         << "  \"step_len\": " << stepLen << ",\n"
         << "  \"warmup\": " << warmup << ",\n"
         << "  \"duration\": " << duration << ",\n"
         << "  \"begin_time\": " << beginTime << ",\n"
         << "  \"node_pool\": " << nodePool << ",\n"
         << "  \"rsu_count\": " << rsuCount << ",\n"
         << "  \"net_bbox\": [" << xmin << ", " << ymin << ", " << xmax << ", " << ymax << "],\n"
         << "  \"tx_power_dbm\": " << txPowerDbm << ",\n"
         << "  \"pl_exponent\": " << plExp << ",\n"
         << "  \"ref_loss_db\": " << refLossDb << ",\n"
         << "  \"ref_distance\": " << refDist << ",\n"
         << "  \"shadowing_sigma_db\": " << shadowSigmaDb << ",\n"
         << "  \"shadowing_tau_s\": " << shadowTauS << ",\n"
         << "  \"nakagami\": " << (nakagami ? "true" : "false") << ",\n"
         << "  \"rx_sensitivity_dbm\": " << rxSensDbm << ",\n"
         << "  \"cca_threshold_dbm\": " << ccaThreshDbm << ",\n"
         << "  \"data_rate\": \"" << dataRate << "\",\n"
         << "  \"bsm_rate_hz\": " << bsmRateHz << ",\n"
         << "  \"bsm_size_bytes\": " << bsmSizeBytes << ",\n"
         << "  \"attack_fraction\": " << attackFraction << ",\n"
         << "  \"mix_spec\": \"" << mixSpec << "\",\n"
         << "  \"seed\": " << seed << ",\n"
         << "  \"peak_concurrent\": " << orch.Peak() << ",\n"
         << "  \"activations\": " << orch.Activations() << "\n"
         << "}\n";
    meta.close();

    return 0;
}
