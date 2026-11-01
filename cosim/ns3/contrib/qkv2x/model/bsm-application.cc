#include "bsm-application.h"

#include "bsm-header.h"

#include "ns3/log.h"
#include "ns3/packet-socket-factory.h"
#include "ns3/packet.h"
#include "ns3/random-variable-stream.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"

#include <algorithm>
#include <random>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("BsmApplication");
NS_OBJECT_ENSURE_REGISTERED(BsmApplication);

namespace
{
// Globally-unique BSM id source (single-threaded ns-3 simulator).
uint64_t g_bsmCounter = 1;
constexpr uint16_t QKV2X_PROTO = 0x88dc; // WSMP-like EtherType
// Micro-stagger between a Sybil node's ghost transmissions (only when power varies)
// so each ghost is a clean, separate frame whose PHY TX power can be set on its own.
// A single radio cannot emit multiple identities at the exact same instant anyway,
// so this is also more physically faithful; 800 us > one 300-B BSM's air time.
constexpr double GHOST_STAGGER_S = 0.0008;
} // namespace

BsmApplication::BsmApplication() = default;

BsmApplication::~BsmApplication() = default;

TypeId
BsmApplication::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::BsmApplication").SetParent<Application>().AddConstructor<BsmApplication>();
    return tid;
}

void
BsmApplication::Setup(uint32_t nodeId,
                      Ptr<NetDevice> device,
                      CrossLayerLogger* logger,
                      double rateHz,
                      uint32_t sizeBytes,
                      double historyHorizonS)
{
    m_nodeId = nodeId;
    m_device = device;
    m_logger = logger;
    m_interval = (rateHz > 0.0) ? (1.0 / rateHz) : 0.1;
    m_sizeBytes = sizeBytes;
    m_historyHorizon = historyHorizonS;
    // Seed-controlled jitter source (draws from the per-run ns-3 RNG stream).
    m_rng = CreateObject<UniformRandomVariable>();
}

void
BsmApplication::StartApplication()
{
    if (!m_socketReady)
    {
        m_socket = Socket::CreateSocket(GetNode(), PacketSocketFactory::GetTypeId());
        m_peer.SetSingleDevice(m_device->GetIfIndex());
        m_peer.SetPhysicalAddress(m_device->GetBroadcast());
        m_peer.SetProtocol(QKV2X_PROTO);
        if (m_socket->Bind(m_peer) == -1)
        {
            NS_FATAL_ERROR("BsmApplication: packet socket Bind failed on node " << m_nodeId);
        }
        m_socket->SetAllowBroadcast(true);
        m_socketReady = true;
    }
    // Grab the 802.11p PHY once so Sybil 'offset'/'random' modes can vary the
    // per-ghost transmit power. GetTxPowerStart() is the base power configured on
    // the device (TxPowerStart==TxPowerEnd in this testbed).
    if (!m_phy && m_device)
    {
        Ptr<WifiNetDevice> wnd = DynamicCast<WifiNetDevice>(m_device);
        if (wnd && wnd->GetPhy())
        {
            m_phy = wnd->GetPhy();
            m_txPowerBaseDbm = m_phy->GetTxPowerStart();
        }
    }
}

void
BsmApplication::StopApplication()
{
    Deactivate();
    if (m_socket)
    {
        m_socket->Close();
        m_socket = nullptr;
        m_socketReady = false;
    }
}

void
BsmApplication::Activate(const std::string& sumoId,
                         uint32_t claimedId,
                         const AttackProfile& profile)
{
    m_sumoId = sumoId;
    m_claimedId = claimedId;
    m_profile = profile;
    m_history.clear();
    m_haveTrue = false;
    m_active = true;
    m_firstSend = true;   // randomize this vehicle's send phase (desync the fleet)
    Simulator::Cancel(m_sendEvent);
    ScheduleNext();
}

void
BsmApplication::Deactivate()
{
    // Restore base TX power if this vehicle varied it (so a node-pool slot reused
    // by a later vehicle starts clean); no-op for every non-power-varying vehicle.
    if (m_phy && m_profile.type == QkAttackType::SYBIL && m_profile.sybilPowerMode != 0)
    {
        SetTxPowerDbm(m_txPowerBaseDbm);
    }
    m_active = false;
    Simulator::Cancel(m_sendEvent);
    m_history.clear();
    m_haveTrue = false;
}

void
BsmApplication::UpdateTrueState(const TrueState& s)
{
    m_true = s;
    m_haveTrue = true;
    m_history.push_back(s);
    const double cutoff = s.t - m_historyHorizon;
    while (!m_history.empty() && m_history.front().t < cutoff)
    {
        m_history.pop_front();
    }
}

void
BsmApplication::ScheduleNext()
{
    if (!m_active)
    {
        return;
    }
    double interval = m_interval;
    const double now = Simulator::Now().GetSeconds();
    if (m_profile.type == QkAttackType::FLOODING && now >= m_profile.onset &&
        m_profile.floodingFactor > 1.0)
    {
        interval = m_interval / m_profile.floodingFactor;
    }
    // BUG-4 fix: do NOT schedule on a fixed grid. The FIRST send lands at a uniformly
    // random phase in [0, interval) so each vehicle has its own offset; subsequent sends
    // carry ±10% jitter so vehicles never re-align. Without this, every vehicle transmits
    // at the same 0.1 s step instants and collides en masse (~0 reception for normal
    // single-shot senders, while Sybil's 4-packet burst survives — an artifact).
    double delay;
    if (m_firstSend)
    {
        delay = m_rng->GetValue(0.0, interval);
        m_firstSend = false;
    }
    else
    {
        delay = interval + m_rng->GetValue(-0.1 * interval, 0.1 * interval);
        if (delay < 0.0)
        {
            delay = 0.0;
        }
    }
    m_sendEvent = Simulator::Schedule(Seconds(delay), &BsmApplication::SendBsm, this);
}

void
BsmApplication::SendBsm()
{
    if (!m_active || !m_socketReady || !m_haveTrue)
    {
        ScheduleNext();
        return;
    }
    const double now = Simulator::Now().GetSeconds();

    const bool sybilActive =
        (m_profile.type == QkAttackType::SYBIL && now >= m_profile.onset);
    const bool varyPower = sybilActive && m_profile.sybilPowerMode != 0 && m_phy;

    // Primary identity always transmits at the base power.
    if (varyPower)
    {
        SetTxPowerDbm(m_txPowerBaseDbm);
    }
    ClaimResult c = AttackEngine::ComputeClaim(m_profile, m_true, m_history, now);
    DoSendOne(m_claimedId, c.x, c.y, c.speed, c.headingDeg, c.accel, c.type, c.isAttack);

    // Sybil ghosts: extra pseudonyms from the same physical transmitter.
    if (sybilActive)
    {
        if (!varyPower)
        {
            // 'same' power mode (default): all ghosts at base power, sent inline.
            for (int g = 0; g < m_profile.sybilGhosts; ++g)
            {
                uint32_t gid;
                double gx, gy, gspd, ghdg, gacc;
                AttackEngine::ComputeSybilGhost(m_profile, m_true, g, m_claimedId,
                                                gid, gx, gy, gspd, ghdg, gacc);
                DoSendOne(gid, gx, gy, gspd, ghdg, gacc, QkAttackType::SYBIL, true);
            }
        }
        else
        {
            // 'offset'/'random' modes: stagger each ghost so its own PHY TX power
            // is applied to a clean, separate frame (distinct received power per
            // identity — the realistic Sybil power signature).
            for (int g = 0; g < m_profile.sybilGhosts; ++g)
            {
                Simulator::Schedule(Seconds(GHOST_STAGGER_S * (g + 1)),
                                    &BsmApplication::SendGhost, this, g);
            }
        }
    }

    ScheduleNext();
}

void
BsmApplication::SendGhost(int ghostIdx)
{
    if (!m_active || !m_socketReady || !m_haveTrue ||
        m_profile.type != QkAttackType::SYBIL)
    {
        return;
    }
    uint32_t gid;
    double gx, gy, gspd, ghdg, gacc;
    AttackEngine::ComputeSybilGhost(m_profile, m_true, ghostIdx, m_claimedId,
                                    gid, gx, gy, gspd, ghdg, gacc);
    if (m_phy)
    {
        SetTxPowerDbm(m_txPowerBaseDbm + GhostPowerDeltaDb(ghostIdx));
    }
    DoSendOne(gid, gx, gy, gspd, ghdg, gacc, QkAttackType::SYBIL, true);
}

void
BsmApplication::SetTxPowerDbm(double dbm)
{
    if (m_phy)
    {
        m_phy->SetTxPowerStart(dbm);
        m_phy->SetTxPowerEnd(dbm);
    }
}

double
BsmApplication::GhostPowerDeltaDb(int ghostIdx)
{
    const double off = m_profile.sybilPowerOffsetDb;
    if (m_profile.sybilPowerMode == 1) // 'offset': distinct, deterministic per ghost
    {
        return off * (ghostIdx + 1);
    }
    if (m_profile.sybilPowerMode == 2) // 'random': per-ghost draw in [-off, +off]
    {
        std::uniform_real_distribution<double> u(-off, off);
        return u(m_profile.rng);
    }
    return 0.0; // 'same'
}

void
BsmApplication::DoSendOne(uint32_t claimedId,
                          double cx,
                          double cy,
                          double cspeed,
                          double cheading,
                          double caccel,
                          QkAttackType attackType,
                          bool isAttack)
{
    const double now = Simulator::Now().GetSeconds();

    BsmHeader hdr;
    hdr.m_bsmId = g_bsmCounter++;
    hdr.m_claimedId = claimedId;
    hdr.m_claimedX = cx;
    hdr.m_claimedY = cy;
    hdr.m_claimedSpeed = cspeed;
    hdr.m_claimedHeading = cheading;
    hdr.m_claimedAccel = caccel;
    hdr.m_genTimeNs = Simulator::Now().GetNanoSeconds();
    hdr.m_txNodeId = m_nodeId;
    hdr.m_attackType = static_cast<uint8_t>(attackType);
    hdr.m_isAttack = isAttack ? 1 : 0;

    const uint32_t headerSize = hdr.GetSerializedSize();
    const uint32_t pad = (m_sizeBytes > headerSize) ? (m_sizeBytes - headerSize) : 0;
    Ptr<Packet> packet = Create<Packet>(pad);
    packet->AddHeader(hdr);

    m_socket->SendTo(packet, 0, m_peer);

    if (m_logger)
    {
        m_logger->LogTx(hdr.m_bsmId,
                        now,
                        m_nodeId,
                        m_sumoId,
                        claimedId,
                        cx,
                        cy,
                        cspeed,
                        cheading,
                        caccel,
                        hdr.m_attackType,
                        hdr.m_isAttack);
    }
}

} // namespace ns3
