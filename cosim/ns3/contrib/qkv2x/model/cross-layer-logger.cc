#include "cross-layer-logger.h"

#include "ns3/fatal-error.h"

#include <cstdio>

namespace ns3
{

CrossLayerLogger::~CrossLayerLogger()
{
    Close();
}

void
CrossLayerLogger::Open(const std::string& outDir, double logStart)
{
    m_logStart = logStart;
    m_tx.open(outDir + "/tx.csv", std::ios::out | std::ios::trunc);
    m_rx.open(outDir + "/rx.csv", std::ios::out | std::ios::trunc);
    m_truth.open(outDir + "/truth.csv", std::ios::out | std::ios::trunc);
    if (!m_tx.good() || !m_rx.good() || !m_truth.good())
    {
        NS_FATAL_ERROR("CrossLayerLogger: cannot open output CSVs in " << outDir);
    }
    // Big buffers: these files get large and we never want partial flushes to dominate.
    static char txBuf[1 << 20];
    static char rxBuf[1 << 20];
    static char trBuf[1 << 20];
    m_tx.rdbuf()->pubsetbuf(txBuf, sizeof(txBuf));
    m_rx.rdbuf()->pubsetbuf(rxBuf, sizeof(rxBuf));
    m_truth.rdbuf()->pubsetbuf(trBuf, sizeof(trBuf));

    m_tx << "bsm_id,t_gen,tx_node,sumo_id,claimed_id,claimed_x,claimed_y,claimed_speed,"
            "claimed_heading,claimed_accel,attack_type,is_attack\n";
    m_rx << "bsm_id,t_rx,rx_node,rx_is_rsu,rx_x,rx_y,rssi_dbm,noise_dbm,claimed_id,claimed_x,"
            "claimed_y,claimed_speed,claimed_heading,claimed_accel,t_gen,tx_node,attack_type,"
            "is_attack,delay_s\n";
    m_truth << "t,sumo_id,node,true_x,true_y,true_speed,true_heading,true_accel\n";
}

void
CrossLayerLogger::Close()
{
    if (m_tx.is_open())
    {
        m_tx.flush();
        m_tx.close();
    }
    if (m_rx.is_open())
    {
        m_rx.flush();
        m_rx.close();
    }
    if (m_truth.is_open())
    {
        m_truth.flush();
        m_truth.close();
    }
}

void
CrossLayerLogger::LogTx(uint64_t bsmId,
                        double tGen,
                        uint32_t txNode,
                        const std::string& sumoId,
                        uint32_t claimedId,
                        double cx,
                        double cy,
                        double cspeed,
                        double cheading,
                        double caccel,
                        uint8_t attackType,
                        uint8_t isAttack)
{
    if (tGen < m_logStart)
    {
        return;
    }
    char buf[256];
    const int n = std::snprintf(buf,
                                sizeof(buf),
                                "%llu,%.4f,%u,%s,%u,%.4f,%.4f,%.4f,%.3f,%.4f,%u,%u\n",
                                static_cast<unsigned long long>(bsmId),
                                tGen,
                                txNode,
                                sumoId.c_str(),
                                claimedId,
                                cx,
                                cy,
                                cspeed,
                                cheading,
                                caccel,
                                static_cast<unsigned>(attackType),
                                static_cast<unsigned>(isAttack));
    if (n > 0)
    {
        m_tx.write(buf, n);
    }
}

void
CrossLayerLogger::LogRx(uint64_t bsmId,
                        double tRx,
                        uint32_t rxNode,
                        bool rxIsRsu,
                        double rxX,
                        double rxY,
                        double rssiDbm,
                        double noiseDbm,
                        uint32_t claimedId,
                        double cx,
                        double cy,
                        double cspeed,
                        double cheading,
                        double caccel,
                        double tGen,
                        uint32_t txNode,
                        uint8_t attackType,
                        uint8_t isAttack)
{
    if (tRx < m_logStart)
    {
        return;
    }
    // One-way message latency (generation -> reception), in seconds. This is the
    // per-message propagation/latency field the plan §5.3 reception log calls for;
    // its windowed mean is the Age-of-Information that drives the §5.5 encoding.
    const double delayS = (tRx > tGen) ? (tRx - tGen) : 0.0;
    char buf[360];
    const int n = std::snprintf(
        buf,
        sizeof(buf),
        "%llu,%.6f,%u,%d,%.4f,%.4f,%.4f,%.4f,%u,%.4f,%.4f,%.4f,%.3f,%.4f,%.4f,%u,%u,%u,%.6f\n",
        static_cast<unsigned long long>(bsmId),
        tRx,
        rxNode,
        rxIsRsu ? 1 : 0,
        rxX,
        rxY,
        rssiDbm,
        noiseDbm,
        claimedId,
        cx,
        cy,
        cspeed,
        cheading,
        caccel,
        tGen,
        txNode,
        static_cast<unsigned>(attackType),
        static_cast<unsigned>(isAttack),
        delayS);
    if (n > 0)
    {
        m_rx.write(buf, n);
    }
}

void
CrossLayerLogger::LogTruth(double t,
                           const std::string& sumoId,
                           uint32_t node,
                           double tx,
                           double ty,
                           double tspeed,
                           double theading,
                           double taccel)
{
    if (t < m_logStart)
    {
        return;
    }
    char buf[200];
    const int n = std::snprintf(buf,
                                sizeof(buf),
                                "%.4f,%s,%u,%.4f,%.4f,%.4f,%.3f,%.4f\n",
                                t,
                                sumoId.c_str(),
                                node,
                                tx,
                                ty,
                                tspeed,
                                theading,
                                taccel);
    if (n > 0)
    {
        m_truth.write(buf, n);
    }
}

} // namespace ns3
