/*
 * qkv2x — CSV loggers for the three raw data streams the feature pipeline
 * consumes (plan §5.3). Everything written here is produced by the real
 * SUMO+ns-3 co-simulation; nothing is synthesised.
 *
 *   tx.csv     one row per transmitted BSM (claimed content + ground-truth label)
 *   rx.csv     one row per successfully received BSM at a receiver (carries RSSI)
 *   truth.csv  periodic ground-truth pose of every live vehicle
 */
#ifndef QKV2X_CROSS_LAYER_LOGGER_H
#define QKV2X_CROSS_LAYER_LOGGER_H

#include <cstdint>
#include <fstream>
#include <string>

namespace ns3
{

class CrossLayerLogger
{
  public:
    CrossLayerLogger() = default;
    ~CrossLayerLogger();

    CrossLayerLogger(const CrossLayerLogger&) = delete;
    CrossLayerLogger& operator=(const CrossLayerLogger&) = delete;

    /** Open the three CSVs under outDir; rows before logStart are dropped. */
    void Open(const std::string& outDir, double logStart);
    void Close();

    void LogTx(uint64_t bsmId,
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
               uint8_t isAttack);

    void LogRx(uint64_t bsmId,
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
               uint8_t isAttack);

    void LogTruth(double t,
                  const std::string& sumoId,
                  uint32_t node,
                  double tx,
                  double ty,
                  double tspeed,
                  double theading,
                  double taccel);

  private:
    double m_logStart{0.0};
    std::ofstream m_tx;
    std::ofstream m_rx;
    std::ofstream m_truth;
};

} // namespace ns3

#endif // QKV2X_CROSS_LAYER_LOGGER_H
