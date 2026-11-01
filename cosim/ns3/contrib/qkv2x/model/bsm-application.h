/*
 * qkv2x — BSM broadcast application (plan §5.2, §5.4).
 *
 * One instance is installed on every vehicle node. When the orchestrator maps a
 * live SUMO vehicle onto the node it calls Activate(); the app then broadcasts
 * BSMs at the configured rate over 802.11p, applying the assigned attack to the
 * *content* at send time. The physical transmission is real, so received power
 * at every neighbour reflects the true source position regardless of the claim.
 */
#ifndef QKV2X_BSM_APPLICATION_H
#define QKV2X_BSM_APPLICATION_H

#include "attack-engine.h"
#include "cross-layer-logger.h"

#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/net-device.h"
#include "ns3/packet-socket-address.h"
#include "ns3/ptr.h"
#include "ns3/socket.h"

#include <cstdint>
#include <deque>
#include <string>

namespace ns3
{

class WifiPhy;                // forward declaration; full header included in the .cc
class UniformRandomVariable;  // for per-vehicle send-time jitter (BUG-4 fix)

class BsmApplication : public Application
{
  public:
    static TypeId GetTypeId();
    BsmApplication();
    ~BsmApplication() override;

    /** One-time wiring done by the orchestrator after device install. */
    void Setup(uint32_t nodeId,
               Ptr<NetDevice> device,
               CrossLayerLogger* logger,
               double rateHz,
               uint32_t sizeBytes,
               double historyHorizonS);

    /** Map a live vehicle onto this node. */
    void Activate(const std::string& sumoId, uint32_t claimedId, const AttackProfile& profile);
    /** Vehicle left the network; stop transmitting. */
    void Deactivate();
    bool IsActive() const
    {
        return m_active;
    }

    /** Push the latest ground-truth state (called every sync step). */
    void UpdateTrueState(const TrueState& s);

  private:
    void StartApplication() override;
    void StopApplication() override;

    void ScheduleNext();
    void SendBsm();
    /** Send one Sybil ghost (scheduled, staggered) at its own TX power. */
    void SendGhost(int ghostIdx);
    /** Set this node's 802.11p PHY transmit power [dBm] (Sybil power control). */
    void SetTxPowerDbm(double dbm);
    /** Per-ghost TX-power delta [dB] from the base power, per sybilPowerMode. */
    double GhostPowerDeltaDb(int ghostIdx);
    void DoSendOne(uint32_t claimedId,
                   double cx,
                   double cy,
                   double cspeed,
                   double cheading,
                   double caccel,
                   QkAttackType attackType,
                   bool isAttack);

    // wiring
    uint32_t m_nodeId{0};
    Ptr<NetDevice> m_device;
    CrossLayerLogger* m_logger{nullptr};
    double m_interval{0.1};
    uint32_t m_sizeBytes{300};
    double m_historyHorizon{30.0};

    // socket
    Ptr<Socket> m_socket;
    PacketSocketAddress m_peer;
    bool m_socketReady{false};

    // 802.11p PHY handle + base TX power, for per-ghost Sybil power control.
    Ptr<WifiPhy> m_phy;
    double m_txPowerBaseDbm{0.0};

    // per-vehicle runtime state
    bool m_active{false};
    std::string m_sumoId;
    uint32_t m_claimedId{0};
    AttackProfile m_profile;
    TrueState m_true;
    bool m_haveTrue{false};
    std::deque<TrueState> m_history;
    EventId m_sendEvent;

    // BUG-4 fix: per-vehicle send-time jitter so the fleet does NOT transmit on the same
    // step grid (synchronized sends collided to ~0 reception). Seed-controlled for repro.
    Ptr<UniformRandomVariable> m_rng;
    bool m_firstSend{true};
};

} // namespace ns3

#endif // QKV2X_BSM_APPLICATION_H
