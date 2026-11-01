/*
 * qkv2x — log-distance path loss with temporally-correlated (Gauss-Markov)
 * log-normal shadowing:
 *
 *   P_rx(d) = P_tx - PL0 - 10*eta*log10(d/d0) - X_sigma,   X_sigma ~ N(0, sigma^2)
 *
 * Shadowing is correlated in time per ordered link with constant tau:
 *   X(t) = rho*X(t-) + sqrt(1-rho^2)*sigma*Z,   rho = exp(-dt/tau),  Z ~ N(0,1)
 *
 * This makes the cross-layer RSSI feature (F2) physically meaningful: the
 * signal-implied distance has correlated, not white, error — exactly the
 * regime in which the receiver's path-loss inversion must be robust.
 */
#ifndef QKV2X_CORRELATED_SHADOWING_LOSS_MODEL_H
#define QKV2X_CORRELATED_SHADOWING_LOSS_MODEL_H

#include "ns3/nstime.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/random-variable-stream.h"

#include <map>

namespace ns3
{

class CorrelatedShadowingLossModel : public PropagationLossModel
{
  public:
    static TypeId GetTypeId();
    CorrelatedShadowingLossModel();
    ~CorrelatedShadowingLossModel() override;

    // non-copyable
    CorrelatedShadowingLossModel(const CorrelatedShadowingLossModel&) = delete;
    CorrelatedShadowingLossModel& operator=(const CorrelatedShadowingLossModel&) = delete;

  private:
    double DoCalcRxPower(double txPowerDbm,
                         Ptr<MobilityModel> a,
                         Ptr<MobilityModel> b) const override;
    int64_t DoAssignStreams(int64_t stream) override;

    struct ShadowState
    {
        double valueDb{0.0};
        double timeS{-1.0};
    };

    using LinkKey = std::pair<const MobilityModel*, const MobilityModel*>;
    static LinkKey MakeKey(Ptr<MobilityModel> a, Ptr<MobilityModel> b);

    double m_exponent{2.75};       //!< eta
    double m_referenceLoss{47.86}; //!< PL0 [dB] at d0
    double m_referenceDistance{1.0}; //!< d0 [m]
    double m_sigmaDb{3.0};         //!< shadowing std [dB]
    double m_tau{1.0};             //!< temporal correlation constant [s]

    Ptr<NormalRandomVariable> m_z;
    mutable std::map<LinkKey, ShadowState> m_state;
};

} // namespace ns3

#endif // QKV2X_CORRELATED_SHADOWING_LOSS_MODEL_H
