#include "correlated-shadowing-loss-model.h"

#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/simulator.h"

#include <algorithm>
#include <cmath>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("CorrelatedShadowingLossModel");
NS_OBJECT_ENSURE_REGISTERED(CorrelatedShadowingLossModel);

TypeId
CorrelatedShadowingLossModel::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::CorrelatedShadowingLossModel")
            .SetParent<PropagationLossModel>()
            .SetGroupName("Propagation")
            .AddConstructor<CorrelatedShadowingLossModel>()
            .AddAttribute("Exponent",
                          "The path-loss exponent eta.",
                          DoubleValue(2.75),
                          MakeDoubleAccessor(&CorrelatedShadowingLossModel::m_exponent),
                          MakeDoubleChecker<double>())
            .AddAttribute("ReferenceLoss",
                          "PL0: path loss [dB] at the reference distance d0 "
                          "(default Friis at 5.9 GHz, 1 m).",
                          DoubleValue(47.86),
                          MakeDoubleAccessor(&CorrelatedShadowingLossModel::m_referenceLoss),
                          MakeDoubleChecker<double>())
            .AddAttribute("ReferenceDistance",
                          "d0 [m].",
                          DoubleValue(1.0),
                          MakeDoubleAccessor(&CorrelatedShadowingLossModel::m_referenceDistance),
                          MakeDoubleChecker<double>())
            .AddAttribute("ShadowingSigma",
                          "Log-normal shadowing standard deviation [dB].",
                          DoubleValue(3.0),
                          MakeDoubleAccessor(&CorrelatedShadowingLossModel::m_sigmaDb),
                          MakeDoubleChecker<double>())
            .AddAttribute("ShadowingTau",
                          "Temporal correlation constant of the shadowing [s].",
                          DoubleValue(1.0),
                          MakeDoubleAccessor(&CorrelatedShadowingLossModel::m_tau),
                          MakeDoubleChecker<double>(1e-6));
    return tid;
}

CorrelatedShadowingLossModel::CorrelatedShadowingLossModel()
{
    m_z = CreateObject<NormalRandomVariable>();
    m_z->SetAttribute("Mean", DoubleValue(0.0));
    m_z->SetAttribute("Variance", DoubleValue(1.0));
}

CorrelatedShadowingLossModel::~CorrelatedShadowingLossModel() = default;

CorrelatedShadowingLossModel::LinkKey
CorrelatedShadowingLossModel::MakeKey(Ptr<MobilityModel> a, Ptr<MobilityModel> b)
{
    const MobilityModel* pa = PeekPointer(a);
    const MobilityModel* pb = PeekPointer(b);
    if (pa > pb)
    {
        std::swap(pa, pb);
    }
    return {pa, pb};
}

double
CorrelatedShadowingLossModel::DoCalcRxPower(double txPowerDbm,
                                            Ptr<MobilityModel> a,
                                            Ptr<MobilityModel> b) const
{
    const double d = std::max(a->GetDistanceFrom(b), 1e-3);
    const double d0 = m_referenceDistance;

    double pathLossDb = m_referenceLoss;
    if (d > d0)
    {
        pathLossDb += 10.0 * m_exponent * std::log10(d / d0);
    }

    // Temporally-correlated shadowing per link (Gauss-Markov / OU process).
    const double now = Simulator::Now().GetSeconds();
    const LinkKey key = MakeKey(a, b);
    ShadowState& st = m_state[key];

    double shadow;
    if (st.timeS < 0.0)
    {
        shadow = m_sigmaDb * m_z->GetValue();
    }
    else
    {
        const double dt = std::max(0.0, now - st.timeS);
        const double rho = std::exp(-dt / m_tau);
        const double innov = std::sqrt(std::max(0.0, 1.0 - rho * rho)) * m_sigmaDb * m_z->GetValue();
        shadow = rho * st.valueDb + innov;
    }
    st.valueDb = shadow;
    st.timeS = now;

    return txPowerDbm - pathLossDb - shadow;
}

int64_t
CorrelatedShadowingLossModel::DoAssignStreams(int64_t stream)
{
    m_z->SetStream(stream);
    return 1;
}

} // namespace ns3
