/*
 * qkv2x — attacker injection (plan §5.2).
 *
 * Attacks override the *content* of transmitted BSMs while the physical motion
 * of the attacker stays the real SUMO trajectory, so misbehaviour is detectable
 * only via claim-vs-reality inconsistency (mobility) and claim-vs-signal
 * inconsistency (cross-layer). No attack fabricates radio physics: a Sybil ghost
 * and a falsified position are both emitted by the real transmitter, so their
 * RSSI signature is governed by the true source location.
 *
 * Assignment and per-message randomness are driven by std::mt19937_64 seeded
 * deterministically from (globalSeed, vehicleId), so a run is fully reproducible
 * and independent of event ordering.
 */
#ifndef QKV2X_ATTACK_ENGINE_H
#define QKV2X_ATTACK_ENGINE_H

#include "bsm-header.h"

#include <cstdint>
#include <deque>
#include <map>
#include <random>
#include <string>

namespace ns3
{

/** One sample of a vehicle's true (SUMO ground-truth) state. */
struct TrueState
{
    double t{0.0};          //!< sim time [s]
    double x{0.0};          //!< true position x [m]
    double y{0.0};          //!< true position y [m]
    double speed{0.0};      //!< true speed [m/s]
    double headingDeg{0.0}; //!< true heading [deg, SUMO navigational]
    double accel{0.0};      //!< true longitudinal accel [m/s^2]
};

/** Per-vehicle attack assignment + sampled parameters. */
struct AttackProfile
{
    QkAttackType type{QkAttackType::NONE};
    double onset{0.0}; //!< absolute sim time the attack becomes active

    // sampled parameters
    double offX{0.0}, offY{0.0};   //!< const offset
    double radius{0.0};            //!< random-offset disk radius
    double driftVx{0.0}, driftVy{0.0};
    double replayDelay{0.0};
    double speedScale{1.0}, headingJitter{0.0};
    double stopDelay{0.0};
    int sybilGhosts{0};
    double sybilSpacing{0.0};
    int sybilPowerMode{0};
    double sybilPowerOffsetDb{0.0};
    double floodingFactor{1.0};

    // lazily-initialised state
    bool fixedInit{false};
    double fixedX{0.0}, fixedY{0.0};
    bool frozen{false};
    double frozenX{0.0}, frozenY{0.0};

    // playground bbox (for random-position attack)
    double xmin{0.0}, ymin{0.0}, xmax{0.0}, ymax{0.0};

    std::mt19937_64 rng; //!< per-message randomness (seeded per vehicle)
};

/** Output of a claim computation for one primary BSM. */
struct ClaimResult
{
    double x{0.0}, y{0.0}, speed{0.0}, headingDeg{0.0}, accel{0.0};
    bool active{false};    //!< attack content is active at this time
    bool isAttack{false};  //!< ground-truth malicious label for this message
    QkAttackType type{QkAttackType::NONE};
};

/**
 * \brief Assigns attack profiles to vehicles and computes claimed payloads.
 */
class AttackEngine
{
  public:
    struct Config
    {
        uint64_t seed{1};
        double fraction{0.15};
        double onsetMin{10.0};
        double onsetMax{60.0};
        std::map<QkAttackType, double> mix; //!< relative weights
        // parameter ranges
        double constOffsetMin{50.0}, constOffsetMax{200.0};
        double randomOffsetRadius{100.0};
        double driftSpeed{2.0};
        double replayDelay{5.0};
        double dataFalsSpeedMin{0.0}, dataFalsSpeedMax{2.5};
        double dataFalsHeadingJitter{60.0};
        double eventualStopDelay{10.0};
        int sybilGhosts{3};
        double sybilSpacing{8.0};
        int sybilPowerMode{0};
        double sybilPowerOffsetDb{3.0};
        double floodingFactor{10.0};
        double xmin{0.0}, ymin{0.0}, xmax{0.0}, ymax{0.0};
    };

    explicit AttackEngine(const Config& cfg);

    /** Deterministically decide whether/which attack a vehicle gets. */
    AttackProfile AssignProfile(const std::string& sumoId, double departTime) const;

    /** Compute the claimed primary BSM content for `now`. Mutates lazy state in `p`. */
    static ClaimResult ComputeClaim(AttackProfile& p,
                                    const TrueState& cur,
                                    const std::deque<TrueState>& history,
                                    double now);

    /** Compute one Sybil ghost BSM (ghostIdx in [0, sybilGhosts)). */
    static void ComputeSybilGhost(AttackProfile& p,
                                  const TrueState& cur,
                                  int ghostIdx,
                                  uint32_t baseClaimedId,
                                  uint32_t& ghostClaimedId,
                                  double& gx,
                                  double& gy,
                                  double& gspeed,
                                  double& gheadingDeg,
                                  double& gaccel);

    const Config& GetConfig() const
    {
        return m_cfg;
    }

  private:
    static uint64_t MixSeed(uint64_t a, const std::string& s);
    QkAttackType SampleType(std::mt19937_64& rng) const;

    Config m_cfg;
    double m_mixTotal{0.0};
};

/** String <-> enum helpers (used by the CLI / logger). */
const char* AttackTypeName(QkAttackType t);
QkAttackType AttackTypeFromName(const std::string& name);

} // namespace ns3

#endif // QKV2X_ATTACK_ENGINE_H
