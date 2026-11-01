#include "attack-engine.h"

#include <algorithm>
#include <cmath>

namespace ns3
{

namespace
{
constexpr double PI = 3.14159265358979323846;

// Convert SUMO navigational heading (deg, 0=North, clockwise) to a unit vector
// in SUMO/ns-3 cartesian coordinates (x east, y north).
inline void
HeadingToUnit(double headingDeg, double& ux, double& uy)
{
    const double a = headingDeg * PI / 180.0;
    ux = std::sin(a);
    uy = std::cos(a);
}
} // namespace

const char*
AttackTypeName(QkAttackType t)
{
    switch (t)
    {
    case QkAttackType::NONE: return "none";
    case QkAttackType::CONST_POS: return "const_pos";
    case QkAttackType::CONST_OFFSET: return "const_offset";
    case QkAttackType::RANDOM_POS: return "random_pos";
    case QkAttackType::RANDOM_OFFSET: return "random_offset";
    case QkAttackType::EVENTUAL_STOP: return "eventual_stop";
    case QkAttackType::DRIFT: return "drift";
    case QkAttackType::REPLAY: return "replay";
    case QkAttackType::DATA_FALSIFICATION: return "data_falsification";
    case QkAttackType::SYBIL: return "sybil";
    case QkAttackType::FLOODING: return "flooding";
    }
    return "none";
}

QkAttackType
AttackTypeFromName(const std::string& name)
{
    if (name == "const_pos") return QkAttackType::CONST_POS;
    if (name == "const_offset") return QkAttackType::CONST_OFFSET;
    if (name == "random_pos") return QkAttackType::RANDOM_POS;
    if (name == "random_offset") return QkAttackType::RANDOM_OFFSET;
    if (name == "eventual_stop") return QkAttackType::EVENTUAL_STOP;
    if (name == "drift") return QkAttackType::DRIFT;
    if (name == "replay") return QkAttackType::REPLAY;
    if (name == "data_falsification") return QkAttackType::DATA_FALSIFICATION;
    if (name == "sybil") return QkAttackType::SYBIL;
    if (name == "flooding") return QkAttackType::FLOODING;
    return QkAttackType::NONE;
}

AttackEngine::AttackEngine(const Config& cfg)
    : m_cfg(cfg)
{
    for (const auto& kv : m_cfg.mix)
    {
        m_mixTotal += std::max(0.0, kv.second);
    }
}

uint64_t
AttackEngine::MixSeed(uint64_t a, const std::string& s)
{
    // splitmix64 over the global seed mixed with FNV-1a of the vehicle id.
    uint64_t h = 1469598103934665603ULL;
    for (char c : s)
    {
        h ^= static_cast<uint8_t>(c);
        h *= 1099511628211ULL;
    }
    uint64_t z = a + 0x9E3779B97F4A7C15ULL + h;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

QkAttackType
AttackEngine::SampleType(std::mt19937_64& rng) const
{
    if (m_mixTotal <= 0.0)
    {
        return QkAttackType::NONE;
    }
    std::uniform_real_distribution<double> u(0.0, m_mixTotal);
    double r = u(rng);
    for (const auto& kv : m_cfg.mix)
    {
        const double w = std::max(0.0, kv.second);
        if (r < w)
        {
            return kv.first;
        }
        r -= w;
    }
    return m_cfg.mix.rbegin()->first;
}

AttackProfile
AttackEngine::AssignProfile(const std::string& sumoId, double departTime) const
{
    AttackProfile p;
    std::mt19937_64 rng(MixSeed(m_cfg.seed, sumoId));
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    if (u01(rng) >= m_cfg.fraction)
    {
        return p; // benign
    }

    p.type = SampleType(rng);
    std::uniform_real_distribution<double> onset(m_cfg.onsetMin, m_cfg.onsetMax);
    p.onset = departTime + onset(rng);

    // bbox for random-position attack
    p.xmin = m_cfg.xmin;
    p.ymin = m_cfg.ymin;
    p.xmax = m_cfg.xmax;
    p.ymax = m_cfg.ymax;

    std::uniform_real_distribution<double> uOff(m_cfg.constOffsetMin, m_cfg.constOffsetMax);
    std::uniform_real_distribution<double> uAng(0.0, 2.0 * PI);
    const double mag = uOff(rng);
    const double ang = uAng(rng);
    p.offX = mag * std::cos(ang);
    p.offY = mag * std::sin(ang);

    p.radius = m_cfg.randomOffsetRadius;

    const double driftAng = uAng(rng);
    p.driftVx = m_cfg.driftSpeed * std::cos(driftAng);
    p.driftVy = m_cfg.driftSpeed * std::sin(driftAng);

    p.replayDelay = m_cfg.replayDelay;

    std::uniform_real_distribution<double> uSpd(m_cfg.dataFalsSpeedMin, m_cfg.dataFalsSpeedMax);
    p.speedScale = uSpd(rng);
    p.headingJitter = m_cfg.dataFalsHeadingJitter;

    p.stopDelay = m_cfg.eventualStopDelay;

    p.sybilGhosts = m_cfg.sybilGhosts;
    p.sybilSpacing = m_cfg.sybilSpacing;
    p.sybilPowerMode = m_cfg.sybilPowerMode;
    p.sybilPowerOffsetDb = m_cfg.sybilPowerOffsetDb;

    p.floodingFactor = m_cfg.floodingFactor;

    // dedicated per-message RNG stream
    p.rng.seed(MixSeed(m_cfg.seed ^ 0xD1B54A32D192ED03ULL, sumoId));
    return p;
}

ClaimResult
AttackEngine::ComputeClaim(AttackProfile& p,
                           const TrueState& cur,
                           const std::deque<TrueState>& history,
                           double now)
{
    ClaimResult r;
    r.x = cur.x;
    r.y = cur.y;
    r.speed = cur.speed;
    r.headingDeg = cur.headingDeg;
    r.accel = cur.accel;
    r.active = false;
    r.isAttack = false;
    r.type = QkAttackType::NONE;

    if (p.type == QkAttackType::NONE || now < p.onset)
    {
        return r; // benign content
    }

    r.active = true;
    r.type = p.type;
    r.isAttack = true;

    switch (p.type)
    {
    case QkAttackType::CONST_POS: {
        if (!p.fixedInit)
        {
            p.fixedInit = true;
            p.fixedX = cur.x;
            p.fixedY = cur.y;
        }
        r.x = p.fixedX;
        r.y = p.fixedY;
        break;
    }
    case QkAttackType::CONST_OFFSET: {
        r.x = cur.x + p.offX;
        r.y = cur.y + p.offY;
        break;
    }
    case QkAttackType::RANDOM_POS: {
        std::uniform_real_distribution<double> ux(p.xmin, p.xmax);
        std::uniform_real_distribution<double> uy(p.ymin, p.ymax);
        r.x = ux(p.rng);
        r.y = uy(p.rng);
        break;
    }
    case QkAttackType::RANDOM_OFFSET: {
        std::uniform_real_distribution<double> ur(0.0, p.radius);
        std::uniform_real_distribution<double> ua(0.0, 2.0 * PI);
        const double rad = ur(p.rng);
        const double ang = ua(p.rng);
        r.x = cur.x + rad * std::cos(ang);
        r.y = cur.y + rad * std::sin(ang);
        break;
    }
    case QkAttackType::EVENTUAL_STOP: {
        if (now < p.onset + p.stopDelay)
        {
            r.isAttack = false; // still truthful during the lead-in
            r.active = false;
            r.type = QkAttackType::NONE;
        }
        else
        {
            if (!p.frozen)
            {
                p.frozen = true;
                p.frozenX = cur.x;
                p.frozenY = cur.y;
            }
            r.x = p.frozenX;
            r.y = p.frozenY;
            r.speed = 0.0;
            r.accel = 0.0;
        }
        break;
    }
    case QkAttackType::DRIFT: {
        const double dt = now - p.onset;
        r.x = cur.x + p.driftVx * dt;
        r.y = cur.y + p.driftVy * dt;
        break;
    }
    case QkAttackType::REPLAY: {
        const double target = now - p.replayDelay;
        if (history.empty() || history.front().t > target)
        {
            r.isAttack = false; // no past state to replay yet
            r.active = false;
            r.type = QkAttackType::NONE;
            break;
        }
        // nearest history sample at or before `target`
        const TrueState* best = &history.front();
        for (const auto& s : history)
        {
            if (s.t <= target)
            {
                best = &s;
            }
            else
            {
                break;
            }
        }
        r.x = best->x;
        r.y = best->y;
        r.speed = best->speed;
        r.headingDeg = best->headingDeg;
        r.accel = best->accel;
        break;
    }
    case QkAttackType::DATA_FALSIFICATION: {
        // position truthful; kinematics inconsistent with motion
        std::uniform_real_distribution<double> uj(-p.headingJitter, p.headingJitter);
        r.speed = cur.speed * p.speedScale;
        r.headingDeg = std::fmod(cur.headingDeg + uj(p.rng) + 360.0, 360.0);
        break;
    }
    case QkAttackType::SYBIL: {
        // The PRIMARY identity stays truthful in content; the malice is the
        // ghost identities (emitted separately by the application).
        r.isAttack = false;
        r.active = false;
        r.type = QkAttackType::NONE;
        break;
    }
    case QkAttackType::FLOODING: {
        // truthful content; DoS via abnormal rate (label malicious behaviour)
        r.isAttack = true;
        r.type = QkAttackType::FLOODING;
        break;
    }
    case QkAttackType::NONE:
        break;
    }
    return r;
}

void
AttackEngine::ComputeSybilGhost(AttackProfile& p,
                                const TrueState& cur,
                                int ghostIdx,
                                uint32_t baseClaimedId,
                                uint32_t& ghostClaimedId,
                                double& gx,
                                double& gy,
                                double& gspeed,
                                double& gheadingDeg,
                                double& gaccel)
{
    // Stable distinct pseudonym per ghost.
    ghostClaimedId = baseClaimedId ^ (0xA5A50000u + static_cast<uint32_t>(ghostIdx + 1) * 2654435761u);

    double ux, uy;
    HeadingToUnit(cur.headingDeg, ux, uy);
    const double d = p.sybilSpacing * (ghostIdx + 1);
    // Ghosts trail behind the true position along the heading axis.
    gx = cur.x - ux * d;
    gy = cur.y - uy * d;
    gspeed = cur.speed;
    gheadingDeg = cur.headingDeg;
    gaccel = cur.accel;
}

} // namespace ns3
