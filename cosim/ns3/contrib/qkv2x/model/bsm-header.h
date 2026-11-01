/*
 * qkv2x — Cross-layer quantum-kernel V2X misbehaviour detection.
 * Basic Safety Message (BSM) wire format carried inside each 802.11p broadcast.
 *
 * The header carries the *claimed* kinematics that a receiver observes, plus
 * two ground-truth-only fields (txNodeId, isAttack/attackType) that are written
 * to disk by the transmitter for labelling and NEVER used as detection features.
 */
#ifndef QKV2X_BSM_HEADER_H
#define QKV2X_BSM_HEADER_H

#include "ns3/header.h"
#include <cstdint>

namespace ns3
{

/** Attack taxonomy (plan §5.2). Kept in sync with attack-engine.h. */
enum class QkAttackType : uint8_t
{
    NONE = 0,
    CONST_POS = 1,
    CONST_OFFSET = 2,
    RANDOM_POS = 3,
    RANDOM_OFFSET = 4,
    EVENTUAL_STOP = 5,
    DRIFT = 6,
    REPLAY = 7,
    DATA_FALSIFICATION = 8,
    SYBIL = 9,
    FLOODING = 10
};

/**
 * \brief Fixed-length (66 byte) BSM header.
 *
 * Fields are serialised big-endian. Doubles are bit-cast to uint64 so the
 * format is portable across machines (Linux <-> macOS).
 */
class BsmHeader : public Header
{
  public:
    BsmHeader();

    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;

    // --- claimed / observable fields -------------------------------------
    uint64_t m_bsmId{0};         //!< globally unique BSM id (tx side) — join key
    uint32_t m_claimedId{0};     //!< claimed pseudonym/station id (spoofed for Sybil)
    double m_claimedX{0.0};      //!< claimed position x [m]
    double m_claimedY{0.0};      //!< claimed position y [m]
    double m_claimedSpeed{0.0};  //!< claimed speed [m/s]
    double m_claimedHeading{0.0};//!< claimed heading [deg, SUMO navigational]
    double m_claimedAccel{0.0};  //!< claimed longitudinal acceleration [m/s^2]
    uint64_t m_genTimeNs{0};     //!< generation timestamp [ns] (for Age-of-Information)

    // --- ground-truth-only (logged, never a feature) ---------------------
    uint32_t m_txNodeId{0};      //!< ns-3 node index of the *physical* transmitter
    uint8_t m_attackType{0};     //!< QkAttackType active for THIS message
    uint8_t m_isAttack{0};       //!< 1 if this message is malicious in content/identity/rate
};

} // namespace ns3

#endif // QKV2X_BSM_HEADER_H
