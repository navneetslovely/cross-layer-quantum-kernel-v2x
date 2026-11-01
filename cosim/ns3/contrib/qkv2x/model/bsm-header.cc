#include "bsm-header.h"

#include "ns3/log.h"

#include <cstring>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("BsmHeader");
NS_OBJECT_ENSURE_REGISTERED(BsmHeader);

namespace
{
// Portable double <-> uint64 bit-cast (avoids strict-aliasing UB).
inline uint64_t
DoubleToBits(double d)
{
    uint64_t u;
    std::memcpy(&u, &d, sizeof(u));
    return u;
}

inline double
BitsToDouble(uint64_t u)
{
    double d;
    std::memcpy(&d, &u, sizeof(d));
    return d;
}
} // namespace

BsmHeader::BsmHeader() = default;

TypeId
BsmHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::BsmHeader").SetParent<Header>().AddConstructor<BsmHeader>();
    return tid;
}

TypeId
BsmHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
BsmHeader::GetSerializedSize() const
{
    // 8 (bsmId) + 4 (claimedId) + 5*8 (claimed doubles) + 8 (genTime)
    // + 4 (txNodeId) + 1 (attackType) + 1 (isAttack) = 66 bytes.
    return 66;
}

void
BsmHeader::Serialize(Buffer::Iterator start) const
{
    start.WriteHtonU64(m_bsmId);
    start.WriteHtonU32(m_claimedId);
    start.WriteHtonU64(DoubleToBits(m_claimedX));
    start.WriteHtonU64(DoubleToBits(m_claimedY));
    start.WriteHtonU64(DoubleToBits(m_claimedSpeed));
    start.WriteHtonU64(DoubleToBits(m_claimedHeading));
    start.WriteHtonU64(DoubleToBits(m_claimedAccel));
    start.WriteHtonU64(m_genTimeNs);
    start.WriteHtonU32(m_txNodeId);
    start.WriteU8(m_attackType);
    start.WriteU8(m_isAttack);
}

uint32_t
BsmHeader::Deserialize(Buffer::Iterator start)
{
    m_bsmId = start.ReadNtohU64();
    m_claimedId = start.ReadNtohU32();
    m_claimedX = BitsToDouble(start.ReadNtohU64());
    m_claimedY = BitsToDouble(start.ReadNtohU64());
    m_claimedSpeed = BitsToDouble(start.ReadNtohU64());
    m_claimedHeading = BitsToDouble(start.ReadNtohU64());
    m_claimedAccel = BitsToDouble(start.ReadNtohU64());
    m_genTimeNs = start.ReadNtohU64();
    m_txNodeId = start.ReadNtohU32();
    m_attackType = start.ReadU8();
    m_isAttack = start.ReadU8();
    return GetSerializedSize();
}

void
BsmHeader::Print(std::ostream& os) const
{
    os << "bsmId=" << m_bsmId << " claimedId=" << m_claimedId << " pos=(" << m_claimedX << ","
       << m_claimedY << ") spd=" << m_claimedSpeed << " hdg=" << m_claimedHeading
       << " atk=" << static_cast<uint32_t>(m_attackType)
       << " mal=" << static_cast<uint32_t>(m_isAttack);
}

} // namespace ns3
