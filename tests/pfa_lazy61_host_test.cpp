// Host algebra check of the actual bounded odd-radix helper, without OpenCL.
// The primitive weak reduction is modeled with uint128; GPU instruction
// lowering and end-to-end carry behavior require separate device validation.
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>

#include "../src/UseOptions.h"

using Z61 = std::uint64_t;
using u32 = std::uint32_t;
using Wide = unsigned __int128;
constexpr Z61 M61 = (Z61(1) << 61) - 1;
struct GF61 { Z61 x, y; };
GF61 U2(Z61 x, Z61 y) { return {x, y}; }
GF61 operator+(GF61 a, GF61 b) { return {a.x + b.x, a.y + b.y}; }
GF61 operator-(GF61 a, GF61 b) { return {a.x - b.x, a.y - b.y}; }
Z61 modM61(Z61 a) { return (a & M61) + (a >> 61); }
GF61 modM61(GF61 a) { return {modM61(a.x), modM61(a.y)}; }
GF61 neg(GF61 a, u32 count) { return {count * M61 - a.x, count * M61 - a.y}; }
GF61 sub(GF61 a, GF61 b) { return modM61(a + neg(b, 2)); }
GF61 modM61q(GF61 a, u32 count) {
    return modM61(a + U2(count * M61, count * M61));
}
Z61 weakMul(Z61 a, Z61 b, u32 ac, u32 bc) {
    // Actual callers have at most six summed sloppy inputs (+42).
    assert(Wide(a) <= Wide(ac - 1) * M61 + 64);
    assert(Wide(b) <= Wide(bc - 1) * M61 + 7);
    assert(ac + bc <= 9);
    assert(Wide(a) + b <= std::numeric_limits<Z61>::max());
    assert((ac - 1) * (bc - 1) <= 6);
    const Wide product = Wide(a) * b;
    const Wide folded = (product & M61) + (product >> 61);
    assert(folded <= std::numeric_limits<Z61>::max());
    return static_cast<Z61>(folded);
}
#define OVERLOAD
#define PFA_RADIX 7
#define PFA7_ROOT61 ((1773674115463894065ULL + 673484497713682504ULL) % M61)
#include "../src/cl/pfa-lazy61.cl"

Z61 mulmod(Z61 a, Z61 b) { return static_cast<Z61>(Wide(a) * b % M61); }
Z61 power(Z61 a, Z61 n) {
    Z61 result = 1;
    for (; n; n >>= 1, a = mulmod(a, a))
        if (n & 1) result = mulmod(result, a);
    return result;
}
using Values = std::array<GF61, 7>;
Values reference(const Values& input, unsigned n, Z61 root) {
    Values out{};
    for (unsigned k = 0; k < n; ++k) {
        Wide x = 0, y = 0;
        for (unsigned j = 0; j < n; ++j) {
            const Z61 w = power(root, j * k);
            x += mulmod(input[j].x, w);
            y += mulmod(input[j].y, w);
        }
        out[k] = U2(static_cast<Z61>(x % M61), static_cast<Z61>(y % M61));
    }
    return out;
}
Values transform(Values u, unsigned n, Z61 root) {
    if (n == 7) aevumPfa7Lazy61(u.data(), root);
    else aevumPfa3Lazy61(u[0], u[1], u[2], root, &u[0], &u[1], &u[2]);
    return u;
}
void check(const Values& input, unsigned n) {
    const Z61 root = n == 7 ? PFA7_ROOT61 : 1669582390241348315ULL;
    const Z61 inverse = power(root, M61 - 2);
    assert(root != 1 && power(root, n) == 1);
    for (const Z61 direction : {root, inverse}) {
        const Values actual = transform(input, n, direction);
        const Values expected = reference(input, n, direction);
        for (unsigned j = 0; j < n; ++j) {
            assert(actual[j].x <= M61 + 7 && actual[j].y <= M61 + 7);
            assert(actual[j].x % M61 == expected[j].x);
            assert(actual[j].y % M61 == expected[j].y);
        }
    }
    const Values roundtrip = transform(transform(input, n, root), n, inverse);
    const Z61 inv_n = power(n, M61 - 2);
    for (unsigned j = 0; j < n; ++j) {
        assert(mulmod(roundtrip[j].x, inv_n) == input[j].x % M61);
        assert(mulmod(roundtrip[j].y, inv_n) == input[j].y % M61);
    }
}
int main() {
    assert(aevumUseKey("AEVUM_PFA_LAZY61"));
    for (unsigned n : {3u, 7u}) {
        Values u{};
        for (Z61 value : {Z61(0), Z61(1), M61 - 1, M61, M61 + 1, M61 + 7}) {
            u.fill(U2(value, value));
            check(u, n);
            for (unsigned j = 0; j < n; ++j) {
                u.fill(U2(0, 0));
                u[j] = U2(value, M61 + 7 - value);
                check(u, n);
            }
        }
        // All corners of the input range, with opposite real/imaginary lanes.
        for (unsigned mask = 0; mask < (1u << n); ++mask) {
            for (unsigned j = 0; j < n; ++j)
                u[j] = (mask & (1u << j)) ? U2(M61 + 7, 0) : U2(0, M61 + 7);
            check(u, n);
        }
        std::mt19937_64 rng(0x61A3E7);
        for (unsigned trial = 0; trial < 2048; ++trial) {
            for (auto& value : u) value = U2(rng() % (M61 + 8), rng() % (M61 + 8));
            check(u, n);
        }
    }
}
