// Direct host check of the production normalized inverse helper.
// Primitive reductions model the current sloppy ranges; an independent
// uint128 DFT checks the transform. Device lowering is a separate gate.
#include <array>
#include <cassert>
#include <cstdint>
#include <random>
#include "../src/UseOptions.h"

using Z31 = std::uint32_t;
using Z61 = std::uint64_t;
using Wide = unsigned __int128;
constexpr Z31 M31 = (Z31(1) << 31) - 1;
constexpr Z61 M61 = (Z61(1) << 61) - 1;
struct GF31 { Z31 x, y; };
struct GF61 { Z61 x, y; };
GF31 U2(Z31 x, Z31 y) { return {x, y}; }
GF61 U2(Z61 x, Z61 y) { return {x, y}; }
GF61 operator+(GF61 a, GF61 b) { return U2(a.x + b.x, a.y + b.y); }
GF61 operator-(GF61 a, GF61 b) { return U2(a.x - b.x, a.y - b.y); }
Z31 fold31(Z31 a) { return (a & M31) + (a >> 31); }
Z61 fold61(Z61 a) { return (a & M61) + (a >> 61); }
Z31 add(Z31 a, Z31 b) { return fold31(a + b); }
Z61 add(Z61 a, Z61 b) { return fold61(a + b); }
Z31 sub(Z31 a, Z31 b) {
    const std::int32_t d = static_cast<std::int32_t>(a - b);
    return static_cast<Z31>((d & M31) + (d >> 31));
}
Z61 sub(Z61 a, Z61 b) { return fold61(a + 2 * M61 - b); }
GF31 add(GF31 a, GF31 b) { return U2(add(a.x, b.x), add(a.y, b.y)); }
GF61 add(GF61 a, GF61 b) { return U2(add(a.x, b.x), add(a.y, b.y)); }
GF31 sub(GF31 a, GF31 b) { return U2(sub(a.x, b.x), sub(a.y, b.y)); }
GF61 sub(GF61 a, GF61 b) { return U2(sub(a.x, b.x), sub(a.y, b.y)); }
GF61 modM61q(GF61 a, unsigned count) {
    return U2(fold61(a.x + count * M61), fold61(a.y + count * M61));
}
unsigned products31 = 0, products61 = 0;
Z31 mul(Z31 a, Z31 b) {
    ++products31;
    assert(a <= M31 && b <= M31);
    const Z61 p = Z61(a) * b;
    return fold31(static_cast<Z31>((p & M31) + (p >> 31)));
}
Z61 mul(Z61 a, Z61 b) {
    ++products61;
    assert(a <= M61 + 7 && b < M61);
    const Wide p = Wide(a) * b;
    return fold61(static_cast<Z61>((p & M61) + (p >> 61)));
}

#define OVERLOAD
#define NTT_GF31 1
#define NTT_GF61 1
#define PFA7_INV7_31 1840700269U
#define PFA7_INV7_61 1976436865040309101ULL
#include "../src/cl/pfa7-normalized.cl"

Z61 mulmod(Z61 a, Z61 b, Z61 p) { return static_cast<Z61>(Wide(a) * b % p); }
Z61 power(Z61 a, Z61 n, Z61 p) {
    Z61 result = 1;
    for (; n; n >>= 1, a = mulmod(a, a, p))
        if (n & 1) result = mulmod(result, a, p);
    return result;
}
template<class Vec>
using Values = std::array<Vec, 7>;
template<class Vec>
Values<Vec> dft(const Values<Vec>& input, Z61 root, Z61 p, Z61 scale) {
    using Scalar = decltype(Vec{}.x);
    Values<Vec> result{};
    for (unsigned k = 0; k < 7; ++k) {
        Wide x = 0, y = 0;
        for (unsigned j = 0; j < 7; ++j) {
            const Z61 w = power(root, j * k, p);
            x += mulmod(input[j].x, w, p);
            y += mulmod(input[j].y, w, p);
        }
        result[k] = {static_cast<Scalar>(mulmod(static_cast<Z61>(x % p), scale, p)),
                     static_cast<Scalar>(mulmod(static_cast<Z61>(y % p), scale, p))};
    }
    return result;
}
template<class Vec>
void check(const Values<Vec>& input, Z61 root, Z61 p, Z61 limit) {
    const auto expected = dft(input, power(root, p - 2, p), p, power(7, p - 2, p));
    auto got = input;
    products31 = products61 = 0;
    aevumPfa7InverseNormalized(got.data());
    // Nine products in each of the two independent scalar components.
    assert(products31 + products61 == 18);
    for (unsigned i = 0; i < 7; ++i) {
        assert(got[i].x <= limit && got[i].y <= limit);
        assert(got[i].x % p == expected[i].x);
        assert(got[i].y % p == expected[i].y);
    }
    auto roundtrip = dft(input, root, p, 1);
    aevumPfa7InverseNormalized(roundtrip.data());
    for (unsigned i = 0; i < 7; ++i) {
        assert(roundtrip[i].x % p == input[i].x % p);
        assert(roundtrip[i].y % p == input[i].y % p);
    }
}
template<class Vec>
void exercise(Z61 p, Z61 root, Z61 limit) {
    using Scalar = decltype(Vec{}.x);
    assert(root != 1 && power(root, 7, p) == 1);
    Values<Vec> input{};
    for (Z61 value : {Z61(0), Z61(1), p - 1, p, limit}) {
        input.fill({static_cast<Scalar>(value), static_cast<Scalar>(value)});
        check(input, root, p, limit);
        for (unsigned i = 0; i < 7; ++i) {
            input.fill({0, 0});
            input[i] = {static_cast<Scalar>(value), static_cast<Scalar>(limit - value)};
            check(input, root, p, limit);
        }
    }
    for (unsigned mask = 0; mask < 128; ++mask) {
        for (unsigned i = 0; i < 7; ++i)
            input[i] = (mask & (1u << i)) ? Vec{static_cast<Scalar>(limit), 0} :
                                          Vec{0, static_cast<Scalar>(limit)};
        check(input, root, p, limit);
    }
    std::mt19937_64 rng(0x7A31A61);
    for (unsigned trial = 0; trial < 2048; ++trial) {
        for (auto& a : input) a = {static_cast<Scalar>(rng() % (limit + 1)),
                                  static_cast<Scalar>(rng() % (limit + 1))};
        check(input, root, p, limit);
    }
}
int main() {
    assert(aevumUseKey("AEVUM_PFA7_FOLD_INV7"));
    exercise<GF31>(M31, (141971251ULL + 752284155ULL) % M31, M31);
    exercise<GF61>(M61, (1773674115463894065ULL + 673484497713682504ULL) % M61, M61 + 7);
}
