// Copyright (C) Mihai Preda

// Bounded GF61 odd-radix arithmetic. Inputs and outputs are in [0, M61+7].
// See docs/PFA_LAZY61.md for the bounds and the exact normalization points.

GF61 OVERLOAD aevumPfaMulBounded61(GF61 a, Z61 w, const u32 count) {
  return U2(modM61(weakMul(a.x, w, count, 2)),
            modM61(weakMul(a.y, w, count, 2)));
}

void OVERLOAD aevumPfa3Lazy61(GF61 a0, GF61 a1, GF61 a2, Z61 w,
                             GF61 *o0, GF61 *o1, GF61 *o2) {
  // d <= 3*M61+7; wd <= 4*M61+7. Keep the weak product until the outputs.
  const GF61 d = a1 + neg(a2, 2);
  const GF61 wd = U2(weakMul(d.x, w, 4, 2), weakMul(d.y, w, 4, 2));
  *o0 = modM61(a0 + a1 + a2);
  *o1 = modM61q(a0 - a2 + wd, 2);
  *o2 = modM61q(a0 - a1 - wd, 6);
}

#if PFA_RADIX == 7
// Same eight-product Nussbaumer factorization and roots as aevum_pfa7_nuss8.
// Defer folds across the additions, retaining bounded sine inputs and even
// outputs so no final pre-fold expression can reach 2^64.
void OVERLOAD aevumPfa7Lazy61(GF61 *u, Z61 w) {
  const GF61 x0 = u[0];
  const GF61 a1 = u[1] + u[6], a2 = u[2] + u[5], a3 = u[3] + u[4];
  const GF61 b1 = sub(u[1], u[6]), b2 = sub(u[2], u[5]), b3 = sub(u[3], u[4]);
  const GF61 total = a1 + a2 + a3;
  const GF61 m2 = aevumPfaMulBounded61(a1 + neg(a3, 3), (Z61)916476061952082211UL, 6);
  const GF61 tc = aevumPfaMulBounded61(a2 + neg(a3, 3), (Z61)1789062672330551777UL, 6) + m2;
  const GF61 mc = aevumPfaMulBounded61(a2 + neg(a1, 3), (Z61)399695725068940037UL, 6) + m2;
  const GF61 center = x0 + aevumPfaMulBounded61(total, (Z61)384307168202282325UL, 7);
  u[0] = modM61(x0 + total);
  const GF61 e1 = modM61q(center - tc, 3);
  const GF61 e2 = modM61(center + mc);
  const GF61 e3 = modM61q(center + tc - mc, 3);

  const bool forward = w == (Z61)PFA7_ROOT61;
  const GF61 m6 = aevumPfaMulBounded61(b3 + b1,
      forward ? (Z61)2007045408887657666UL : (Z61)298797600326036285UL, 3);
  const GF61 tq = aevumPfaMulBounded61(b2 + b3,
      forward ? (Z61)1848645854929387642UL : (Z61)457197154284306309UL, 3) + m6;
  const GF61 sq = aevumPfaMulBounded61(b1 + neg(b2, 2),
      forward ? (Z61)755994754610342594UL : (Z61)1549848254603351357UL, 4) + m6;
  const GF61 q0 = aevumPfaMulBounded61(b2 + neg(b3, 2) + b1,
      forward ? (Z61)374686897387646219UL : (Z61)1931156111826047732UL, 5);
  u[1] = modM61q(e1 + q0 - tq, 3);
  u[6] = modM61q(e1 - q0 + tq, 2);
  u[2] = modM61(e2 + q0 + sq);
  u[5] = modM61q(e2 - q0 - sq, 4);
  u[3] = modM61q(e3 + sq - q0 - tq, 4);
  u[4] = modM61q(e3 - sq + q0 + tq, 3);
}
#endif
