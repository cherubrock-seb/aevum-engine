// Copyright (C) Mihai Preda

// Normalized inverse only; forward transforms keep their existing arithmetic.
// Derivation and range proof: docs/PFA7_NORMALIZED_INVERSE.md.

#if NTT_GF31
GF31 OVERLOAD aevumPfa7InvMul(GF31 a, Z31 s) {
  return U2(mul(a.x, s), mul(a.y, s));
}

// Inverse Nussbaumer DFT with 1/7 incorporated in all eight constants.
// C1=-1/6 gives output[0] = input[0] - 6*center after center is scaled by 1/7.
void OVERLOAD aevumPfa7InverseNormalized(GF31 *u) {
  const GF31 x0 = u[0];
  const GF31 a1 = add(u[1], u[6]), a2 = add(u[2], u[5]), a3 = add(u[3], u[4]);
  const GF31 b1 = sub(u[1], u[6]), b2 = sub(u[2], u[5]), b3 = sub(u[3], u[4]);
  const GF31 total = add(add(a1, a2), a3);
  const GF31 m2 = aevumPfa7InvMul(sub(a1, a3), (Z31)644415712U);
  const GF31 tc = add(aevumPfa7InvMul(sub(a2, a3), (Z31)2140722852U), m2);
  const GF31 mc = add(aevumPfa7InvMul(sub(a2, a1), (Z31)637654917U), m2);
  const GF31 center = add(aevumPfa7InvMul(x0, (Z31)PFA7_INV7_31),
      aevumPfa7InvMul(total, (Z31)51130563U));
  // Keep the DC reconstruction inside the existing 0..M31 range.
  const GF31 twice = add(center, center);
  u[0] = sub(x0, add(twice, add(twice, twice)));
  u[1] = sub(center, tc);
  u[2] = add(center, mc);
  u[3] = sub(add(center, tc), mc);

  const GF31 m6 = aevumPfa7InvMul(add(b3, b1), (Z31)876673178U);
  const GF31 tq = add(aevumPfa7InvMul(add(b2, b3), (Z31)348117295U), m6);
  const GF31 sq = add(aevumPfa7InvMul(sub(b1, b2), (Z31)922693174U), m6);
  const GF31 q0 = aevumPfa7InvMul(add(sub(b2, b3), b1), (Z31)769204013U);
  const GF31 q1 = sub(q0, tq), q2 = add(q0, sq), q3 = sub(sub(sq, q0), tq);
  u[6] = sub(u[1], q1); u[1] = add(u[1], q1);
  u[5] = sub(u[2], q2); u[2] = add(u[2], q2);
  u[4] = sub(u[3], q3); u[3] = add(u[3], q3);
}
#endif

#if NTT_GF61
GF61 OVERLOAD aevumPfa7InvMul(GF61 a, Z61 s) {
  return U2(mul(a.x, s), mul(a.y, s));
}

// Inverse Nussbaumer DFT with 1/7 incorporated in all eight constants.
// C1=-1/6 gives output[0] = input[0] - 6*center after center is scaled by 1/7.
void OVERLOAD aevumPfa7InverseNormalized(GF61 *u) {
  const GF61 x0 = u[0];
  const GF61 a1 = add(u[1], u[6]), a2 = add(u[2], u[5]), a3 = add(u[3], u[4]);
  const GF61 b1 = sub(u[1], u[6]), b2 = sub(u[2], u[5]), b3 = sub(u[3], u[4]);
  const GF61 total = add(add(a1, a2), a3);
  const GF61 m2 = aevumPfa7InvMul(sub(a1, a3), (Z61)460331295880825166UL);
  const GF61 tc = add(aevumPfa7InvMul(sub(a2, a3), (Z61)914392670108277097UL), m2);
  const GF61 mc = add(aevumPfa7InvMul(sub(a2, a1), (Z61)1374723965989102263UL), m2);
  const GF61 center = add(aevumPfa7InvMul(x0, (Z61)PFA7_INV7_61),
      aevumPfa7InvMul(total, (Z61)54901024028897475UL));
  // The pre-fold value lies in [M61-42, 8*M61+7]; the upper endpoint
  // is UINT64_MAX. One fold restores the existing [0,M61+7] range.
  const GF61 twice = center + center;
  u[0] = modM61q(x0 - (twice + twice + twice), 7);
  u[1] = sub(center, tc);
  u[2] = add(center, mc);
  u[3] = sub(add(center, tc), mc);

  const GF61 m6 = aevumPfa7InvMul(add(b3, b1), (Z61)701497659821917741UL);
  const GF61 tq = add(aevumPfa7InvMul(add(b2, b3), (Z61)394720023356857180UL), m6);
  const GF61 sq = add(aevumPfa7InvMul(sub(b1, b2), (Z61)1209625326034919030UL), m6);
  const GF61 q0 = aevumPfa7InvMul(add(sub(b2, b3), b1), (Z61)1593504021240117648UL);
  const GF61 q1 = sub(q0, tq), q2 = add(q0, sq), q3 = sub(sub(sq, q0), tq);
  u[6] = sub(u[1], q1); u[1] = add(u[1], q1);
  u[5] = sub(u[2], q2); u[2] = add(u[2], q2);
  u[4] = sub(u[3], q3); u[3] = add(u[3], q3);
}
#endif
