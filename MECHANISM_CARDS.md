# Mechanism cards v1 — vehicle (seed 0, 761 train / 85 test)

Status: behavioral + attention + probe-lens evidence. No patching yet (causal confirmation pending).
Weight caveat: TabPFN test-query weights reconstructed manually (ssmax applied, /8 scale assumed);
relative patterns trustworthy, absolute mass less so. TabICL weights exact (post-RoPE/ssmax Q/K).

## Card: TabICL on vehicle (acc 0.871)

- Closest reference algorithm: sharp kNN-like lookup (final same-label mass 0.59, eff. rows 362).
- Convergence: probe-lens accuracy 0.812 (L0) → 0.894 (L11), monotonic-ish.
- Attention: diffuse→sharp→diffuse→sharp across 12 blocks (L8 re-broadens: mass 0.29, eff 644).
- Feature selection: n/a (row-attention only in this card).
- Failure signature: opel/saab same-label mass stuck at 0.48–0.55 vs 0.65–0.68 bus/van at EVERY layer.
- Readout gap: final probe 0.894 > model 0.871 — decoder is the bottleneck, not representations.
- One-sentence mechanism: sharp same-class lookup that cannot resolve confusable classes,
  with additional loss in the readout head.

## Card: TabPFN-3.5 on vehicle (acc 0.918)

- Closest reference algorithm: diffuse prototype aggregation (final same-label mass 0.43, eff. 444,
  uniform across classes 0.42–0.45).
- Convergence: probe-lens volatile early (opel probe 0.00 at L1, rebuilds by L8), 0.87–0.89 late.
- Attention: near-uniform (mass ≈0.25 ≈ chance) through L16, mild sharpening L17–19 (0.31→0.43).
- Failure signature: none on vehicle — wins every class slice; weakest slice saab still 0.77+ recall.
- Readout gap: model 0.918 > final probe 0.894 — decoder ADDS value beyond linear readout.
- One-sentence mechanism: diffuse aggregation over the full context with a strong decoder that
  resolves confusable classes the representations alone do not linearly separate.

## Contrast (the publishable claim)

"Sharper attention = better model" is REJECTED on this pair: the winner attends more diffusely
and more uniformly. The two models place the classification work in opposite stages
(TabICL: representation-rich/decoder-poor; TabPFN: representation-diffuse/decoder-strong).
Next: activation patching on opel/saab rows for causal confirmation (plan §7 order: patching
before any SAE/circuit claim).
