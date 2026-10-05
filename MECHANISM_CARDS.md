# Mechanism cards v1 — vehicle (seed 0, 761 train / 85 test)

Status: behavioral + attention + probe-lens + CAUSAL (head ablation) evidence for TabICL.
Weight caveat: TabPFN test-query weights reconstructed manually (ssmax applied, /8 scale assumed);
relative patterns trustworthy, absolute mass less so. TabICL weights exact (post-RoPE/ssmax Q/K).

## Card: TabICL on vehicle (acc 0.871)

- Closest reference algorithm: sharp kNN-like lookup (final same-label mass 0.59, eff. rows 362).
- Convergence: probe-lens accuracy 0.812 (L0) → 0.894 (L11), monotonic-ish.
- Attention: diffuse→sharp→diffuse→sharp across 12 blocks (L8 re-broadens: mass 0.29, eff 644).
- Feature selection: n/a (row-attention only in this card).
- Failure signature: opel/saab same-label mass stuck at 0.48–0.55 vs 0.65–0.68 bus/van at EVERY layer.
- Readout gap: final probe 0.894 > model 0.871 — decoder is the bottleneck, not representations.
- CAUSAL (12x8 head-ablation sweep, no-op control at base exactly): the opel/saab verdict
  localizes to final-block heads — (L11,H1) ablation: acc 0.741, opel 0.333, bus/van untouched;
  (L11,H4): saab 0.409; (L11,H5/H3/H7): 0.788-0.800. All other blocks/heads ~= base.
  Early layers are causally redundant for the output.
- CAUSAL-REFINED (row-split ablation of L11H1 across all call shapes): zeroing train-side only
  and test-side only EACH collapse opel to 0.333 (acc 0.765 both). The head carries train-test
  RELATIONAL signal, not a test-side readout — breaking either side breaks the verdict.
  (An intermediate "no effect" reading was a shape-filter bug missing 5D sdpa calls; corrected.)
- RETRACTION: a first sweep implicated head 1 in blocks 0-5 (hook-staleness artifact);
  the controlled re-sweep refutes it. Rule adopted: every ablation claim needs a no-op control.
- One-sentence mechanism: sharp same-class lookup whose confusable-class verdict is rendered
  almost entirely by final-block heads 1/4/5, with additional loss in the readout head.

## Card: TabPFN-3.5 on vehicle (acc 0.918)

- Closest reference algorithm: diffuse prototype aggregation (final same-label mass 0.43, eff. 444,
  uniform across classes 0.42–0.45).
- Convergence: probe-lens volatile early (opel probe 0.00 at L1, rebuilds by L8), 0.87–0.89 late.
- Attention: near-uniform (mass ≈0.25 ≈ chance) through L16, mild sharpening L17–19 (0.31→0.43).
- Failure signature: none on vehicle — wins every class slice; weakest slice saab still 0.77+ recall.
- Readout gap: model 0.918 > final probe 0.894 — decoder ADDS value beyond linear readout.
- One-sentence mechanism: diffuse aggregation over the full context with a strong decoder that
  resolves confusable classes the representations alone do not linearly separate.

## Interventions v1 (retrieval-restricted context) — FAILED, informatively

Pre-registered prediction: kNN-local support (k=50/100/200) sharpens same-class neighborhoods,
opel/saab recall rises toward TabPFN, bus/van unaffected. Result (clean env, per-row fits):
k=50: 0.847 (opel 0.667), k=100: 0.847 (opel 0.714), k=200: 0.871, full: 0.871 (opel 0.762).
Gap-closed: NEGATIVE — local context hurts opel, helps bus only marginally (1.000 vs 0.955).
Interpretation: TabICL's opel verdict needs GLOBAL context (class balance/prototypes for the
relational computation at L11); restriction removes the comparisons the final block needs.
This is consistent with, and predicted by, the relational-signal card. Next lever: context
TRANSPLANT (which rows), not restriction (how many).

## Interventions v4 (context-ensembling across compositions) — FAILED held-out validation

Seed 0: ens[opel-heavy, saab-heavy, balanced] = 0.894 vs full 0.871 (gap-closed 49%, no
class degraded: bus 0.955/opel 0.810/saab 0.818/van 1.000). Seed 1 (held-out): ensemble 0.835
vs full 0.882 — LOSES. The composition was tuned to seed 0 and did not transfer.
Validated gap-closed: NEGATIVE. The validation rule caught a split-overfit false positive —
the program working as designed. Standing: steering wheel confirmed (v2), operating-point
frontier confirmed (v3), no passing accuracy intervention yet. Next candidates: per-row
ADAPTIVE composition (kNN-chosen, not fixed menu) or conceding the frontier as the contribution.

## Interventions v3 (joint opel+saab enrichment) — no free lunch, operating-point frontier

both-heavy (35/35/15/15): 0.824, opel 0.667 (DOWN vs random 0.714); confusable-only: 0.435
(bus/van → 0, opel 0.762/saab 0.955); saab-heavy: 0.788 (saab 0.909, opel 0.429).
Nothing beats full natural context overall (0.871). Reframe: support composition is a TUNABLE
OPERATING POINT along a confusable-recall frontier (opel 0.00–0.95 trade vs saab), not an
accuracy maximizer. Deployment-relevant where one class is critical (e.g. fault detection):
opel-heavy buys opel 0.952 at overall 0.788. Next: context-ensembling (average across
compositions) as the only remaining free-lunch candidate.

## Interventions v2 (context TRANSPLANT, fixed n=200) — CAUSAL CONFIRMATION

Varying support composition at fixed size: random (opel 0.714), balanced (0.810),
opel-heavy/50% (0.952), opel-free (0.000). Opel recall tracks opel support share almost
deterministically — the verdict IS the context composition. Seesaw with saab
(1.000 → 0.273): zero-sum between confusables at fixed budget, overall acc 0.847 → 0.788.
Gap-closed on opel vs TabPFN (0.875): 168% (overshoot) but fails the no-degradation rule.
Verdict: steering wheel found; free lunch not found. Next: opel+saab joint enrichment
(preserve both confusables, cut bus/van).

Pre-registered prediction: kNN-local support (k=50/100/200) sharpens same-class neighborhoods,
opel/saab recall rises toward TabPFN, bus/van unaffected. Result (clean env, per-row fits):
k=50: 0.847 (opel 0.667), k=100: 0.847 (opel 0.714), k=200: 0.871, full: 0.871 (opel 0.762).
Gap-closed: NEGATIVE — local context hurts opel, helps bus only marginally (1.000 vs 0.955).
Interpretation: TabICL's opel verdict needs GLOBAL context (class balance/prototypes for the
relational computation at L11); restriction removes the comparisons the final block needs.
This is consistent with, and predicted by, the relational-signal card. Next lever: context
TRANSPLANT (which rows), not restriction (how many).

## Contrast (the publishable claim)

"Sharper attention = better model" is REJECTED on this pair: the winner attends more diffusely
and more uniformly. The two models place the classification work in opposite stages
(TabICL: representation-rich/decoder-poor; TabPFN: representation-diffuse/decoder-strong).
Next: cross-row activation patching at TabICL L11 (bus activations into opel rows) and the
TabPFN-side causal test, then the intervention loop (plan §7 order: patching
before any SAE/circuit claim).
