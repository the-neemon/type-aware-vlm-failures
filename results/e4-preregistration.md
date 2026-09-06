# E4: pre-registered test for the cross-intervention matrix

Written 2 September, **before any intervention has been run**. TASKS P6.7 asks
for the test to be stated in advance rather than chosen after seeing the matrix,
which is the whole value of this document.

## What "diagonal structure" means

SPEC asks whether the failure-type x intervention matrix has diagonal structure.
That phrase needs sharpening: the matrix is 2x4 rather than square, and
`I_abstain` is scored as risk avoided rather than accuracy gained, so it has no
recovery rate to compare.

Two claims of different strength:

- **A. Interaction.** The effect of an intervention differs by failure type.
- **B. Argmax flip.** The *best* intervention differs by failure type.

B is what the project needs, since a type-aware controller earns nothing over a
single fixed policy unless the argmax actually moves. A is a precondition for B
and is what we have power to test.

## Primary test, pre-registered

A directional difference in differences:

    delta_crop = [P(ok | structural, I_crop)  - P(ok | structural, I_0)]
               - [P(ok | fabrication, I_crop) - P(ok | fabrication, I_0)]

**Prediction: `delta_crop > 0`.** Cropping should rescue structural misreadings,
where the answer was on the figure all along, more than it rescues fabrications,
where it never was. Inference is a 95 percent cluster bootstrap over figures.

Secondary: the same contrast for `I_verify`. Reported, but the primary claim
rests on `I_crop` alone, so no multiplicity correction is applied to it.

`I_abstain` is excluded from all of these deliberately. Abstention produces no
correct answer, so its value is not a recovery rate; it enters E5's
risk-coverage analysis instead. `difference_in_differences` raises rather than
silently returning a number, because comparing it here would mix units.

## Reported alongside, never instead of

`argmax_flip_stability`: the fraction of bootstrap resamples in which the two
failure types select different repairs. An argmax is fragile, and two cells
within noise of each other flip on resampling while the point matrix looks
clean. **Below about 0.9, the flip is unsupported** however good the point
estimates look.

## Power, simulated

400 items per cell, 3 errors per figure, 200 simulated runs, baseline recovery
0.25 for structural and 0.10 for fabrication:

| True delta | Power (CI excludes 0) | Mean estimate |
| --- | --- | --- |
| 0.00 | 0.03 | +0.001 |
| 0.05 | 0.27 | +0.052 |
| 0.10 | 0.70 | +0.101 |
| 0.15 | 0.93 | +0.152 |
| 0.20 | 0.99 | +0.202 |

The estimator is unbiased and the false-positive rate at a true zero is 0.03,
slightly conservative against the nominal 0.05, which is what a cluster
bootstrap should do.

**The minimum detectable effect at 80 percent power is about 0.12.** That is a
large effect: cropping would have to help structural errors by twelve
percentage points more than it helps fabrications.

### How many items would fix it

Power at 150 simulated runs per point:

| Items per cell | delta=0.05 | delta=0.10 | delta=0.15 |
| --- | --- | --- | --- |
| 400 | 0.27 | 0.73 | 0.93 |
| 800 | 0.49 | **0.94** | 1.00 |
| 1600 | **0.81** | 1.00 | 1.00 |
| 3200 | 0.97 | 1.00 | 1.00 |

Detecting a 10-point effect reliably needs about 800 per cell. Detecting a
5-point effect needs about 1600, and that is a different scale of experiment.

## The consequence that matters

**A null E4 at 400 per cell does not mean the taxonomy earns nothing.** It means
no effect larger than roughly 12 points, and we are obliged to report it that
way. At a true effect of 0.05 this design finds nothing 73 percent of the time.

### Synthetic items are free; the errors they produce are not

The obvious fix is "generate more synthetic figures", and that is only half
right. A cell is filled by an *erroneous* answer, not by an item. Generating a
figure costs render time, but turning it into an E4 row costs a VLM forward
pass, and only the fraction the model actually gets wrong counts. At a plausible
synthetic error rate of roughly 30 percent, 800 errors per cell needs on the
order of 2,700 items per failure type per model.

**But E4 needs inference, not activations.** The recovery matrix is built from
outcomes alone; no probe touches it (SPEC 4.3 is explicit that this experiment
does not use the probes). So scaling the synthetic arm for E4 power costs
inference compute and nothing in the activation cache, and the 30 GiB home quota
is not the binding constraint here. Cache activations for the probe subset;
run inference on the larger set for E4.

**Recommended target: 800 errors per failure type per model.** That buys 0.94
power at a 10-point effect for a tractable amount of inference. A 5-point effect
is out of reach for this project and we should say so in the write-up rather
than implying the null was informative at that scale.

## Reproducing

Implementation and tests: `src/intervene/matrix.py`,
`src/intervene/test_matrix.py`. Power simulations:
`scripts/e4_power_simulation.py` and `scripts/e4_power_sweep.py`.
