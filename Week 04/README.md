# Week 04: attention experiments and Week 03 wrap-up

This directory contains the attention notebook and the documentation completing the Week 03 internship-preparation learning objectives. Related neuronal dynamics notebooks remain in `Week 03`. Week 04 learning builds on this baseline.

## Files

- [Izhikevich model](../Week%2003/Izhekevich%20Model.ipynb): neuron simulation, firing behavior, nullclines, and trajectories.
- [Hopf bifurcation](../Week%2003/Hopf%20bifurcation.ipynb): current sweeps, initial-condition comparisons, and local phase portraits.
- [Hopf continuation](../Week%2003/Hopf_Bifurcation_continued.ipynb): Euler drift, return-crossing interpolation, Richardson extrapolation, and cubic displacement scaling.
- [Softmax and attention notebook](LLM%20Tokens.ipynb): probability, temperature, negative log-likelihood examples, and causal attention.
- [Attention research note](Attention_Research_Note.md): paper question, method, evidence, limitation, equation explanation, and research question.

## Running the notebooks

Use a Python 3 Jupyter environment with NumPy and Matplotlib installed. Select that environment as the notebook kernel, restart the kernel, and run all cells in order. These are small CPU examples; no GPU or model download is needed. Examples use manually specified deterministic inputs. Printed floating-point results may differ slightly across environments. Exact package versions are not pinned.

## Attention baseline and expected result

The notebook defines X = [[1,0,1,0], [0,1,0,1], [1,1,0,0]], W_Q = [[1,0], [0,1], [1,0], [0,1]], W_K = [[1,0], [0,1], [0,1], [1,0]], and W_V = [[1,0,0], [0,1,0], [0,0,1], [1,1,0]].

With the causal mask, the expected attention weights are [[1,0,0], [1/2,1/2,0], [1/3,1/3,1/3]]. The expected output is [[1,0,1], [1,1,1/2], [1,1,1/3]]. The function returns (A, S, O), where S is the unmasked score matrix.

Checks performed during learning and independent verification: row sums equal one within floating-point tolerance; weights above the diagonal equal zero; three- and four-position inputs work; changing the final input to [2,0,0,0] leaves earlier outputs unchanged while changing the final output. Some checks were interactive and are not stored as automated test cells.

## Neuronal experiment and interpretation

For the Izhikevich parameters a=0.02 and b=0.2, the local Hopf threshold is I_H=3.7975. The continuation studies the transformed smooth equations Xdot=-0.06Y+0.04X^2 and Ydot=0.06X-(0.04/3)X^2 near that equilibrium.

A neutral linear oscillator exposes Euler's artificial outward drift. The nonlinear experiment measures displacement after one revolution using an interpolated positive-axis return crossing. Step halving and first-order Richardson extrapolation separate the leading integration error from the physical displacement. The finest reported extrapolated displacements are approximately 0.000232088 for initial radius 0.1 and 0.0000290156 for radius 0.05, a ratio near eight, consistent with cubic scaling.

Numerical observations support the analysis but are not a proof. The local smooth calculation does not determine global reset-based firing behavior. One older notebook comment still labels the smaller-radius check as a next step, although the later cell performs it.

## Week 03 completion record

- Probability vectors, array shapes, and softmax: completed.
- Queries, keys, values, and weighted aggregation: completed.
- Neuronal nullclines, equilibria, stability, and parameter comparisons: completed in the recorded sessions.
- Causal masking, prediction loss, and evaluation limitations: completed.
- Paper discussion, explained equation, and research question: recorded in the research note.
- Numerical sensitivity and distinction between evidence and proof: completed.
- Fresh-kernel rerun, settings, README, and limitations: completed.

Optional LoRA work has not been performed and is not counted as completed. This record covers the internship-preparation curriculum, not the other research and teaching blocks in the weekly timetable.

## Next: Week 04, Day 1

Use a small deterministic baseline whose inputs, expected output, and correctness check can be explained. Our attention baseline already supplies a foundation. A tiny text-to-vector example can extend it with a toy tokenizer, embedding lookup, and eventually vocabulary scores; completing that extension is not claimed here.
