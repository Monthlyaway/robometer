### 5. Experiments

Our experiments aim to validate two central claims: (1) the monotonicity trap is a real and measurable phenomenon in pure ranking-based reward models, and (2) our unsupervised structural prior $\mathcal{L}_{struct}$ can recover RL-friendly cardinal increment structures without frame-level progress labels.

#### 5.1 Experimental Setup

**Benchmark.** We conduct all experiments on the LIBERO simulated manipulation benchmark (Liu et al., 2023). Following the Robometer ablation protocol (Liang et al., 2026), models are trained on 1,709 successful demonstrations from LIBERO-{10, Object, Goal, Spatial} combined with 1,929 generated failure trajectories. Evaluation is performed on the held-out LIBERO-90 suite, which contains 8,262 paired successful and failed trajectories across unseen tasks.

**Base Model and Training.** All reward model variants are initialized from the Qwen3-VL-4B-Instruct vision-language model and fine-tuned with LoRA adapters. Training uses 8 subsampled frames per trajectory, a batch size of 8, learning rate $2 \times 10^{-5}$ with 10% warmup, and runs for 5,000 gradient steps. Evaluation is performed every 250 steps.

**Metrics.** We report three standard reward model evaluation metrics on LIBERO-90:

- **VOC $r$ (Reward Alignment)**: Value Order Correlation, the Pearson correlation between per-frame predicted rewards and ground-truth timestep indices for successful trajectories. Measures whether the reward model assigns monotonically increasing values along a successful execution.
- **Kendall $\tau$ (Policy Ranking)**: Kendall rank correlation coefficient measuring the alignment between model-assigned trajectory-level rewards and ground-truth quality orderings (failed < suboptimal < successful).
- **Suc--Fail Diff**: The average difference in final predicted reward between successful and failed trajectories of the same task. Larger values indicate clearer discrimination.

**Compared Methods.** We evaluate four reward model variants that form a controlled ablation:

| ID | Method | Training Objective | Labels Required |
|----|--------|--------------------|-----------------|
| A  | Pure BT | $\mathcal{L}_{BT}$ only | Pairwise preferences |
| B  | BT + $L_2$ Smooth | $\mathcal{L}_{BT} + \lambda \cdot L_2(\Delta\Phi)$ | Pairwise preferences |
| C  | BT + $\mathcal{L}_{struct}$ (Ours) | $\mathcal{L}_{BT} + \lambda \cdot \mathcal{L}_{struct}$ | Pairwise preferences |
| D  | Full Robometer | $\mathcal{L}_{BT} + \mathcal{L}_{progress}$ | Pairwise preferences + frame-level progress |

Methods A--C use only trajectory-level pairwise preference labels. Method D additionally requires dense, per-frame absolute progress annotations (timestep ratio $t/T$), serving as a supervised oracle upper bound. The regularization weight is set to $\lambda = 0.1$ for methods B and C.

---

#### 5.2 Core Results

**Table 1.** Reward model evaluation on LIBERO-90. Models are trained on LIBERO-{10, Object, Goal, Spatial}; LIBERO-90 is held out for evaluation only.

| Method | VOC $r$ $\uparrow$ | Kendall $\tau$ $\uparrow$ | Suc--Fail Diff $\uparrow$ |
|--------|:-------------------:|:-------------------------:|:-------------------------:|
| A. Pure BT | [TBD] | [TBD] | [TBD] |
| B. BT + $L_2$ Smooth | [TBD] | [TBD] | [TBD] |
| C. BT + $\mathcal{L}_{struct}$ (Ours) | [TBD] | [TBD] | [TBD] |
| D. Full Robometer | [TBD] | [TBD] | [TBD] |

The core comparison lies between rows A and C: both use identical pairwise preference data, but C additionally applies our unsupervised entropy prior. Row B serves as a naive regularization baseline, demonstrating that generic $L_2$ temporal smoothing is insufficient. Row D provides the supervised upper bound using per-frame progress labels.

We expect method C (ours) to substantially outperform the unregularized baseline A and the naive smoothing baseline B on all three metrics, while approaching the performance of the fully supervised oracle D -- validating our claim that $\mathcal{L}_{struct}$ recovers an RL-friendly potential curvature without privileged labels.

---

#### 5.3 Analysis

**5.3.1 Diagnosing the Monotonicity Trap**

To empirically validate the monotonicity trap described in Section 3.3, we visualize the learned potential functions from models A (Pure BT) and D (Full Robometer) on held-out LIBERO-90 successful trajectories.

<!-- FIGURE 1: Potential Curve Visualization
Left panel: Phi(s_t) vs. t for Pure BT (model A) -- expected to show pathological step-function
or logarithmic curvature despite correct ordinal ordering.
Right panel: Phi(s_t) vs. t for Full Robometer (model D) -- expected to show smooth,
approximately linear progression.
Additional panel: Phi(s_t) vs. t for BT + L_struct (model C) -- expected to approximate model D.
-->

**Figure 1.** Potential function $\Phi(s_t)$ over time for representative successful trajectories from LIBERO-90. [TBD: Insert after training completes]

<!-- FIGURE 2: Increment Distribution Histograms
Histograms of delta_t = Phi(s_{t+1}) - Phi(s_t) for models A, C, and D.
Pure BT (A): expected to show extreme concentration near 0 with occasional large spikes.
BT + L_struct (C): expected to show approximately uniform distribution.
Full Robometer (D): expected to show smooth distribution.
-->

**Figure 2.** Distribution of per-step increments $\Delta\Phi_t$ across all evaluation trajectories. [TBD: Insert after training completes]

Pure BT models are expected to exhibit pathologically concentrated increment distributions (most increments near zero with rare large spikes), confirming the monotonicity trap. Our $\mathcal{L}_{struct}$ regularizer should produce a significantly more uniform increment distribution, approaching that of the fully supervised model.

**5.3.2 Hyperparameter Sensitivity**

We analyze the sensitivity of our method to the regularization weight $\lambda$ by training model C with $\lambda \in \{0.01, 0.1, 1.0\}$.

<!-- FIGURE 3: Lambda Sensitivity
Two line plots:
(a) VOC r vs. lambda
(b) Kendall tau vs. lambda
Expected: performance peaks around lambda=0.1, degrades for very small (reverts to Pure BT)
and very large (over-constrains ranking fidelity) values.
-->

**Figure 3.** Effect of regularization weight $\lambda$ on (a) VOC $r$ and (b) Kendall $\tau$. [TBD: Insert after lambda sweep completes]

| $\lambda$ | VOC $r$ | Kendall $\tau$ | Suc--Fail Diff |
|:---------:|:-------:|:--------------:|:--------------:|
| 0.01      |  [TBD]  |     [TBD]      |     [TBD]      |
| 0.1       |  [TBD]  |     [TBD]      |     [TBD]      |
| 1.0       |  [TBD]  |     [TBD]      |     [TBD]      |

We expect the method to be robust within a moderate range of $\lambda$, with performance degrading when $\lambda$ is too small (insufficient regularization, reverting to Pure BT) or too large (over-constraining the ranking loss).

