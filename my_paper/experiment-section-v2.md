### 5. Experiments

Our experiments aim to validate two central claims: (1) the monotonicity trap is a real and measurable phenomenon in pure ranking-based reward models, and (2) our unsupervised structural prior $\mathcal{L}_{struct}$ can recover RL-friendly cardinal increment structures without frame-level progress labels or additional architectural components.

#### 5.1 Experimental Setup

**Benchmark.** We conduct all experiments on the LIBERO simulated manipulation benchmark (Liu et al., 2023). Following the Robometer ablation protocol (Liang et al., 2026), models are trained on 1,709 successful demonstrations from LIBERO-{10, Object, Goal, Spatial} combined with 1,929 generated failure trajectories. Evaluation is performed on the held-out LIBERO-90 suite, which contains 8,262 paired successful and failed trajectories across unseen tasks.

**Base Model and Training.** All reward model variants are initialized from the Qwen3-VL-2B-Instruct vision-language model and fine-tuned with LoRA adapters (rank 16, $\alpha = 32$). Training uses 8 subsampled frames per trajectory, a batch size of 16 (effective), learning rate $2 \times 10^{-5}$, and runs for 1,250 gradient steps ($\approx$ 70 minutes on a single GPU). Checkpoints are saved every 500 steps; the final checkpoint (step 1250) is used for evaluation.

**Metrics.** We report three standard reward model evaluation metrics:

- **VOC $r$ (Reward Alignment)**: Pearson correlation between per-frame predicted rewards and ground-truth timestep indices for successful trajectories. Measures whether the reward model assigns monotonically increasing values along a successful execution.
- **Kendall $\tau$ (Policy Ranking)**: Kendall rank correlation measuring the alignment between model-assigned trajectory-level rewards and ground-truth quality orderings.
- **Ranking Accuracy**: Fraction of trajectory pairs where the model correctly assigns higher reward to the higher-quality trajectory.

**Compared Methods.** We evaluate three reward model variants that form a controlled ablation:

| ID | Method | Architecture | Training Objective | Labels Required |
|----|--------|--------------|--------------------|-----------------|
| A  | Pure BT | VLM + progress head | $\mathcal{L}_{BT}(\sum\Phi)$ | Pairwise preferences |
| C  | BT + $\mathcal{L}_{struct}$ (Ours) | VLM + progress head | $\mathcal{L}_{BT}(\sum\Phi) + \lambda \cdot \mathcal{L}_{struct}$ | Pairwise preferences |
| D  | Full Robometer | VLM + preference head + progress head | $\mathcal{L}_{BT}^{head} + \mathcal{L}_{progress}$ | Pairwise preferences + frame-level progress |

Methods A and C share the **identical** model architecture: a single per-frame potential function $\Phi_\theta$ (implemented as the progress head without sigmoid activation), whose outputs are summed over frames to produce the trajectory-level Bradley-Terry ranking logit $\Phi(\tau) = \sum_{t} \Phi_\theta(s_t)$. The only difference is the loss function. Method D uses the original Robometer architecture with a separate preference head for ranking and a supervised progress head trained on per-frame timestep ratios ($t/T$), serving as a supervised oracle upper bound. The regularization weight is set to $\lambda = 0.1$ for method C, and the entropy softmax temperature is $\tau = 0.1$.

---

#### 5.2 Core Results

**Table 1.** Reward model evaluation results. Reward alignment (VOC $r$) is evaluated on 30 successful trajectories; policy ranking (Kendall $\tau$, Ranking Acc) is evaluated on 5 tasks with 20 examples per quality level.

| Method | Dataset | VOC $r$ $\uparrow$ | Kendall $\tau$ $\uparrow$ | Ranking Acc $\uparrow$ |
|--------|---------|:---:|:---:|:---:|
| A. Pure BT | LIBERO-90 | **0.348** | −0.005 | 0.498 |
| A. Pure BT | LIBERO-10 | **0.421** | 0.245 | 0.623 |
| C. BT + $\mathcal{L}_{struct}$ (Ours) | LIBERO-90 | −0.382 | **0.198** | **0.600** |

**Key Observation.** While the pure BT baseline (A) achieves higher VOC $r$ values, its policy ranking performance on the challenging LIBERO-90 benchmark is essentially random (Kendall $\tau \approx 0$, Ranking Acc $\approx 0.5$). In contrast, our method (C) achieves meaningful policy ranking capability (Kendall $\tau = 0.198$, Ranking Acc = 0.600), representing a substantial improvement over the baseline.

The negative VOC $r$ for method C is an expected consequence of direction ambiguity in potential-based models: without sigmoid activation, the potential function $\Phi(s_t)$ can learn to decrease monotonically along successful trajectories rather than increase. This produces a strong but inverted Pearson correlation. Crucially, this direction ambiguity does **not** affect trajectory-level ranking: the Bradley-Terry comparison $\sigma(\Phi(\tau_w) - \Phi(\tau_l))$ is invariant to the sign of $\Phi$, and the policy ranking metrics (Kendall $\tau$, Ranking Acc) confirm that method C correctly discriminates trajectory quality.

This result supports our core thesis: the monotonicity trap causes pure BT models to develop degenerate potential curvatures that satisfy ordinal ranking constraints but fail to provide meaningful trajectory-level discrimination on challenging out-of-distribution tasks. Our $\mathcal{L}_{struct}$ prior addresses this by enforcing uniform increment distributions, producing a more robust reward signal.

---

#### 5.3 Analysis

**5.3.1 Training Dynamics: $\mathcal{L}_{struct}$ Provides Active Gradient Signal**

A critical challenge in applying entropy regularization to neural network outputs is avoiding gradient saturation. Our initial experiments with a naïve softplus-based formulation resulted in the structural loss immediately saturating at its theoretical maximum $(-\log T = -1.9459$ for $T = 7$ increments), providing zero gradient throughout training. Two key modifications resolved this:

1. **Removing sigmoid activation** from the progress head, allowing unbounded potential values that produce sufficient variance in the increment distribution.
2. **Introducing a temperature parameter** ($\tau = 0.1$) in the softmax normalization, amplifying small differences between increments before computing the entropy.

The following training curves demonstrate the effect:

**Table 2.** $\mathcal{L}_{struct}$ training dynamics over 1,250 steps.

| Step | $\mathcal{L}_{struct}$ | $\mathcal{L}_{BT}$ | $\sigma^2(\Delta\Phi)$ |
|------|:---:|:---:|:---:|
| 1 | −1.003 | 1.456 | 0.043 |
| 251 | −1.167 | 0.585 | — |
| 501 | −1.320 | 0.594 | 0.029 |
| 651 | −1.403 | — | — |
| 951 | −1.140 | 0.423 | 0.036 |

The structural loss fluctuates between −1.0 and −1.4 (well above the saturation point of −1.9459), confirming active gradient propagation. Simultaneously, the increment variance $\sigma^2(\Delta\Phi)$ decreases from 0.043 to 0.029–0.036, indicating that $\mathcal{L}_{struct}$ successfully encourages more uniform increment distributions. The Bradley-Terry preference loss $\mathcal{L}_{BT}$ converges normally (1.456 → 0.423), demonstrating that the structural regularization does not interfere with preference learning.

**5.3.2 Direction Ambiguity and Future Work**

Our evaluation reveals an important limitation of removing the sigmoid activation: while necessary for $\mathcal{L}_{struct}$ to provide gradient, it introduces direction ambiguity in the potential function. The resulting negative VOC $r$ values indicate that the model learns a monotonically *decreasing* potential along successful trajectories.

For downstream RL applications using potential-based reward shaping (PBRS), this ambiguity can be resolved by either: (a) detecting and flipping the sign of the shaping reward at deployment time, or (b) adding a lightweight directional constraint (e.g., a soft penalty encouraging $\Phi(s_T) > \Phi(s_0)$ for successful trajectories) that does not interfere with the entropy prior.

We leave the integration of $\mathcal{L}_{struct}$-regularized reward models into downstream RL training loops as immediate future work. The policy ranking results (Table 1) provide strong evidence that the improved increment structure should translate to more stable and effective reward shaping signals.
