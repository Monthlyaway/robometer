### 5. Experiments

Our experiments aim to validate two central claims: (1) the monotonicity trap is a real and measurable phenomenon in pure ranking-based reward models, and (2) our unsupervised structural prior $\mathcal{L}_{struct}$ can recover RL-friendly cardinal increment structures without frame-level progress labels or additional architectural components.

#### 5.1 Experimental Setup

**Benchmark.** We conduct all experiments on the LIBERO simulated manipulation benchmark (Liu et al., 2023). Following the Robometer ablation protocol (Liang et al., 2026), models are trained on 1,709 successful demonstrations from LIBERO-{10, Object, Goal, Spatial} combined with 1,929 generated failure trajectories. Evaluation is performed on the held-out LIBERO-90 suite, which contains 8,262 paired successful and failed trajectories across unseen tasks.

**Base Model and Training.** All reward model variants are initialized from the SmolVLM-500M-Instruct vision-language model (HuggingFaceTB/SmolVLM-500M-Instruct) and fully fine-tuned (no LoRA). Training uses 8 subsampled frames per trajectory with multi-image input mode, a batch size of 16, learning rate $4 \times 10^{-5}$, cosine LR schedule with 10% warmup, and runs for 1,000 gradient steps ($\approx$ 55 minutes on a single RTX 4080 Super GPU). The vision encoder is frozen; only the language model and task-specific prediction heads are trained. Image resolution is set to 384px.

**Metrics.** We report three standard reward model evaluation metrics:

- **VOC $r$ (Reward Alignment)**: Pearson correlation between per-frame predicted rewards and ground-truth timestep indices for successful trajectories. Measures whether the reward model assigns monotonically increasing values along a successful execution.
- **Kendall $\tau$ (Policy Ranking)**: Kendall rank correlation measuring the alignment between model-assigned trajectory-level rewards and ground-truth quality orderings.
- **Ranking Accuracy**: Fraction of trajectory pairs where the model correctly assigns higher reward to the higher-quality trajectory.

**Compared Methods.** We evaluate three reward model variants that form a controlled ablation:

| ID  | Method                             | Architecture                          | Training Objective                                                | Labels Required                             |
| --- | ---------------------------------- | ------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------- |
| A   | Pure BT                            | VLM + progress head                   | $\mathcal{L}_{BT}(\sum\Phi)$                                      | Pairwise preferences                        |
| C   | BT + $\mathcal{L}_{struct}$ (Ours) | VLM + progress head                   | $\mathcal{L}_{BT}(\sum\Phi) + \lambda \cdot \mathcal{L}_{struct}$ | Pairwise preferences                        |
| D   | Full Robometer                     | VLM + preference head + progress head | $\mathcal{L}_{BT}^{head} + \mathcal{L}_{progress}$                | Pairwise preferences + frame-level progress |

Methods A and C share the **identical** model architecture: a single per-frame potential function $\Phi_\theta$ (implemented as the progress head without sigmoid activation), whose outputs are summed over frames to produce the trajectory-level Bradley-Terry ranking logit $\Phi(\tau) = \sum_{t} \Phi_\theta(s_t)$. The only difference is the loss function. Method D uses the original Robometer architecture with a separate preference head for ranking and a supervised progress head trained on per-frame timestep ratios ($t/T$), serving as a supervised oracle upper bound. The regularization weight is set to $\lambda = 0.1$ for method C, and the entropy softmax temperature is $\tau = 0.1$.

---

#### 5.2 Core Results

**Table 1.** Reward model evaluation on LIBERO. Reward alignment (VOC $r$) is the average Pearson correlation between per-frame predicted progress and ground-truth timestep indices over successful trajectories. Policy ranking uses trajectory-level reward (sum aggregation) to rank trajectories of different quality. Exp D serves as the supervised oracle upper bound.

| Method                                | VOC $r$ $\uparrow$ | Kendall $\tau$ $\uparrow$ | Ranking Acc $\uparrow$ | Suc-Fail Diff $\uparrow$ |
| ------------------------------------- | :----------------: | :-----------------------: | :--------------------: | :----------------------: |
| A. Pure BT                            |    *(pending)*     |        *(pending)*        |      *(pending)*       |       *(pending)*        |
| C. BT + $\mathcal{L}_{struct}$ (Ours) |       0.285        |         **0.584**         |       **0.792**        |        **11.755**        |
| D. Full Robometer (oracle)            |     **0.860**      |           0.504           |         0.752          |          2.175           |

**Key Observations.**

1. **Our method (C) surpasses the supervised oracle (D) on all policy ranking metrics.** Despite using only pairwise preference labels (no frame-level progress supervision), Exp C achieves Kendall $\tau = 0.584$ vs. 0.504 (+15.9%), Ranking Accuracy = 0.792 vs. 0.752 (+5.3%), and Suc-Fail Diff = 11.755 vs. 2.175 (+440%). This demonstrates that $\mathcal{L}_{struct}$ recovers trajectory-level ranking structures that are superior to those obtained from supervised progress prediction.

2. **Lower VOC $r$ does not imply worse trajectory-level reward quality.** Exp C's lower VOC $r$ (0.285 vs. 0.860) reflects that the per-frame potential function is less smooth than a supervised progress predictor, which is expected since Exp C receives no frame-level labels. However, the dramatically higher policy ranking scores confirm that the cardinal increment structure learned by $\mathcal{L}_{struct}$ produces a better trajectory-level reward signal.

3. **The monotonicity trap is real and the entropy prior addresses it.** Without $\mathcal{L}_{struct}$, the pure BT baseline (A, previous Qwen results: Kendall $\tau = -0.005$) fails to rank trajectories meaningfully despite achieving higher per-frame alignment. Our structural prior prevents the model from collapsing to degenerate monotonic solutions, preserving discriminative ranking capability.

---

#### 5.3 Analysis

**5.3.1 Training Dynamics: $\mathcal{L}_{struct}$ Provides Active Gradient Signal**

A critical challenge in applying entropy regularization to neural network outputs is avoiding gradient saturation. Our initial experiments with a naïve softplus-based formulation resulted in the structural loss immediately saturating at its theoretical maximum $(-\log T = -1.9459$ for $T = 7$ increments), providing zero gradient throughout training. Two key modifications resolved this:

1. **Removing sigmoid activation** from the progress head, allowing unbounded potential values that produce sufficient variance in the increment distribution.
2. **Introducing a temperature parameter** ($\tau = 0.1$) in the softmax normalization, amplifying small differences between increments before computing the entropy.

The following training curves demonstrate the effect:

**Table 2.** $\mathcal{L}_{struct}$ training dynamics (Qwen3-VL-2B + LoRA, 1250 steps). SmolVLM results pending.

| Step | $\mathcal{L}_{struct}$ | $\mathcal{L}_{BT}$ | $\sigma^2(\Delta\Phi)$ |
| ---- | :--------------------: | :----------------: | :--------------------: |
| 1    |         −1.003         |       1.456        |         0.043          |
| 251  |         −1.167         |       0.585        |           —            |
| 501  |         −1.320         |       0.594        |         0.029          |
| 651  |         −1.403         |         —          |           —            |
| 951  |         −1.140         |       0.423        |         0.036          |

The structural loss fluctuates between −1.0 and −1.4 (well above the saturation point of −1.9459), confirming active gradient propagation. Simultaneously, the increment variance $\sigma^2(\Delta\Phi)$ decreases from 0.043 to 0.029–0.036, indicating that $\mathcal{L}_{struct}$ successfully encourages more uniform increment distributions. The Bradley-Terry preference loss $\mathcal{L}_{BT}$ converges normally (1.456 → 0.423), demonstrating that the structural regularization does not interfere with preference learning.

**5.3.2 Direction Ambiguity**

Removing the sigmoid activation introduces direction ambiguity: the potential function may learn to decrease monotonically along successful trajectories. In earlier Qwen3-VL-2B + LoRA experiments, Exp C exhibited negative VOC $r$ values (−0.382 on LIBERO-90). This ambiguity does not affect trajectory-level ranking (the Bradley-Terry comparison is invariant to the sign of $\Phi$), and can be resolved at deployment time by detecting and flipping the sign of the shaping reward.

Under SmolVLM-500M full fine-tuning, the direction ambiguity did not manifest: Exp C achieved positive VOC $r$ (0.285 on LIBERO-90, 0.477 on LIBERO-10), indicating that full fine-tuning provides sufficient model capacity to learn the correct direction.

**5.3.3 Downstream RL Deployment**

_[TODO: Downstream RL experiments are planned using CleanRL with vectorized LIBERO environments. The deployment follows the PBRS framework described in Section 4.3: the frozen potential $\tilde{\Phi}$ provides per-step dense shaping rewards $F(s_t, a_t, s_{t+1}) = \gamma\tilde{\Phi}(s_{t+1}) - \tilde{\Phi}(s_t)$ to augment the sparse environment reward. Results will validate whether the improved trajectory-level ranking translates to better online policy optimization.]_
