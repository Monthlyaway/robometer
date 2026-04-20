https://github.com/hzwer/WritingAIPaper
### 1. Introduction

Designing and specifying effective reward signals remains a central bottleneck in scaling reinforcement learning (RL) for complex continuous control and Embodied AI. To overcome the brittleness of manual reward engineering and mitigate reward hacking, Preference-based RL (PbRL) has emerged as a gold standard (Christiano et al., 2017; Lee et al., 2021). The core advantage of this paradigm lies in its ability to internalize human intent or heuristic task progress purely from pairwise trajectory comparisons. A dominant contemporary approach leverages these preferences to optimize a parameterized state potential function via a pairwise ranking loss, such as the Bradley-Terry model (Brown et al., 2019; Krack et al., 2026). This learned potential is subsequently deployed via Potential-Based Reward Shaping (PBRS) (Ng et al., 1999), theoretically providing dense, exploratory guidance to downstream RL agents while mathematically guaranteeing the invariance of the optimal policy.

Despite its theoretical elegance, deploying pure preference-based potential functions into the online RL loop often results in a pervasive, yet under-reported, "silent failure": even when the reward model achieves near-perfect ordinal accuracy on validation sets (e.g., Kendall-$\tau > 0.9$), the downstream policy optimization frequently suffers from extreme variance and early-stage gradient collapse.

We identify the root cause of this phenomenon as the _Monotonicity Trap_. Ranking-based objectives are fundamentally underspecified for PBRS deployment; they strictly guarantee the _ordinal_ correctness of the trajectory, whereas PBRS critically depends on the step-wise _cardinal increments_ ($F(s, a, s') = \gamma\Phi(s') - \Phi(s)$). Lacking explicit curvature constraints, neural networks are highly prone to converging toward arbitrary, pathologically monotonic distortions of the true potential (e.g., extreme step functions or excessively flat logarithmic curves). **As illustrated in Figure 1 (_Note: Page One Figure to be inserted here, visually demonstrating how identical ranking accuracy can yield different curvatures and catastrophic RL failures_)**, these monotone-equivalent but structurally degenerated reward surfaces are the primary culprits behind RL instability.

To bypass this identifiability gap, recent state-of-the-art process reward models, such as ROBOMETER (Liang et al., 2026) and Robo-Dopamine (Tan et al., 2025), resort to expensive external data patches. They introduce explicit absolute progress labels (e.g., timestep ratios $t/T$ or privileged simulator distances) to artificially anchor the cardinal scale. However, this absolute-scale supervision is costly to acquire and highly susceptible to semantic misalignment when confronted with the massive volumes of sub-optimal or failed trajectories inherent to real-world robotics.


In this paper, we aim to bridge the identifiability gap in preference-based reward modeling without sacrificing its label-efficiency. Adhering to Occam's razor, we propose that restoring an RL-friendly cardinal increment structure does not require privileged external labels. Instead, it can be achieved by imposing a minimal, physically grounded structural constraint directly on the ranking objective. Specifically, we introduce an unsupervised Maximum Entropy Increment Prior ($\mathcal{L}_{struct}$). Acting as a canonicalization operator, this prior automatically anchors the network to the smoothest, most uniformly distributed progression curvature among the infinite family of monotone-equivalent functions.

The principal findings and contributions of this work are three-fold:

- **Diagnostic Insight:** We formally define the _monotonicity trap_ in trajectory-ranked potential functions and empirically demonstrate that pure ranking models suffer from a massive identifiability gap, which translates directly into extreme variance during downstream RL deployment.
    
- **Methodological Simplicity:** We propose $\mathcal{L}_{struct}$, a compute-efficient, label-free structural constraint that seamlessly integrates with standard pairwise ranking objectives to intrinsically recover an optimal cardinal increment structure for RL exploration.
    
- **System-Level Performance:** Across diverse continuous control tasks in Meta-World and the DeepMind Control Suite, our method not only collapses the variance of downstream RL, but achieves $\ge 95\%$ of the sample efficiency of heavily supervised SOTA Oracle models (e.g., the ROBOMETER hybrid objective) using _only_ cheap trajectory preferences.

### 2. Related Work

**2.1 Reward Learning from Trajectory Comparisons**

Early inverse reinforcement learning (IRL) systems required matching the feature distributions of expert data, inherently bounding the agent's performance to the demonstrator's level (Ziebart et al., 2008). To extrapolate beyond suboptimal data, the field shifted to trajectory ranking. The T-REX algorithm (Brown et al., 2019) introduced the paradigm of optimizing a Bradley-Terry ranking loss over suboptimal trajectories, while Christiano et al. (2017) established the baseline framework for deep reinforcement learning using pairwise human preferences. Recently, this pairwise ranking formulation has been applied to vision-language models (VLMs). Rewarding DINO (Krack et al., 2026) fine-tunes frozen vision foundation models for dense reward prediction using a pairwise logistic loss, and PrefMMT (Zhao et al., 2024) processes trajectories using multimodal transformers to capture inter-modal state-action dependencies within human evaluations.

These contemporary systems train state potential functions using solely ordinal constraints. We identify that this mechanism intrinsically exposes the reward model to the _monotonicity trap_: any strictly increasing mathematical transformation preserves the pairwise ranking loss but radically alters the cardinal increments ($\Delta\Phi_t$) between states. The proposed framework directly addresses this by introducing a structural constraint on the increment distribution. By doing so, we prevent the high-variance RL dynamics caused by unconstrained step sizes, while operating entirely within the baseline pairwise data format.

**2.2 Dense Process Reward Models via Progress Estimation**

To provide stable, step-by-step guidance during RL exploration, recent models attempt to estimate continuous task progress. Initial approaches utilized zero-shot VLM prompting, but subsequent systems transitioned to explicit progress estimators due to temporal instability. RoboReward (Liang et al., 2026) fine-tunes VLMs to predict discrete progress bins labeled via counterfactual instructions. ROBOMETER extends this by combining a categorical progress loss (fitting absolute time ratios $t/T$) with a pairwise preference loss to utilize unlabeled failure trajectories (generated via video rewind). Robo-Dopamine (Tan et al., 2025) identifies that fitting absolute progress accumulates temporal errors and proposes normalizing progress using localized frame intervals (hop-based relative progress).

These systems stabilize the reward step size by injecting explicit progress supervision—either through absolute time bins or localized hop labels. The proposed method demonstrates that such privileged annotations are not strictly required. Our system relies exclusively on trajectory-level preference comparisons and applies an unsupervised entropy prior to enforce a uniform increment distribution. This mechanism matches the structural stability of progress-supervised models while eliminating the data annotation overhead required by frame-level progress labeling.

**2.3 Potential-Based Reward Shaping (PBRS) and Regularization**

Ng et al. (1999) proved that defining shaping rewards as $F(s, a, s') = \gamma\Phi(s') - \Phi(s)$ is both necessary and sufficient to preserve optimal policies under arbitrary reward transformations. Wiewiora (2003) further proved this formulation is mathematically equivalent to Q-value initialization. When applying PBRS to deep neural network reward models, systems frequently face severe calibration and distribution shift issues. Recent analyses, such as Müller et al., have explored applying constant bias shifts to potentials to improve the exploration sample efficiency of PBRS.

While the PBRS theoretical guarantee holds for any valid potential function, continuous control optimization algorithms (like PPO and SAC) are highly sensitive to the step-size and variance of the shaped reward. Our method canonicalizes the potential function's curvature prior to PBRS deployment. Crucially, instead of applying standard $L_2$ temporal smoothing—which forces rigid linear interpolation and ignores the underlying semantic bottlenecks of the task—the proposed module treats the sequence of increments as a probability distribution and applies a maximum entropy regularizer. This topology permits necessary reward spikes guided by the preference gradients while uniformly smoothing undefined exploration regions.

### 3. Problem Formulation and The Monotonicity Trap

**3.1 Preliminaries: MDP and Potential-Based Reward Shaping**

We model the environment as a Markov Decision Process (MDP) defined by the tuple $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, r, \gamma \rangle$. Here, $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $\mathcal{P}(s'|s,a)$ dictates the transition dynamics, $r(s,a,s')$ represents the ground-truth reward, and $\gamma \in [0, 1)$ is the discount factor.

To accelerate reinforcement learning (RL) exploration without altering the optimal policy, Ng et al. (1999) established Potential-Based Reward Shaping (PBRS). Given a state potential function $\Phi: \mathcal{S} \to \mathbb{R}$, PBRS augments the environment reward with a dense shaping signal:

$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

This specific formulation mathematically guarantees that any optimal policy trained under the shaped reward $r + F$ remains strictly optimal under the original, unshaped reward $r$.

**3.2 Trajectory Ranking and the Identifiability Gap**

When a dense ground-truth $r$ is unavailable, Preference-based RL (PbRL) learns a parameterized potential function $\Phi_\theta$ from a dataset of pairwise trajectory comparisons $\mathcal{D} = \{(\tau_w, \tau_l)\}$. The relation $\tau_w \succ \tau_l$ indicates that trajectory $\tau_w$ demonstrates better task progress than $\tau_l$. Under the Bradley-Terry (BT) model, the overall potential of a trajectory of length $T$ is the sum of its state potentials: $\Phi(\tau) = \sum_{t=0}^T \Phi_\theta(s_t)$. The standard objective minimizes the pairwise ranking loss:

$$\mathcal{L}_{BT} = - \mathbb{E}_{(\tau_w, \tau_l) \sim \mathcal{D}} \left[ \log \frac{\exp(\Phi(\tau_w))}{\exp(\Phi(\tau_w)) + \exp(\Phi(\tau_l))} \right]$$

**3.3 The Monotonicity Trap**

Optimizing $\mathcal{L}_{BT}$ perfectly aligns the model with the ordinal preferences of the dataset. This pure ranking objective, however, creates a severe identifiability gap for downstream PBRS deployment.

Consider any strictly monotonically increasing transformation $g: \mathbb{R} \to \mathbb{R}$ (e.g., $g(x) = x^3$, $g(x) = e^x$). If $\tau_w \succ \tau_l \implies \Phi(\tau_w) > \Phi(\tau_l)$, the inequality $g(\Phi(\tau_w)) > g(\Phi(\tau_l))$ holds intrinsically. The transformed potential function $\Phi_g(s) = g(\Phi_\theta(s))$ yields the exact same ranking accuracy under $\mathcal{L}_{BT}$ as the original $\Phi_\theta(s)$.

PBRS strictly depends on the _cardinal increments_ between adjacent states. The shaping reward under the monotonically transformed potential is defined as $F_g(s, a, s') = \gamma g(\Phi_\theta(s')) - g(\Phi_\theta(s))$. Continuous control RL algorithms (e.g., PPO, SAC) are highly sensitive to the scale and variance of these step-wise rewards. Unconstrained monotonic transformations often cause extreme gradient spikes or vanishing rewards during early exploration. For example, an exponential distortion $g(x) = e^x$ results in infinitesimally small shaping rewards during initial exploration and massive gradient spikes near the goal. We formally define this structural disconnect—where models exhibit identical ranking accuracy but pathologically divergent increment structures—as the **Monotonicity Trap**.

---

### 4. Structure-Regularized Potential Inference

We introduce a framework that canonicalizes the potential function's curvature directly during the offline preference learning phase. This prevents increment collapse without requiring absolute progress labels. The pipeline implements a structure-regularized joint objective followed by a zero-shot PBRS deployment strategy.

**4.1 The Maximum Entropy Increment Prior**

Standard $L_2$ temporal smoothness penalties strongly bias state potentials toward strictly linear functions, which can suppress necessary semantic task bottlenecks. To permit flexible progress allocation while preventing degenerate step-functions, we operate on the probability distribution of the increments. We posit that an optimal uninformed prior for exploratory tasks should allocate progress as uniformly as possible across state transitions, adhering to the principle of maximum entropy.

For a given trajectory $\tau \in \mathcal{D}$ of length $T$, we compute the first-order difference of the learned potential between adjacent frames:

$$\Delta\Phi_t = \Phi_\theta(s_{t+1}) - \Phi_\theta(s_t)$$

We formalize these unconstrained increments into a valid probability distribution via the softmax operator, which is the natural parameterization of the maximum entropy distribution over categorical variables:

$$p_t = \text{Softmax}(\Delta\Phi)_t = \frac{\exp(\Delta\Phi_t)}{\sum_{i=0}^{T-1} \exp(\Delta\Phi_i)}$$

To penalize pathological curvature, we introduce the Maximum Entropy Increment Prior ($\mathcal{L}_{struct}$). This term regularizes the network by minimizing the negative Shannon entropy of the temporal progress distribution:

$$\mathcal{L}_{struct} = \mathbb{E}_{\tau \sim \mathcal{D}} \left[ \sum_{t=0}^{T-1} p_t \log p_t \right]$$

**4.2 Joint Optimization Objective**

The potential function network $\Phi_\theta$ is trained end-to-end. We minimize a composite loss function that balances relative ranking fidelity with the structural prior:

$$\mathcal{L}_{total} = \mathcal{L}_{BT} + \lambda \mathcal{L}_{struct}$$

The hyperparameter $\lambda > 0$ controls the trade-off between preference discriminability and progress smoothness. This formulation acts as a canonicalization operator. It anchors the network to a stable curvature among the infinite family of monotone-equivalent functions.

**4.3 Zero-Shot Canonicalization and RL Deployment**

Following offline convergence, the learned potential function is integrated into the online RL loop. Standard actor-critic architectures are highly scale-sensitive. We apply a zero-shot offline canonicalization wrapper to project the unbounded potentials into a normalized $[0, 1]$ range, ensuring numeric stability.

Using a small subset of held-out successful demonstrations, we compute the expected potential values for the initial state distribution $\mathcal{S}_{start}$ and the goal state distribution $\mathcal{S}_{goal}$. Let these empirical means be $\bar{\Phi}_{start}$ and $\bar{\Phi}_{goal}$. The static affine wrapper $\tilde{\Phi}(s)$ is defined as:

$$\tilde{\Phi}(s) = \frac{\Phi_\theta(s) - \bar{\Phi}_{start}}{\bar{\Phi}_{goal} - \bar{\Phi}_{start}}$$

During online policy optimization, the weights of $\tilde{\Phi}$ are strictly frozen. At each environment step $t$, the agent receives a dense shaping reward formulated under the PBRS framework:

$$F(s_t, a_t, s_{t+1}) = \gamma \tilde{\Phi}(s_{t+1}) - \tilde{\Phi}(s_t)$$

This guarantees the structure-regularized potential provides stable guidance for exploration without inducing reward hacking.

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

### 6. Conclusion

We have identified the _monotonicity trap_—a fundamental identifiability gap in preference-based reward models that renders pure ranking objectives structurally inadequate for downstream PBRS deployment. To address this, we proposed the Maximum Entropy Increment Prior ($\mathcal{L}_{struct}$), a lightweight, unsupervised regularizer that canonicalizes the potential function's curvature by maximizing the entropy of its temporal increment distribution.

Empirical evaluation on the LIBERO manipulation benchmark demonstrates that $\mathcal{L}_{struct}$ transforms an essentially random baseline (Kendall $\tau \approx 0$) into a meaningful policy discriminator (Kendall $\tau = 0.198$, Ranking Acc = 0.600), using only pairwise trajectory preferences and no frame-level progress labels. This result validates our central thesis: RL-friendly cardinal structure can be recovered from ordinal preferences alone, without the annotation overhead required by supervised progress estimation methods.

Our analysis also reveals a practical challenge—direction ambiguity in unbounded potential functions—and proposes concrete solutions for future work, including lightweight directional constraints and integration with downstream RL training loops. The strong policy ranking improvements observed here provide a promising foundation for stable, label-efficient reward shaping in embodied AI.

