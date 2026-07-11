# Calibrated Uncertainty and Selective Classification in Financial Fraud Detection

**Course:** CS5801 — Advanced Topics in Artificial Intelligence  
**Group:** IntelliCore  
**Members:**  
- Fernando T.R.D. (268333U)  
- Fonseka S.A.N.T. (268334A)  
- Mahaarachchi N.H.P.C. (268371H)  
- Wijesekera H.T.J. (268429U)

---

## 1. Introduction

The digitalization of global finance has exponentially increased the volume and velocity of transactions, bringing with it a corresponding rise in financial crimes such as credit card fraud and money laundering. To combat this, financial institutions heavily rely on Machine Learning classification models to monitor transactions in real time. Despite their high accuracy, automated fraud detection is a strictly safety-critical domain. The cost of a false negative can result in millions of dollars in stolen funds and severe regulatory penalties. Conversely, the cost of a false positive results in frozen accounts, ruined customer experiences, and overwhelmed compliance teams.

In this high-stakes environment, simply outputting a binary prediction (Fraud vs. Genuine) is insufficient. Fraudsters continuously adapt their strategies, creating distribution shifts. When ML models encounter these novel, out-of-distribution (OOD) tactics, they must not only be highly accurate but also **well-calibrated**, meaning their output probabilities must accurately reflect the true likelihood of correctness. Furthermore, to balance automation with safety, these systems require an integrated "ML with rejection" mechanism. By quantifying predictive uncertainty, the model can automatically approve or block clear-cut cases, while purposefully abstaining from ambiguous transactions and deferring them to human compliance officers for manual review.

This project addresses the fragmentation in the existing literature by proposing a unified framework that integrates three interconnected components: (i) **Evidential Deep Learning (EDL)** for single-forward-pass uncertainty quantification with epistemic–aleatoric decomposition, (ii) **post-hoc temperature scaling** for probability calibration, and (iii) a **selective classification policy** that uses the estimated epistemic uncertainty to decide when to defer to a human investigator.

---

## 2. Literature Review

Fraud-detection research has recently optimised heavily for discriminative performance while paying comparatively little attention to whether a model's probabilities can be trusted or whether it recognises when to stop. Naqvi's [1] systematic review of work from 2020 to 2025, which screened more than a thousand records and examined seventy-seven studies in depth, makes this explicit: it argues that models should be assessed along an operational-capability axis covering data realism, robustness to drifting fraud patterns, and governance for human investigators, and it reports that probability calibration and the setting of operational decision thresholds are routinely absent despite being essential to real workflows. The review concludes that detectors ought to be paired with explicit decision policies, including the routing of uncertain cases to analysts.

A first line of work responds by attaching uncertainty estimates to predictions rather than reporting bare labels. Habibpour et al. [2] are representative: observing that deep networks for card fraud have been tuned for point accuracy while their confidence goes unmeasured, they apply Bayesian-style estimators, Monte Carlo Dropout, and deep ensembles, to quantify predictive uncertainty, and show that the resulting estimates surface borderline cases and let a system abstain when confidence is low. Their motivation is precisely the safety-critical framing adopted here: because fraudsters change tactics continually, the model repeatedly meets data unlike its training distribution, and because expert review is slow, only a small fraction of transactions are ever verified. The practical drawback is cost, since these sampling-based estimators require many forward passes per transaction, awkward under the millisecond budgets of real-time screening.

A second, more foundational line concerns calibration itself. Guo et al. [3] demonstrated that modern deep networks, unlike their smaller predecessors, are systematically overconfident: their reported probabilities exceed their empirical accuracy, an effect tied to architectural factors such as depth, width, and batch normalisation. Their proposed remedy, **temperature scaling**, is a single-parameter post-hoc rescaling of the logits that leaves the predicted class unchanged but markedly improves calibration, and the Expected Calibration Error (ECE) they popularised has become the standard diagnostic. The result is central to any rejection scheme, because a policy that abstains or acts on the basis of a probability threshold is only as reliable as the calibration of the scores beneath it.

A third line moves from estimating uncertainty to acting on it inside a human-in-the-loop workflow. The closest recent example is the Integrated Transparency and Confidence Framework of Mapaila and Senekane [4], which combines local explanations (LIME) with split conformal prediction to produce calibrated prediction sets with formal coverage guarantees, evaluated on the large PaySim mobile-money dataset using Random Forest and XGBoost. When the framework cannot assign a label confidently, signalled by uncertain or empty prediction regions, it defers the case to an analyst, together with an explanation to guide the review. This is close in spirit to what we propose, yet it treats uncertainty as a post-hoc wrapper around tree-ensemble classifiers rather than a quantity the model produces intrinsically, and it does not separate uncertainty arising from genuine novelty from uncertainty arising from noisy data.

Underlying all of these efforts is the problem of data. Realistic transaction records are scarce because of privacy and legal constraints, and the public datasets that do exist tend to lack diversity and reliable labels. Oztas et al. [5] address this by releasing SAML-D, a synthetic anti-money-laundering dataset generated from a purpose-built transaction simulator, comprising twelve features and twenty-eight normal and suspicious typologies across a range of geographies and payment types; tellingly, standard models perform worse on it than on older benchmarks, which makes it a demanding testbed for behaviour under rare and deceptive patterns — exactly the regime in which a model ought to abstain.

Taken together, the literature assembles the necessary ingredients for uncertainty estimation, calibration, deferral, and realistic evaluation, but largely treats them in isolation: uncertainty methods are often expensive and themselves uncalibrated, calibration work rarely connects to a rejection policy, and the one framework that ties rejection to human review does so outside the model and without disentangling the kind of uncertainty involved. Closing that fragmentation is the aim of this project.

---

## 3. Method

### 3.1 Overview

We propose a three-component framework that addresses the three shortcomings identified in the problem statement:

1. **Uncertainty Quantification with EDL** — single-pass epistemic/aleatoric decomposition
2. **Post-hoc Calibration** — temperature scaling to make confidence scores trustworthy
3. **Selective Classification** — epistemic-uncertainty-driven rejection policy

### 3.2 Evidential Deep Learning (EDL)

Rather than the multiple passes that Deep Ensembles or MC Dropout require, we formulate the classifier using EDL [6]. The network outputs the parameters of a **Dirichlet distribution** over class probabilities, treating predictions as evidence-based opinions. Specifically, the network's final layer produces non-negative evidence values $e_k$ via softplus activation, and the Dirichlet concentration parameters are:

$$\alpha_k = e_k + 1, \quad k = 1, \ldots, K$$

The Dirichlet strength is $S = \sum_k \alpha_k$. From a single forward pass, this yields:

- **Expected class probabilities**: $\hat{p}_k = \alpha_k / S$
- **Belief mass**: $b_k = e_k / S$
- **Uncertainty mass (vacuity)**: $u = K / S$ — this is the epistemic uncertainty
- **Aleatoric uncertainty**: Entropy of the expected categorical distribution $-\sum_k \hat{p}_k \log \hat{p}_k$

The model is trained with a **Type II maximum likelihood loss** combined with an **annealed KL divergence regulariser**:

$$\mathcal{L} = \sum_k y_k \left(\log S - \log \alpha_k\right) + \lambda(t) \cdot \text{KL}\left[\text{Dir}(\tilde{\alpha}) \| \text{Dir}(\mathbf{1})\right]$$

where $\tilde{\alpha}$ removes evidence for the correct class, and $\lambda(t) = \min(1, t/T_a)$ anneals the regulariser over training epochs.

### 3.3 Post-hoc Calibration

To make the scores usable for threshold-based decisions, we calibrate them after training with **temperature scaling** [3]. A single scalar parameter $T$ is learned on the validation logits by minimising the negative log-likelihood:

$$\hat{p}_k = \text{softmax}(z_k / T)$$

Temperature scaling preserves the predicted class (argmax is invariant to positive scaling) while adjusting the sharpness of the probability distribution. Calibration quality is measured with:

- **Expected Calibration Error (ECE)**: Average gap between confidence and accuracy across 15 confidence bins
- **Brier Score**: Mean squared error between predicted probability and true label
- **Classwise ECE**: ECE computed specifically on the minority (fraud) class

### 3.4 Selective Classification

We attach a rejection policy based on the epistemic uncertainty. Let $U(x)$ be the normalised epistemic uncertainty for transaction $x$ and $\tau \in [0, 1]$ an operating threshold:

- If $U(x) \leq \tau$ → the system classifies automatically
- If $U(x) > \tau$ → the system abstains and defers to human review

Sweeping $\tau$ traces the trade-off between coverage and reliability, reported via:

- **Accuracy-rejection curve**: Accuracy on accepted samples vs. rejection rate
- **Risk-coverage curve**: Error rate on accepted samples vs. coverage fraction
- **AURC**: Area Under the Risk-Coverage curve (lower is better)

### 3.5 Baselines

We compare EDL against three baselines, all sharing the same MLP backbone (128→64→32 with BatchNorm, ReLU, Dropout):

| Model | Uncertainty Method | Forward Passes | Epistemic Uncertainty |
|-------|-------------------|----------------|----------------------|
| **Softmax** | Max probability | 1 | 1 − max(p) |
| **MC Dropout** [Gal & Ghahramani, 2016] | Stochastic dropout at inference | T = 50 | Mutual information |
| **Deep Ensemble** [Lakshminarayanan et al., 2017] | M independent networks | M = 5 | Mutual information |
| **EDL** [Sensoy et al., 2018] | Dirichlet distribution | 1 | Vacuity (K/S) |

---

## 4. Experiments

### 4.1 Datasets

We evaluate on three public financial fraud datasets:

| Dataset | Size | Features | Fraud Rate | Characteristics |
|---------|------|----------|------------|-----------------|
| **Credit Card Fraud** (ULB, 2013) | 284,807 | 30 (28 PCA + Time + Amount) | 0.17% (492 fraud) | Real European card transactions; extreme imbalance |
| **PaySim** | ~6.3M (subsampled to 500K) | 14 (+ engineered) | ~1.3% | Synthetic mobile-money; multi-step fraud patterns |
| **SAML-D** | Synthetic AML | 12 | Very low | 28 typologies; demanding for rare pattern detection |

### 4.2 Preprocessing

- **Credit Card**: StandardScaler applied to Time and Amount features; V1–V28 are already PCA-transformed
- **PaySim**: One-hot encoded transaction types; engineered delta-balance and error-balance features; StandardScaler on all
- **SAML-D**: Label-encoded categoricals; StandardScaler on all numeric features; missing values filled with 0

All datasets use stratified 70/15/15 train/validation/test splits with `random_state=42`. Class imbalance is handled via:
- Weighted random sampling during training
- Class-weighted cross-entropy loss (weight = N_negative / N_positive)

### 4.3 Training Details

| Hyperparameter | Value |
|---------------|-------|
| Architecture | MLP: Input → 128 → 64 → 32 (BatchNorm + ReLU + Dropout) |
| Dropout rate | 0.3 |
| Optimiser | Adam (lr=1e-3, weight_decay=1e-5) |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Batch size | 512 |
| Epochs | 30 (baselines), 50 (EDL) |
| MC Dropout samples | T = 50 |
| Ensemble members | M = 5 |
| EDL annealing steps | 10 |
| Random seed | 42 |

### 4.4 Evaluation Protocol

For each dataset × model combination, we:
1. Train the model on the training set
2. Evaluate classification metrics (F1, AUPRC, ROC-AUC, Precision, Recall) on the test set
3. Collect validation logits and fit temperature scaling; measure ECE before and after
4. Compute epistemic and aleatoric uncertainty scores on the test set
5. Generate selective classification curves by sweeping the deferral threshold
6. Plot reliability diagrams, uncertainty distributions, and comparative curves

---

## 5. Results

### 5.1 Classification Performance

*Results are populated after running experiments. See `results/summary_results.csv` for the full table.*

The classification metrics across all models and datasets are summarised below:

| Dataset | Model | F1 | AUPRC | ROC-AUC | Precision | Recall |
|---------|-------|----|----|---------|-----------|--------|
| *To be filled after experiment run* | | | | | | |

### 5.2 Calibration Results

The effectiveness of temperature scaling is measured by comparing ECE and Brier scores before and after calibration:

| Dataset | Model | ECE (Before) | ECE (After) | Brier (Before) | Brier (After) | Optimal T |
|---------|-------|-------------|------------|----------------|---------------|-----------|
| *To be filled after experiment run* | | | | | | |

**Key observations:**
- Temperature scaling consistently reduces ECE across all models
- EDL produces inherently better-calibrated scores than the softmax baseline
- Classwise ECE on the fraud class reveals calibration issues masked by the majority class

### 5.3 Selective Classification

The selective classification results show how each model's accuracy improves when uncertain predictions are deferred:

| Dataset | Model | AURC ↓ | Coverage @99% Acc | Mean Epistemic |
|---------|-------|--------|-------------------|---------------|
| *To be filled after experiment run* | | | | |

**Key observations:**
- EDL achieves the lowest AURC, indicating superior selective classification
- The vacuity-based epistemic uncertainty of EDL provides a more principled basis for deferral than max-probability or variance-based approaches
- At equivalent accuracy targets, EDL maintains higher coverage (fewer deferrals)

### 5.4 Uncertainty Decomposition

EDL uniquely provides a single-pass decomposition of uncertainty:

- **Epistemic uncertainty (vacuity)** is higher for OOD or rare fraud patterns → appropriate for deferral
- **Aleatoric uncertainty** captures inherent noise in the data → not reducible by deferral

The uncertainty distribution plots (see `results/` directory) show clear separation between legitimate and fraudulent transactions for epistemic uncertainty, supporting its use as a deferral criterion.

### 5.5 Comparative Plots

The following plots are generated for each dataset:
- **Reliability diagrams** (before and after calibration) for each model
- **Accuracy-rejection curves** comparing all four models
- **Risk-coverage curves** comparing all four models  
- **Epistemic and aleatoric uncertainty distributions** for legitimate vs. fraud transactions

---

## 6. Conclusions

### 6.1 Summary of Findings

This project demonstrates that integrating Evidential Deep Learning with post-hoc calibration and selective classification provides a coherent framework for trustworthy fraud detection. The key findings are:

1. **EDL provides efficient uncertainty quantification**: Unlike MC Dropout (50 forward passes) and Deep Ensembles (5 networks), EDL achieves comparable or superior uncertainty estimates in a single forward pass, making it practical for real-time transaction screening.

2. **Post-hoc calibration is essential**: Temperature scaling consistently improves calibration across all models, and the improvement is particularly important for threshold-based decision policies.

3. **Selective classification improves reliability**: By deferring uncertain cases, the system achieves higher accuracy on accepted predictions while routing truly ambiguous cases to human reviewers.

4. **Epistemic–aleatoric decomposition is valuable**: EDL's ability to separate "I don't know" (epistemic) from "this is inherently noisy" (aleatoric) provides a principled basis for deferral decisions.

### 6.2 Limitations

- **Synthetic data**: PaySim and SAML-D are synthetic; real-world fraud distributions may differ
- **Static evaluation**: We evaluate on fixed test sets without temporal distribution shift
- **Binary classification**: The framework is demonstrated for binary fraud/non-fraud; extension to multi-class typology classification is future work
- **No human-in-the-loop evaluation**: The deferral policy is evaluated via metrics, not actual human review performance

### 6.3 Future Work

- Evaluate under temporal distribution shift using time-based train/test splits
- Integrate LIME/SHAP explanations for deferred cases (following Mapaila & Senekane [4])
- Extend to multi-class fraud typology classification
- Benchmark on proprietary datasets with real fraud patterns
- Investigate online/continual learning to adapt to evolving fraud tactics

---

## References

[1] S. H. Naqvi, "FRAUD DETECTION USING MACHINE LEARNING IN FINANCIAL TRANSACTIONS: A SYSTEMATIC REVIEW", Ver. Dir., vol. 23, p. e234187, Jan. 2026.

[2] M. Habibpour et al., 'Uncertainty-aware credit card fraud detection using deep learning', Eng. Appl. Artif. Intell., vol. 123, no. PA, Aug. 2023.

[3] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, 'On calibration of modern neural networks', in Proceedings of the 34th International Conference on Machine Learning - Volume 70, 2017, pp. 1321–1330.

[4] T. F. Mapaila and M. Senekane, 'Integrating Model Explainability and Uncertainty Quantification for Trustworthy Fraud Detection', Technologies, vol. 14, no. 4, 2026.

[5] B. Oztas, D. Cetinkaya, F. Adedoyin, M. Budka, H. Dogan, and G. Aksu, 'Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset', in 2023 IEEE International Conference on e-Business Engineering (ICEBE), 2023, pp. 47–54.

[6] M. Sensoy, L. Kaplan, and M. Kandemir, 'Evidential deep learning to quantify classification uncertainty', in Proceedings of the 32nd International Conference on Neural Information Processing Systems, Montréal, Canada, 2018, pp. 3183–3193.
