# Cross-Regime Robustness of Volatility Forecasting: Why a Two-Parameter Mean-Reversion Anchor Generalizes Where XGBoost, Random Forests, and Transformers Fail

## Abstract

Machine-learning models for predicting stock market volatility are usually tested by splitting the data randomly, or by cutting it once in time. Both ways let the model train and get tested on similar market conditions. But real markets change: during the 2008 and 2020 crises, volatility rose to about three times its calm level. This paper asks a simple question — when the market moves to a level the model never saw in training, how much of its skill survives? And can we fix the problem by building financial theory into a neural network?

We sort trading days into three groups by how volatile the market is at the time: low, medium, and high. We then run five experiments that train a model on one group and test it on a different one, forcing the model to predict volatility levels it never trained on. Six models are compared on exactly the same data: XGBoost, a Random Forest, a Transformer, two versions of a physics-informed neural network (PINN), and a simple two-parameter formula based on the idea that volatility tends to return to a normal level over time.

On the KOSPI200 index (2008 and 2016–2021), all three standard machine-learning models predict well *inside* a group (R² between 0.38 and 0.92) but completely fail *across* groups — every cross-group R² is negative. The failure is lopsided: predicting a calm market after training on a crisis is far worse (R² from −2.6 down to −27.2) than the other way around (−0.67 to −1.15). This same pattern shows up for all three model types. In contrast, the simple two-parameter formula — which does no training at all — gets *positive* R² (0.57 to 0.76) on four of the five experiments. A neural network only keeps this robustness when the formula is built directly into its output (the "additive" version), not when the formula is added as a training penalty (which barely helps). On the crisis-to-calm test, the additive network beats the Transformer by ΔR² = +23.0 (95% confidence interval [18.2, 28.9]). The lesson: doing well across market regimes is not about model size or flexibility. Flexible models throw away a simple, reusable pattern that a two-parameter formula keeps.

**Keywords:** cross-regime generalization; realized volatility; physics-informed neural networks; out-of-distribution robustness; mean reversion

---

## 1. Introduction

Volatility — how much prices swing — is a key input for pricing options, managing risk, and building portfolios. It can be forecast because of one steady fact: volatility comes in clusters and changes slowly. This idea is behind the popular GARCH models [2] and, for realized volatility, the Heterogeneous Autoregressive (HAR) model [1], a simple and widely used benchmark. Machine learning is now often used for this task and usually reported to beat these older methods. But almost all of these comparisons split the data randomly or cut it once in time, so the model is trained and tested on similar market conditions. In other words, the model only has to fill in gaps within a familiar setting.

Real markets don't cooperate. Sometimes the whole *level* of volatility jumps and stays there. A model trained during the calm years 2016–2019 and then run in March 2020 has to predict volatility values it has never seen. The reverse is just as important and gets less attention: a model trained during a crisis and used in the calm afterward must predict values far *below* anything in its training data. Both cases ask the model to *extrapolate* — to go beyond its training range — which is a very different job from what the usual tests measure.

Two things make us doubt the models can do this. First, tree-based models (the most common tool for this kind of data) can only output values inside the range they saw in training. A regression tree predicts the average of the training points that land in each "leaf," so its output can never go higher or lower than the training data. Random Forests have this same limit built in [7]: they cannot produce a volatility value outside their training range. Second, physics-informed neural networks (PINNs) — which add a known equation into the training process — have shown promise for option pricing, especially when data is limited. But no one has tested whether they still work when the market regime shifts, and researchers have even warned that trusting a PINN blindly could make people miss sudden changes the model was never trained on.

So there is a gap with three parts. Nobody has measured how well volatility models handle a deliberate, controlled change in market regime. Nobody has compared learned models, on the exact same data, against a simple formula treated as a real competitor — so we don't even know if learning helps at all when the regime changes. And nobody has asked *how* the theory should be built into a network, only *whether* it should be.

This paper tackles all three. We define market regimes by volatility level, which gives clean groups that automatically capture the 2008 and 2020 crises without us pointing them out. We run five experiments that move between regimes and compare six models under one careful setup, using a resampling method to check which differences are real. One of the six models is a plain formula with just two numbers to estimate and no training at all.

The results flip our starting question. Every standard model predicts well within a regime but has *zero* skill across regimes, and it fails in the same lopsided way for all three model types. The two-number formula, on the other hand, does work across regimes and beats every learned model. A network keeps this ability only when the formula is built into its structure. So the real finding is not "a PINN beats its baselines." It is that succeeding across regimes is about *structure, not size*: flexible models trade away a reusable pattern that a simple formula keeps. And how you insert the theory matters as much as the theory itself.

---

## 2. Related Work

**Realized volatility.** Realized volatility is a reliable way to measure how much prices actually moved. The HAR model [1] forecasts it by averaging past volatility over a day, a week, and a month — a simple setup that captures volatility's slow-fading "memory." Working with the logarithm of variance, rather than variance itself, reduces measurement noise [3]. This is the basis for our forecast target and for our use of log-variance.

**Machine learning for volatility.** Tree models [7, 8] and neural networks [9] are widely used and usually beat HAR-style baselines — but only under tests that never change the volatility level, which is exactly what we question here. In the Korean market, gradient boosting has been reported to beat GARCH on KOSPI volatility [14], and HAR-style models describe the VKOSPI implied-volatility index [15]; neither tests what happens across regimes. Korean-market ML studies also reach mixed conclusions about whether ML really beats simpler methods [16], which is another reason to study robustness rather than just headline accuracy.

**Why tree models can't extrapolate.** Tree models split the data into regions and predict a constant in each one, so their outputs are stuck inside the range of training values; Random Forests in particular cannot leave that range [7]. This explains our tree results and lets us make a claim about the whole *family* of tree models, not just one.

**Distribution shift.** When a model's training data doesn't match the data it later sees, performance drops — a well-studied problem known as out-of-distribution generalization [17]. In finance, regime changes have been handled with tree ensembles that keep updating [18] and with normalization tricks that rescale each sample to look similar [19]. We do the opposite of that last idea on purpose: instead of rescaling the shift away, we fit our scaling on the training data only, so that test data from a different regime really does arrive as "out of range." That way we *measure* the shift instead of hiding it. Our point is that a specific piece of financial structure, not a generic trick, is what carries over.

**PINNs in finance.** PINNs [10] add a governing equation into a network's training and have been used for option pricing [11], weather derivatives [13], and modeling the volatility surface [12]. Notably, [12] also points out that the Black–Scholes model's assumption of *constant* volatility is a weakness. This literature almost always adds the equation as a soft penalty during training, treats that choice as obvious, and never tests it across regimes — two gaps we target. Mean-reverting volatility models [4, 5] give the right building block, unlike the Geometric Brownian Motion often used in this area [6], which assumes constant volatility and so doesn't fit a volatility-forecasting target.

---

## 3. Method

### 3.1 Target and regimes
For daily log returns $r_t=\log(C_t/C_{t-1})$, our target is the volatility over the next 21 trading days, scaled to a yearly figure: $RV_t=\sqrt{252}\,\mathrm{sd}(r_{t+1},\dots,r_{t+21})$. We define regimes by *volatility level*: each day is labeled low, medium, or high based on which third it falls into, using the volatility over the past 21 days (about 550 days in each group). This makes each group internally consistent. As a check that the labeling makes sense, the "high" group — chosen purely by volatility — turns out to be mostly 2008 (200 days) and 2020 (191 days). So it finds both crises on its own, without us using any dates. We first tried grouping by calendar date instead, but that failed: putting all of 2020 into one "crisis" group mixed the March crash with the later recovery, and the model's within-group score fell apart.

### 3.2 Experiments and preventing leakage
Five experiments test different kinds of extrapolation: E1 (low→high, going up), E2 (high→low, going down), E3 (mid→high, partway up), E4 (mid→low, partway down), and E5 (low+high→mid, filling the middle). Because nearby days share overlapping windows (the target looks 21 days ahead and the features look back), training and test days that sit close in time could leak information. To stop this, we remove a 42-day buffer between any training and test day. Within-regime baselines use a similar buffered cross-validation. We also fit all scaling on the training data only, so test data from another regime genuinely arrives out of range.

### 3.3 The mean-reversion anchor
Geometric Brownian Motion doesn't fit here: it assumes volatility is *constant*, so it can't describe a changing volatility target. (When we tried forecasting the next day's return instead — the "direction" of the market — every model scored negative even within a regime, confirming that daily direction isn't forecastable and that this assumption gives us nothing to work with.) Instead we model log-variance $\ell_t=\log(RV_{21,t}^2)$ with an Ornstein–Uhlenbeck process [5] — the smooth version of the mean-reverting Heston model [4] — written as $d\ell=\kappa(\theta-\ell)\,dt+\xi\,dW$. Working this out over the forecast window gives our "anchor" formula:
$$\hat\ell_t=\theta+(\ell_t-\theta)\,\bar\phi,\quad \bar\phi=\frac{\phi(1-\phi^{H})}{H(1-\phi)},\quad \hat v_t=e^{\frac12\hat\ell_t}.$$
The anchor mixes today's *observed* volatility $\ell_t$ (which we can see even in a new regime) with a long-run normal level $\theta$. Unlike a tree, it has no fixed ceiling or floor — it can output whatever value today's data points to. We compare three ways of setting $\theta$: `fixed_train` (the training group's average), `broad` (the whole dataset's average), and `adaptive` (a rolling 63-day average that tracks the current level). The number $\phi$ measures how strongly volatility carries over from one day to the next, estimated on the training group.

### 3.4 How the formula is added, and the models
There are two ways to combine the anchor with a network $f$. The **soft** way adds it as a training penalty: $L=\mathrm{MSE}(f,y)+\lambda\,\mathrm{MSE}(f,\hat v)$ (with $\lambda=0$ this is just the plain network). The **additive** way builds it directly into the output: $f=\hat v+g(x)$, so the network only learns a correction on top of the anchor. These are genuinely different. A soft penalty only nudges the network where training data exists; the additive form guarantees the anchor shows up in every prediction, including in a new regime.

We compare six models on the exact same days. XGBoost [8] (600 trees, depth 5) and Random Forest [7] (600 trees) are the tree baselines. The Transformer [9] reads a 20-day window with $d_{\text{model}}{=}64$, 4 attention heads, 2 blocks, average pooling, trained with Adam and early stopping. The soft-PINN adds the penalty ($\lambda{=}0.3$) to this same network; the additive-PINN uses it to correct the anchor; and the bare anchor is the formula alone — two numbers, no training.

### 3.5 Evaluation
We report R² on the volatility target, plus three robustness scores: how much error grows across regimes (RPD), how much of the within-regime R² survives (RTR), and how well the model does on high-volatility tests (CRS). To check that a difference between two models is real and not luck, we use a block bootstrap: we resample the test data in 21-day chunks 1000 times and build a 95% confidence interval on the *difference* in R². A difference counts only if that interval stays away from zero. The neural models are run several times with different random seeds. We pick settings like $\lambda$ and the $\theta$ method using only within-regime data, never the cross-regime score we report.

---

## 4. Results

### 4.1 Within-regime skill
Every model forecasts volatility well *inside* a regime: XGBoost R² = 0.69/0.66/0.92 (low/mid/high), Random Forest 0.67/0.63/0.87, Transformer 0.42/0.38/0.88. Interestingly, the Transformer is *not* the best here — it ties the trees on the high group and does worse on low and mid. With only about 1,650 days of data, the attention model has no edge over gradient boosting.

### 4.2 Failure across regimes
Table~1 gives the cross-regime R² for all six models.

| Exp | Dir. | XGB | RF | Tr. | soft | **add** | **anch** |
|---|---|---|---|---|---|---|---|
| E1 | up | −1.15 | −1.14 | −1.11 | −1.12 | **0.68** | **0.71** |
| E3 | up | −0.80 | −0.78 | −0.67 | −0.76 | **0.59** | **0.57** |
| E2 | down | −22.9 | −27.2 | −22.2 | −10.1 | **0.70** | **0.76** |
| E4 | down | −3.75 | −5.77 | −2.55 | −4.41 | **−0.40** | −0.96 |
| E5 | bridge | −4.72 | −10.5 | −5.02 | −3.47 | **0.32** | **0.69** |

Every cross-regime R² is negative for all four learned models, and their high-volatility robustness score is 0.00. A model that scored above 0.9 within a regime becomes worse than just guessing the average once the level shifts. Four points stand out.

**Within-regime accuracy tells you nothing about across-regime robustness.** Random Forest and XGBoost are basically tied within regimes (0.67/0.63/0.87 vs 0.69/0.66/0.92), yet Random Forest is clearly worse across regimes on every experiment (E5: −10.5 vs −4.72). Two models that look equal on a normal test come apart the moment the regime changes.

**The failure belongs to the whole model family, not one model.** The two tree models use different algorithms, but their "going up" results are nearly identical (E1: −1.15 vs −1.14; E3: −0.80 vs −0.78). Both can only output values inside their training range. Random Forest is worse on every "going down" test, which fits how it works: averaging many trees pulls its guesses toward the training average, and when that average is a crisis level, that pull is exactly wrong. So we can say it about the whole family: tree models can't extrapolate to new volatility levels.

**The failure is lopsided, and the same for all three model types.** Going down (E2, E4) is a disaster (−2.6 to −27.2); going up (E1, E3) is only mildly bad (−0.67 to −1.15). The reason is how squared error grows: a crisis-trained model can't output values low enough for a calm market, so it overshoots by 3 to 4 times on every day, and the error blows up. A calm-trained model just maxes out at its ceiling when volatility rises, and its error is capped. This shows up for boosted trees, bagged trees, *and* the Transformer — so it's a property of the problem, not of any single method.

**Being flexible doesn't save the Transformer.** You might think a smooth neural network would fail gently instead of hitting a hard wall like a tree. It doesn't: on E2 it scores −22.2, about the same as the trees. It just maxes out instead of extrapolating.

### 4.3 What actually goes wrong: the prediction levels
Table~2 shows the average predicted volatility versus the truth. On E2 the learned models predict around 0.18–0.19 when the real answer is 0.103 — they overshoot by about 80% on essentially every day. A plot of predicted-vs-actual (Fig.~1) shows a *flat line*: their predictions barely move and just sit at the training-level floor, ignoring the real values. On the "going up" tests the opposite happens — they undershoot and max out near their ceiling. The anchor and the additive-PINN, by contrast, land right on target (0.104 and 0.102). In short: these models didn't learn how volatility *behaves*; they memorized the *level* of the regime they trained on — and a normal test can't tell the difference.

| Exp | Test | True | XGB | RF | Tr. | soft | add | anch |
|---|---|---|---|---|---|---|---|---|
| E2 | low | 0.103 | 0.184 | 0.192 | 0.182 | 0.156 | **0.102** | **0.104** |
| E1 | high | 0.297 | 0.120 | 0.120 | 0.122 | 0.122 | **0.254** | **0.262** |

### 4.4 The anchor needs a level that adapts
Table~3 shows the bare anchor's cross-regime R² under the three ways of setting $\theta$. The scores climb steadily from `fixed_train` to `adaptive` on E1 (−0.26 → 0.18 → 0.70) and E3 (−0.02 → 0.06 → 0.64). So it's not "mean reversion" by itself that works — it's mean reversion whose normal level is allowed to follow the current regime. If you lock $\theta$ to the training group's level, you carry the wrong level into the new regime, and it becomes a bias. What transfers between regimes is the *shape* of how volatility behaves; the *level* has to be re-estimated on the spot. With an adaptive level, a plain two-number formula with no training gets positive R² (0.57–0.76) on four of five experiments — exactly where every learned model scores between −0.67 and −27.2.

| Exp | `fixed_train` | `broad` | **`adaptive`** |
|---|---|---|---|
| E1 | −0.26 | 0.18 | **0.70** |
| E2 | 0.48 | 0.74 | **0.76** |
| E3 | −0.02 | 0.06 | **0.64** |
| E4 | −1.13 | −2.16 | **−0.19** |
| E5 | 0.70 | 0.70 | **0.69** |

### 4.5 How you add the formula decides everything
The soft-PINN shows why the method of adding matters. On E2 its average prediction is 0.156 — sitting *between* the failing models (about 0.18) and the truth (0.103). The penalty pulled it partway toward the right level but never got it there. On every experiment it stays deeply negative and is basically the same as the plain Transformer; trying many penalty strengths ($\lambda$ from 0 to 1.0) found none that reliably helped, and a promising result at $\lambda{=}0.3$ disappeared once we averaged over random seeds. The additive form, by contrast, turns every disaster into a positive score (E1 0.68, E2 0.70, E3 0.59, E5 0.32), and its results barely change across seeds (E2: ±0.14 versus ±4.08 for the plain models). So *encouraging* a network to match a good formula during training does not make it *use* that formula in a new regime — only building the formula into the output does.

Table~4 gives 95% confidence intervals on the difference between models. Every comparison is clearly real except one — the anchor vs the Transformer on E4, which is the single case where the anchor itself struggles.

| Exp | Contrast | ΔR² | 95% CI | Sig. |
|---|---|---|---|---|
| E2 | add − Transf. | +23.0 | [18.2, 28.9] | ✓ |
| E2 | anchor − XGB | +24.6 | [19.7, 31.1] | ✓ |
| E1 | add − Transf. | +1.93 | [1.55, 2.61] | ✓ |
| E3 | add − Transf. | +1.32 | [1.06, 1.71] | ✓ |
| E4 | add − Transf. | +1.88 | [0.90, 2.89] | ✓ |
| E4 | anchor − Transf. | +1.18 | [−0.44, 2.47] | — |
| E5 | add − Transf. | +1.06 | [0.65, 1.64] | ✓ |

### 4.6 The one hard case: E4
E4 (mid→low) is the only experiment no model solves, and the only one where the additive-PINN (−0.40) actually beats the bare anchor (−0.96). The reason is the carry-over number $\phi$: E4 has the lowest value of any experiment (0.843, versus 0.990 on E2). A low $\phi$ makes the anchor trust today's volatility less and lean on the normal level $\theta$ instead — but here $\theta$ is measured on a "low" group that sits right next to the "mid" training group in time, so it's contaminated by mid-level days. The result is a scattered, noisy set of predictions rather than a clean line. Importantly, the additive-PINN does better than the bare anchor here: the network's learned correction fixes a shaky formula. So the additive setup isn't just passing the formula through — it adds real value exactly where the formula is weakest.

---

## 5. Discussion

The answer to our main question is yes, but with big conditions. A mean-reversion-informed network does generalize across regimes where Transformers and trees fail (on E2 it beats the Transformer by ΔR² = +23.0). But almost all of that benefit comes from a two-number formula, and only when the formula is built into the network's structure. Adding it as a soft training penalty — the usual approach in finance PINNs — does nothing useful.

Why the learned models fail is now clear. Tree models can only output values inside their training range; trained on a crisis, they simply can't produce calm-market numbers, so they overshoot everywhere. The Transformer, even though it's smooth, maxes out instead of extrapolating and lands in the same place. Both learned the *level* of their regime, not how volatility behaves — and a normal test can't catch this.

The lopsided failure has a practical warning. Because squared error punishes big overshoots much more than undershoots, a model trained during a crisis and used in the calm afterward is *more* dangerous than the reverse. That is exactly the situation a risk system faces right after a crash, and it is the case the usual tests never check.

The anchor works for a structural reason: it mixes today's observed volatility with a normal level, so it has no ceiling, and with an adaptive level it simply follows whatever regime it lands in. The result about *how* to add the formula is the most useful takeaway: a rule a network is only encouraged to follow during training is not a rule it will actually rely on in a new regime. Building the rule into the output makes the good behavior automatic. Since the soft-penalty approach is standard across finance PINNs, including option pricing, this point may matter well beyond volatility.

Three findings surprised us: the Transformer was not the best model even within a regime (attention gained nothing at this data size); the plain formula beat every learned model, which flips the usual question from "does theory help the network?" to "does the network help the theory?"; and on E4 it does — the network repairs a shaky formula.

---

## 6. Limitations

The most important caveat: our adaptive level $\theta$ uses recent observed volatility from the test period. This is fair for forecasting (it uses no future data), but it means the anchor's edge comes from *quickly adjusting after* volatility moves, not from *predicting* the move. No model here forecasts a regime change before it happens. Also, volatility is both how we define the regimes and one of our inputs, so the regimes aren't fully independent of the features — this is on purpose, but worth stating. The study uses one index; each regime has only about 550 days, so the confidence intervals for the noisier models are wide; and even the best model fails on E4, so the formula *reduces* the across-regime problem rather than solving it. That E4 failure comes from how we estimate $\theta$ and $\phi$, not from mean reversion itself, so a smarter, regime-aware way of setting the level is the obvious next step.

---

## 7. Conclusion

Doing well across market regimes is about *structure, not size*. On the KOSPI200, XGBoost, Random Forest, and a Transformer all forecast well within a regime but have zero skill across regimes, failing in the same lopsided way for all three. Their within-regime accuracy says nothing about how they'll do across regimes. A two-number mean-reversion formula with an adaptive level beats every learned model on four of five experiments, and a network keeps that ability only when the formula is built into its output — adding it as a soft penalty is no better than not adding it at all. Flexible models don't fail to find a pattern; they throw away a simple, reusable one that a formula keeps. Volatility models meant for real use across market conditions should be tested on regime changes, and should keep a simple formula at their core, with a learned network on top to correct it. Future work: a smarter way to set the level that could fix E4; testing on other markets and finer time scales; and using this "build it into the output" idea for other finance PINN problems, especially option pricing, where the soft-versus-built-in choice hasn't been studied.

---

## References

**Realized volatility and HAR models**

[1] Corsi, F. 2009. A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics* 7, 2, 174–196. DOI:10.1093/jjfinec/nbp001.

[2] Andersen, T. G. and Bollerslev, T. 1998. Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts. *International Economic Review* 39, 4, 885–905.

[3] Andersen, T. G., Bollerslev, T., and Diebold, F. X. 2007. Roughing It Up: Including Jump Components in the Measurement, Modeling, and Forecasting of Return Volatility. *Review of Economics and Statistics* 89, 4, 701–720.

**Mean-reverting volatility processes**

[4] Heston, S. L. 1993. A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options. *Review of Financial Studies* 6, 2, 327–343.

[5] Uhlenbeck, G. E. and Ornstein, L. S. 1930. On the Theory of the Brownian Motion. *Physical Review* 36, 5, 823–841.

[6] Black, F. and Scholes, M. 1973. The Pricing of Options and Corporate Liabilities. *Journal of Political Economy* 81, 3, 637–654.

**Machine-learning models**

[7] Breiman, L. 2001. Random Forests. *Machine Learning* 45, 1, 5–32.

[8] Chen, T. and Guestrin, C. 2016. XGBoost: A Scalable Tree Boosting System. In *Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD '16)*. 785–794.

[9] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. 2017. Attention Is All You Need. In *Advances in Neural Information Processing Systems 30 (NeurIPS)*.

**Physics-informed neural networks**

[10] Raissi, M., Perdikaris, P., and Karniadakis, G. E. 2019. Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations. *Journal of Computational Physics* 378, 686–707.

[11] Dhiman, A. and Hu, Y. 2023. Physics Informed Neural Network for Option Pricing. arXiv:2312.06711 [q-fin.PR].

[12] Chataigner, M., Cousin, A., Crépey, S., Dixon, M., and Gueye, D. 2022. Physics-Informed Convolutional Transformer for Predicting Volatility Surface. arXiv:2209.10771 [q-fin.CP]. *(Notes the Black–Scholes constant-volatility assumption this work also rejects.)*

[13] Bansal, S., Boro, P., and Natesan, S. 2025. Physics-Informed Neural Network for Option Pricing: Weather Derivatives Model. *Computers & Mathematics with Applications* 200, 1–21. DOI:10.1016/j.camwa.2025.09.001.

**Korean market volatility**

[14] *(KOSPI volatility, GARCH vs XGBoost with XAI.)* A Study on KOSPI Volatility Prediction Using eXplainable Artificial Intelligence. *Journal of the Korean Society of Industry Convergence*, 2025. — *verify authors/volume/pages on Korea Science before submission.*

[15] *(VKOSPI, augmented HAR.)* Modeling and Predicting the Market Volatility Index: The Case of VKOSPI. — *EconStor working paper; verify authors, year, and outlet before submission.*

[16] Noh, H., Jang, H., and Yang, C. W. 2023. Forecasting Korean Stock Returns with Machine Learning. *Asia-Pacific Journal of Financial Studies* 52, 2, 193–241.

**Out-of-distribution generalization and distribution shift**

[17] Wu, X., Teng, F., Li, X., Zhang, J., Li, T., and Duan, Q. 2025. Out-of-Distribution Generalization in Time Series: A Survey. arXiv:2503.13868 [cs.LG].

[18] Wong, T. and Barahona, M. 2023. Deep Incremental Learning Models for Financial Temporal Tabular Datasets with Distribution Shifts. arXiv:2303.07925 [cs.LG]. *(Regime changes with XGBoost ensembles.)*

[19] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., and Choo, J. 2021. Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. In *Int. Conf. on Learning Representations (ICLR)*.

