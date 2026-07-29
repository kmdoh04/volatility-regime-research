[![DOI](https://zenodo.org/badge/1316063926.svg)](https://doi.org/10.5281/zenodo.21675714)
# Cross-Regime Robustness of Volatility Forecasting

**Why a two-parameter mean-reversion anchor generalizes where XGBoost, Random Forests, and Transformers fail.**

An empirical study of out-of-distribution robustness in volatility forecasting on the KOSPI200. Six models are compared on identical data under controlled regime transitions, testing whether learned models retain forecasting skill when the volatility level shifts — and whether embedding financial theory into a neural network repairs the failure.

> **Status:** Undergraduate research project. Preprint + code archived for the record.
> If you use this work, please cite via the DOI in `CITATION.cff` (see below).

---

## TL;DR — the main finding

Cross-regime volatility generalization is a problem of **structure, not capacity**. Flexible learners do not fail to find a signal; they discard transferable structure that a two-parameter formula retains, in exchange for regime-specific fit.

**Cross-regime R² (train on one volatility regime, test on another):**

| Experiment | XGBoost | Random Forest | Transformer | soft-PINN | **additive-PINN** | **anchor** |
|---|---|---|---|---|---|---|
| E1 (low→high) | −1.15 | −1.14 | −1.11 | −1.12 | **0.68** | **0.71** |
| E2 (high→low) | −22.9 | −27.2 | −22.2 | −10.1 | **0.70** | **0.76** |
| E3 (mid→high) | −0.80 | −0.78 | −0.67 | −0.76 | **0.59** | **0.57** |
| E4 (mid→low) | −3.75 | −5.77 | −2.55 | −4.41 | **−0.40** | −0.96 |
| E5 (bridge→mid) | −4.72 | −10.5 | −5.02 | −3.47 | **0.32** | **0.69** |

All four data-driven models score negative cross-regime R² everywhere (Crisis Robustness Score = 0.00). A two-parameter closed-form mean-reversion anchor, with **no training**, achieves positive R² on four of five experiments.

---

## Key results

1. **Data-driven models: strong in-regime, zero cross-regime.** XGBoost, Random Forest, and a Transformer all reach in-regime R² of 0.38–0.92 but negative R² on every regime transition.
2. **In-regime accuracy is uninformative about cross-regime robustness.** RF and XGBoost tie in-regime yet RF is strictly worse out-of-regime on every experiment — conventional validation cannot tell them apart.
3. **Failure is directionally asymmetric.** Downward extrapolation (crisis→calm) is catastrophic (R² −2.6 to −27.2); upward is bounded (−0.67 to −1.15). This replicates across three model families.
4. **A two-parameter anchor beats every learned model** — but only with an *adaptive* long-run level (θ).
5. **The injection mechanism is decisive.** The same physics helps decisively as an additive architectural component (`f = anchor + residual`) but does nothing as a soft training penalty. On E2, additive-PINN − Transformer: ΔR² = +23.0, 95% CI [18.2, 28.9].

See [`paper/`](paper/) for the full write-up and [`results/`](results/) for all numbers.

---

## Repository structure

```
.
├── README.md               # this file
├── requirements.txt        # exact package versions
├── CITATION.cff            # citation metadata (links to Zenodo DOI)
├── LICENSE
├── data/
│   └── make_dataset.py     # builds the shared KOSPI200 dataset CSV
├── src/
│   └── (shared helper functions, if extracted from notebooks)
├── code/
│   ├── xgboost_final.ipynb
│   ├── randomforest_final.ipynb
│   ├── transformer_final.ipynb
│   ├── pinn_final.ipynb            # soft-PINN + additive-PINN + anchor
│   └── final_comparison.ipynb     # master table, bootstrap CIs, figures
├── results/
│   ├── RESULTS.md                  # full results section
│   └── *.csv                       # per-model metrics, bootstrap CIs
├── figures/
│   └── (generated plots: E2 scatter, θ ablation, etc.)
└── paper/
    ├── paper.pdf
    └── paper.tex                   # ACM sigconf source
```

---

## How to reproduce

### 1. Environment

```bash
python -m pip install -r requirements.txt
```

Python 3.10+ recommended. TensorFlow 2.x is required for the Transformer and PINN models.

### 2. Build the dataset

```bash
python data/make_dataset.py
```

This downloads KOSPI200 daily data (via `yfinance`), builds the 13 features and the forward-21-day realized-volatility target, assigns volatility-state regimes, and writes a single shared CSV. All notebooks read this file, so every model sees byte-identical inputs.

> **Note on data:** the raw price data is pulled from a public source at run time and is **not** committed to this repository. Running `make_dataset.py` regenerates it. Results may differ marginally if the data provider revises historical values.

### 3. Run the models

Open the notebooks in `notebooks/` in order. Each reads the shared CSV and writes its metrics to `results/`. `final_comparison.ipynb` refits all six models, produces the master table, computes block-bootstrap confidence intervals, and generates the figures.

Neural models are stochastic; set `N_SEEDS` (default 5) higher for tighter confidence intervals. A full run of `final_comparison.ipynb` is compute-heavy (6 models × 5 experiments × seeds) — start with `N_SEEDS=2` and reduced epochs to smoke-test.

---

## Method in brief

- **Target:** forward 21-day annualized realized volatility.
- **Regimes:** defined by *volatility state* (terciles of trailing 21-day realized volatility), not calendar date. The high regime recovers the 2008 and 2020 crises without being told about them.
- **Leakage control:** every train/test contact purged with a 42-day embargo; standardization fit on the training split only.
- **Physics:** an Ornstein–Uhlenbeck process on log-variance (mean-reverting), replacing the constant-volatility GBM conventional in the finance-PINN literature.
- **Inference:** block bootstrap (1000 resamples, 21-day blocks) on R² *differences*; a difference counts only if its 95% CI excludes zero.

Full details in [`paper/paper.pdf`](paper/).

---

## Limitations

The adaptive long-run level uses trailing observed volatility from the test period. This is causal and legitimate for forecasting, but it means the anchor's advantage comes from **adapting quickly once volatility has moved, not from anticipating the shift**. No model here predicts a regime change. Physics *mitigates* cross-regime failure; it does not solve it. The study covers a single index; generalization to other markets is future work.

---

## Citation

If you reference this work, please cite it using the metadata in [`CITATION.cff`](CITATION.cff). A DOI will be minted via Zenodo on release (see below).

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE). You are free to use, modify, and build on this work with attribution.
