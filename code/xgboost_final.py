import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
plt.rcParams["figure.figsize"] = (11, 4)


TICKER = "^KS200"
TRADING_DAYS = 252
H        = 21
WTRAIL   = 21
EMBARGO  = H + WTRAIL
GAP_DAYS = 10
N_SPLITS = 4
QUANTILES = (1/3, 2/3)


RANGES = [("2008-01-01", "2009-01-01"), ("2016-01-01", "2022-01-01")]

REGIME_ORDER = ["low", "mid", "high"]
STRESS_REGIMES = ["high"]
EXPERIMENTS = [
    ("E1", ["low"],          "high"),
    ("E2", ["high"],         "low"),
    ("E3", ["mid"],          "high"),
    ("E4", ["mid"],          "low"),
    ("E5", ["low", "high"],  "mid"),
]


DATA_PATH = "kospi200_dataset.csv"
data = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date").sort_index()
raw = data
FEATURES = ([f"ret_lag_{l}" for l in range(1, 6)]
            + [f"rvol_{w}" for w in (5, 10, 21)]
            + [f"ma_ratio_{w}" for w in (5, 10, 20)]
            + ["hl_range", "vol_ratio"])
TARGET = "target_vol"
DIDX = data.index
F = len(FEATURES)
_missing = [c for c in FEATURES + [TARGET, "regime", "rvol_21"] if c not in data.columns]
assert not _missing, f"CSV missing {_missing}; re-run make_dataset.py"
print(f"loaded {len(data)} rows from {DATA_PATH}")
print("regime counts:", data["regime"].value_counts().to_dict())


cmap = {"low": "tab:green", "mid": "gold", "high": "tab:red"}
fig, ax = plt.subplots()
ax.plot(data.index, data["Close"], color="lightgray", lw=0.8, zorder=1)
for r in REGIME_ORDER:
    g = data[data["regime"] == r]
    ax.scatter(g.index, g["Close"], s=6, c=cmap[r], label=r, zorder=2)
ax.set_title("KOSPI200 colored by volatility regime"); ax.legend()
plt.tight_layout(); plt.show()


for r in REGIME_ORDER:
    plt.hist(data[data.regime == r][TARGET], bins=40, alpha=0.5, label=r, color=cmap[r])
plt.legend(); plt.title(f"Forward {H}-day realized volatility by regime")
plt.xlabel("annualized vol"); plt.tight_layout(); plt.show()


def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        n_jobs=-1, random_state=42)

def compute_metrics(y_true, pred):
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    return {"RMSE": np.sqrt(mean_squared_error(y_true, pred)),
            "MAE":  mean_absolute_error(y_true, pred),
            "MAPE": np.mean(np.abs((y_true - pred) / y_true)) * 100,
            "R2":   r2_score(y_true, pred)}

def embargo(train_df, test_df, span=EMBARGO):
    """Drop training rows within `span` trading days of any test row (both directions)."""
    banned = np.zeros(len(DIDX), dtype=bool)
    for p in DIDX.get_indexer(test_df.index):
        banned[max(0, p - span): min(len(DIDX), p + span + 1)] = True
    return train_df[~banned[DIDX.get_indexer(train_df.index)]]

def fit_predict(train_df, eval_df, model_factory=make_xgb):
    model = model_factory()
    model.fit(train_df[FEATURES], train_df[TARGET])
    return model.predict(eval_df[FEATURES])


def in_regime_cv(regime, model_factory=make_xgb, n_splits=N_SPLITS):
    g = data[data["regime"] == regime].sort_index()
    folds = np.array_split(np.arange(len(g)), n_splits)
    preds, trues = [], []
    for k in range(n_splits):
        te = g.iloc[folds[k]]
        tr = g.iloc[np.concatenate([folds[j] for j in range(n_splits) if j != k])]
        tr = embargo(tr, te)
        if len(tr) == 0:
            continue
        preds.append(fit_predict(tr, te, model_factory))
        trues.append(te[TARGET].values)
    return compute_metrics(np.concatenate(trues), np.concatenate(preds))

baselines, rows = {}, []
for r in REGIME_ORDER:
    m = in_regime_cv(r)
    baselines[r] = m
    rows.append({"regime": r, **m})
baseline_df = pd.DataFrame(rows).set_index("regime")
baseline_df


def run_experiment(train_regimes, test_regime, model_factory=make_xgb):
    tr = embargo(data[data["regime"].isin(train_regimes)], data[data["regime"] == test_regime])
    ev = data[data["regime"] == test_regime]
    return compute_metrics(ev[TARGET].values, fit_predict(tr, ev, model_factory))

exp_rows = []
for name, train_regs, test_reg in EXPERIMENTS:
    m = run_experiment(train_regs, test_reg)
    exp_rows.append({"exp": name, "train": "+".join(train_regs), "test": test_reg, **m})
exp_df = pd.DataFrame(exp_rows).set_index("exp")
exp_df


rob_rows = []
for name, train_regs, test_reg in EXPERIMENTS:
    m, b = exp_df.loc[name], baselines[test_reg]
    rpd = (m["RMSE"] - b["RMSE"]) / b["RMSE"]
    rtr = (max(0.0, m["R2"]) / b["R2"]) if b["R2"] > 0 else np.nan
    rob_rows.append({"exp": name, "train": "+".join(train_regs), "test": test_reg,
                     "RPD": rpd, "RTR": rtr, "R2_cross": m["R2"]})
rob_df = pd.DataFrame(rob_rows).set_index("exp")

stress = rob_df[rob_df["test"].isin(STRESS_REGIMES)]
CRS = float(np.mean(np.clip(stress["R2_cross"], 0, None)))
print(rob_df.to_string())
print(f"\nCrisis Robustness Score (XGBoost, high-vol tests): {CRS:.4f}")
if rob_df["RTR"].isna().any():
    print("NB: RTR is NaN where in-regime R2 <= 0 (ratio undefined).")


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
rob_df["RPD"].plot(kind="bar", ax=axes[0], color="tab:red")
axes[0].set_title("Relative Performance Degradation (volatility RMSE)"); axes[0].axhline(0, color="k", lw=0.6)
comp = pd.DataFrame({"in_regime_R2": [baselines[t]["R2"] for t in rob_df["test"]],
                     "cross_regime_R2": rob_df["R2_cross"].values}, index=rob_df.index)
comp.plot(kind="bar", ax=axes[1]); axes[1].set_title("Volatility R2: in-regime vs cross-regime")
axes[1].axhline(0, color="k", lw=0.6)
plt.tight_layout(); plt.show()
