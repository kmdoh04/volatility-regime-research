import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
plt.rcParams["figure.figsize"] = (11, 4)
print("TensorFlow", tf.__version__)


TICKER = "^KS200"
TRADING_DAYS = 252
H        = 21
WTRAIL   = 21
SEQ_LEN  = 20
EMBARGO  = max(WTRAIL, SEQ_LEN) + H
GAP_DAYS = 10
N_SPLITS = 4
QUANTILES = (1/3, 2/3)
RANGES = [("2008-01-01", "2009-01-01"), ("2016-01-01", "2022-01-01")]


D_MODEL, N_HEADS, N_BLOCKS, FF_DIM = 64, 4, 2, 128
EPOCHS, BATCH, PATIENCE = 80, 32, 8
SEED = 42

REGIME_ORDER = ["low", "mid", "high"]
STRESS_REGIMES = ["high"]
EXPERIMENTS = [
    ("E1", ["low"],          "high"),
    ("E2", ["high"],         "low"),
    ("E3", ["mid"],          "high"),
    ("E4", ["mid"],          "low"),
    ("E5", ["low", "high"],  "mid"),
]
MODEL_NAME = "transformer"
tf.random.set_seed(SEED); np.random.seed(SEED)


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


FEAT = data[FEATURES].values.astype("float32")
YALL = data[TARGET].values.astype("float32")
daydiff = DIDX.to_series().diff().dt.days.fillna(9999).values

valid = np.zeros(len(DIDX), dtype=bool)
for i in range(SEQ_LEN - 1, len(DIDX)):
    if (daydiff[i - SEQ_LEN + 2: i + 1] > GAP_DAYS).any():
        continue
    valid[i] = True

def seqs_for(df):
    pos = DIDX.get_indexer(df.index); pos = pos[valid[pos]]
    if len(pos) == 0:
        return np.empty((0, SEQ_LEN, F), "float32"), np.empty((0,), "float32")
    X = np.stack([FEAT[p - SEQ_LEN + 1: p + 1] for p in pos])
    return X, YALL[pos]

def y_valid(df):
    pos = DIDX.get_indexer(df.index); return YALL[pos[valid[pos]]]

print(f"valid target-days: {int(valid.sum())} / {len(DIDX)}")


def compute_metrics(y_true, pred):
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
            "MAE":  float(mean_absolute_error(y_true, pred)),
            "MAPE": float(np.mean(np.abs((y_true - pred) / y_true)) * 100),
            "R2":   float(r2_score(y_true, pred))}

def embargo(train_df, test_df, span=EMBARGO):
    banned = np.zeros(len(DIDX), dtype=bool)
    for p in DIDX.get_indexer(test_df.index):
        banned[max(0, p - span): min(len(DIDX), p + span + 1)] = True
    return train_df[~banned[DIDX.get_indexer(train_df.index)]]

class TransformerRegressor:
    """sklearn-style .fit/.predict wrapper so it drops into the shared harness."""
    def __init__(self, seed=SEED):
        self.seed = seed
    def _build(self):
        tf.random.set_seed(self.seed)
        inp = keras.Input((SEQ_LEN, F))
        x = layers.Dense(D_MODEL)(inp)
        pe = layers.Embedding(SEQ_LEN, D_MODEL)(tf.range(SEQ_LEN))
        x = x + pe
        for _ in range(N_BLOCKS):
            a = layers.MultiHeadAttention(N_HEADS, D_MODEL // N_HEADS)(x, x)
            x = layers.LayerNormalization()(x + a)
            f = layers.Dense(FF_DIM, activation="relu")(x)
            f = layers.Dense(D_MODEL)(f)
            x = layers.LayerNormalization()(x + f)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1)(x)
        m = keras.Model(inp, out); m.compile("adam", "mse"); return m
    def fit(self, train_df, _=None):
        X, y = seqs_for(train_df)
        self.mu = X.reshape(-1, F).mean(0); self.sd = X.reshape(-1, F).std(0) + 1e-8
        self.ty_mu = y.mean(); self.ty_sd = y.std() + 1e-8
        Xs = (X - self.mu) / self.sd; ys = (y - self.ty_mu) / self.ty_sd
        self.model = self._build()
        self.model.fit(Xs, ys, epochs=EPOCHS, batch_size=BATCH, verbose=0,
                       validation_split=0.15,
                       callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE,
                                                                restore_best_weights=True)])
        return self
    def predict(self, eval_df):
        X, _ = seqs_for(eval_df)
        Xs = (X - self.mu) / self.sd
        return self.model.predict(Xs, verbose=0).ravel() * self.ty_sd + self.ty_mu

def make_transformer():
    return TransformerRegressor(seed=SEED)

def in_regime_cv(regime, model_factory=make_transformer, n_splits=N_SPLITS):
    g = data[data["regime"] == regime].sort_index()
    folds = np.array_split(np.arange(len(g)), n_splits)
    preds, trues = [], []
    for k in range(n_splits):
        te = g.iloc[folds[k]]
        tr = embargo(g.iloc[np.concatenate([folds[j] for j in range(n_splits) if j != k])], te)
        if len(seqs_for(tr)[0]) == 0 or len(seqs_for(te)[0]) == 0:
            continue
        mdl = model_factory().fit(tr)
        preds.append(mdl.predict(te)); trues.append(y_valid(te))
    return compute_metrics(np.concatenate(trues), np.concatenate(preds))

def run_experiment(train_regimes, test_regime, model_factory=make_transformer):
    tr = embargo(data[data["regime"].isin(train_regimes)], data[data["regime"] == test_regime])
    ev = data[data["regime"] == test_regime]
    mdl = model_factory().fit(tr)
    return compute_metrics(y_valid(ev), mdl.predict(ev))


baselines, rows = {}, []
for r in REGIME_ORDER:
    m = in_regime_cv(r); baselines[r] = m
    rows.append({"regime": r, **m}); print("done", r)
baseline_df = pd.DataFrame(rows).set_index("regime")
baseline_df


exp_rows = []
for name, train_regs, test_reg in EXPERIMENTS:
    m = run_experiment(train_regs, test_reg)
    exp_rows.append({"exp": name, "train": "+".join(train_regs), "test": test_reg, **m})
    print("done", name)
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
CRS = float(np.mean(np.clip(rob_df[rob_df["test"].isin(STRESS_REGIMES)]["R2_cross"], 0, None)))


baseline_df.assign(model=MODEL_NAME).to_csv(f"results_{MODEL_NAME}_baselines.csv")
exp_df.assign(model=MODEL_NAME).to_csv(f"results_{MODEL_NAME}_experiments.csv")
rob_df.assign(model=MODEL_NAME, CRS=CRS).to_csv(f"results_{MODEL_NAME}_robustness.csv")
print(rob_df.to_string()); print(f"\nCRS (Transformer, high-vol tests): {CRS:.4f}")


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
rob_df["RPD"].plot(kind="bar", ax=axes[0], color="tab:purple")
axes[0].set_title("Transformer — Relative Performance Degradation"); axes[0].axhline(0, color="k", lw=0.6)
comp = pd.DataFrame({"in_regime_R2": [baselines[t]["R2"] for t in rob_df["test"]],
                     "cross_regime_R2": rob_df["R2_cross"].values}, index=rob_df.index)
comp.plot(kind="bar", ax=axes[1]); axes[1].set_title("Transformer — Volatility R2: in vs cross")
axes[1].axhline(0, color="k", lw=0.6)
plt.tight_layout(); plt.show()


SEEDS = [0, 1, 2]
ms_rows = []
for name, train_regs, test_reg in EXPERIMENTS:
    r2s, rmses = [], []
    for s in SEEDS:
        m = run_experiment(train_regs, test_reg, model_factory=lambda s=s: TransformerRegressor(seed=s))
        r2s.append(m["R2"]); rmses.append(m["RMSE"])
    ms_rows.append({"exp": name, "test": test_reg,
                    "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
                    "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses)})
pd.DataFrame(ms_rows).set_index("exp")
