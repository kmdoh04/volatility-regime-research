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
H, WTRAIL, SEQ_LEN = 21, 21, 20
EMBARGO  = max(WTRAIL, SEQ_LEN) + H
GAP_DAYS = 10
N_SPLITS = 4
QUANTILES = (1/3, 2/3)
RANGES = [("2008-01-01", "2009-01-01"), ("2016-01-01", "2022-01-01")]

D_MODEL, N_HEADS, N_BLOCKS, FF_DIM = 64, 4, 2, 128
EPOCHS, BATCH, PATIENCE = 80, 32, 8
SEED = 42

THETA_MODE = "adaptive"
THETA_WIN  = 63
LAMBDA     = 0.3
ADD_LAMBDA = 0.1


COMPARE_SEEDS  = 5
COMPARE_EPOCHS = 50

REGIME_ORDER = ["low", "mid", "high"]
STRESS_REGIMES = ["high"]
EXPERIMENTS = [
    ("E1", ["low"],          "high"),
    ("E2", ["high"],         "low"),
    ("E3", ["mid"],          "high"),
    ("E4", ["mid"],          "low"),
    ("E5", ["low", "high"],  "mid"),
]
MODEL_NAME = "pinn"
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
RV21 = data["rvol_21"].values.astype("float64")
LV   = 2 * np.log(RV21)
daydiff = DIDX.to_series().diff().dt.days.fillna(9999).values

valid = np.zeros(len(DIDX), dtype=bool)
for i in range(SEQ_LEN - 1, len(DIDX)):
    if (daydiff[i - SEQ_LEN + 2: i + 1] > GAP_DAYS).any():
        continue
    valid[i] = True

THETA_BROAD = float(np.nanmean(LV))
block = pd.Series((daydiff > GAP_DAYS).cumsum(), index=DIDX)
_adapt = (pd.Series(LV, index=DIDX).groupby(block)
          .transform(lambda s: s.rolling(THETA_WIN, min_periods=10).mean()).values)
THETA_ADAPT = np.where(np.isfinite(_adapt), _adapt, THETA_BROAD)

def pos_of(df):
    p = DIDX.get_indexer(df.index); return p[valid[p]]
def seqs_for(df):
    p = pos_of(df)
    if len(p) == 0:
        return np.empty((0, SEQ_LEN, F), "float32"), np.empty((0,), "float32"), p
    return np.stack([FEAT[q - SEQ_LEN + 1: q + 1] for q in p]), YALL[p], p
def y_valid(df):
    return YALL[pos_of(df)]

print(f"valid target-days: {int(valid.sum())} / {len(DIDX)}")


def phi_from(train_pos):
    sp = np.sort(train_pos); consec = np.where(np.diff(sp) == 1)[0]
    if len(consec) > 5:
        return float(np.clip(np.corrcoef(LV[sp[consec]], LV[sp[consec + 1]])[0, 1], 0.0, 0.999))
    return 0.9

def theta_for(pos, train_pos, mode):
    if mode == "fixed_train":
        return np.full(len(pos), float(np.nanmean(LV[train_pos])))
    if mode == "broad":
        return np.full(len(pos), THETA_BROAD)
    if mode == "adaptive":
        return THETA_ADAPT[pos]
    raise ValueError(mode)

def vphys(pos, theta_arr, phi):
    phibar = phi * (1 - phi ** H) / (H * (1 - phi)) if phi < 1 else 1.0
    return np.exp(0.5 * (theta_arr + (LV[pos] - theta_arr) * phibar)).astype("float32")

def anchor_only(train_df, eval_df, mode):
    """Pure mean-reversion forecast on the test regime -- NO training."""
    tp, ep = pos_of(train_df), pos_of(eval_df)
    return vphys(ep, theta_for(ep, tp, mode), phi_from(tp))


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

def _backbone(seed):
    tf.random.set_seed(seed)
    inp = keras.Input((SEQ_LEN, F))
    x = layers.Dense(D_MODEL)(inp)
    x = x + layers.Embedding(SEQ_LEN, D_MODEL)(tf.range(SEQ_LEN))
    for _ in range(N_BLOCKS):
        a = layers.MultiHeadAttention(N_HEADS, D_MODEL // N_HEADS)(x, x)
        x = layers.LayerNormalization()(x + a)
        f = layers.Dense(FF_DIM, activation="relu")(x); f = layers.Dense(D_MODEL)(f)
        x = layers.LayerNormalization()(x + f)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    return keras.Model(inp, layers.Dense(1)(x))

class MeanReversionPINN:
    """arch='soft'  -> loss = data + lam*MSE(f, anchor)   (lam=0 == Transformer)
       arch='additive' -> f = anchor + residual_net;  lam shrinks the residual."""
    def __init__(self, arch="soft", lam=LAMBDA, seed=SEED, theta_mode=THETA_MODE, epochs=EPOCHS):
        self.arch, self.lam, self.seed, self.theta_mode, self.epochs = arch, lam, seed, theta_mode, epochs
    def fit(self, train_df, _=None):
        X, y, pos = seqs_for(train_df)
        self.tp, self.phi = pos, phi_from(pos)
        anchor = vphys(pos, theta_for(pos, pos, self.theta_mode), self.phi)
        self.mu = X.reshape(-1, F).mean(0); self.sd = X.reshape(-1, F).std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        self.model = _backbone(self.seed); lam = self.lam
        if self.arch == "additive":
            resid = y - anchor
            self.r_sd = resid.std() + 1e-8
            Yt = (resid / self.r_sd).reshape(-1, 1)
            def loss(yt, yp):
                return tf.reduce_mean(tf.square(yp - yt)) + lam * tf.reduce_mean(tf.square(yp))
        else:
            self.ty_mu, self.ty_sd = y.mean(), y.std() + 1e-8
            Yt = np.concatenate([((y - self.ty_mu) / self.ty_sd).reshape(-1, 1),
                                 ((anchor - self.ty_mu) / self.ty_sd).reshape(-1, 1)], axis=1)
            def loss(yt, yp):
                return tf.reduce_mean(tf.square(yp - yt[:, 0:1])) + lam * tf.reduce_mean(tf.square(yp - yt[:, 1:2]))
        self.model.compile("adam", loss=loss)
        self.model.fit(Xs, Yt, epochs=self.epochs, batch_size=BATCH, verbose=0,
                       validation_split=0.15,
                       callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
        return self
    def predict(self, eval_df):
        X, _, pos = seqs_for(eval_df); Xs = (X - self.mu) / self.sd
        out = self.model.predict(Xs, verbose=0).ravel()
        if self.arch == "additive":
            anchor = vphys(pos, theta_for(pos, self.tp, self.theta_mode), self.phi)
            return anchor + out * self.r_sd
        return out * self.ty_sd + self.ty_mu

def run_experiment(train_regimes, test_regime, model_factory):
    tr = embargo(data[data["regime"].isin(train_regimes)], data[data["regime"] == test_regime])
    ev = data[data["regime"] == test_regime]
    return compute_metrics(y_valid(ev), model_factory().fit(tr).predict(ev))

def in_regime_cv(regime, model_factory, n_splits=N_SPLITS):
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


diag = []
for name, train_regs, test_reg in EXPERIMENTS:
    tr = embargo(data[data["regime"].isin(train_regs)], data[data["regime"] == test_reg])
    tr_pos, te_pos = pos_of(tr), pos_of(data[data["regime"] == test_reg])
    phi = phi_from(tr_pos)
    row = {"exp": name, "test": test_reg, "phi": phi,
           "true_vol": float(np.exp(0.5 * LV[te_pos]).mean())}
    for mode in ["fixed_train", "broad", "adaptive"]:
        row[f"anchor_{mode}"] = float(vphys(te_pos, theta_for(te_pos, tr_pos, mode), phi).mean())
    diag.append(row)
pd.DataFrame(diag).set_index("exp")


anc_rows = []
for name, train_regs, test_reg in EXPERIMENTS:
    tr = data[data["regime"].isin(train_regs)]; ev = data[data["regime"] == test_reg]
    row = {"exp": name, "test": test_reg}
    for mode in ["fixed_train", "broad", "adaptive"]:
        row[f"R2_{mode}"] = compute_metrics(y_valid(ev), anchor_only(tr, ev, mode))["R2"]
    anc_rows.append(row)
anchor_df = pd.DataFrame(anc_rows).set_index("exp")
anchor_df.to_csv("results_bare_anchor.csv")
anchor_df


head_factory = lambda: MeanReversionPINN(arch="additive", lam=ADD_LAMBDA,
                                         theta_mode=THETA_MODE, seed=SEED)
baselines, rows = {}, []
for r in REGIME_ORDER:
    m = in_regime_cv(r, head_factory); baselines[r] = m
    rows.append({"regime": r, **m}); print("done", r)
baseline_df = pd.DataFrame(rows).set_index("regime")
baseline_df


def seed_stats(trs, te, make):
    r2s, rmses = [], []
    for s in range(COMPARE_SEEDS):
        m = run_experiment(trs, te, lambda s=s: make(s))
        r2s.append(m["R2"]); rmses.append(m["RMSE"])
    return np.mean(r2s), np.std(r2s), np.mean(rmses)

cmp_rows = []
for name, trs, te in EXPERIMENTS:
    anc = compute_metrics(y_valid(data[data.regime == te]),
                          anchor_only(data[data.regime.isin(trs)], data[data.regime == te], "adaptive"))["R2"]
    tf_m, tf_s, _   = seed_stats(trs, te, lambda s: MeanReversionPINN("soft", 0.0, s, "adaptive", COMPARE_EPOCHS))
    sf_m, sf_s, _   = seed_stats(trs, te, lambda s: MeanReversionPINN("soft", LAMBDA, s, "adaptive", COMPARE_EPOCHS))
    ad_m, ad_s, _   = seed_stats(trs, te, lambda s: MeanReversionPINN("additive", ADD_LAMBDA, s, "adaptive", COMPARE_EPOCHS))
    cmp_rows.append({"exp": name, "test": te, "anchor": anc,
                     "transformer": tf_m, "transformer_sd": tf_s,
                     "soft_pinn": sf_m, "soft_pinn_sd": sf_s,
                     "additive_pinn": ad_m, "additive_pinn_sd": ad_s})
    print("done", name)
compare_df = pd.DataFrame(cmp_rows).set_index("exp")
compare_df.to_csv("results_decisive_comparison.csv")
compare_df


labels = compare_df.index.tolist(); x = np.arange(len(labels)); w = 0.22
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - 1.5*w, compare_df["transformer"],   w, yerr=compare_df["transformer_sd"],   label="Transformer (λ=0)")
ax.bar(x - 0.5*w, compare_df["soft_pinn"],      w, yerr=compare_df["soft_pinn_sd"],      label="soft-PINN")
ax.bar(x + 0.5*w, compare_df["additive_pinn"],  w, yerr=compare_df["additive_pinn_sd"],  label="additive-PINN")
ax.bar(x + 1.5*w, compare_df["anchor"],         w, label="bare anchor (no training)", color="black", alpha=0.7)
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("cross-regime R2"); ax.set_title("Decisive comparison (adaptive θ)"); ax.legend()
plt.tight_layout(); plt.show()


SWEEP_MODES   = ["fixed_train", "adaptive"]
SWEEP_LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
SWEEP_SEEDS   = 3
SWEEP_EPOCHS  = 40
sweep_rows = []
for exp_name, trs, te in [("E2", ["high"], "low"), ("E5", ["low", "high"], "mid")]:
    for mode in SWEEP_MODES:
        for lam in SWEEP_LAMBDAS:
            r2s = [run_experiment(trs, te,
                    lambda s=s, lam=lam, mode=mode: MeanReversionPINN("soft", lam, s, mode, SWEEP_EPOCHS))["R2"]
                   for s in range(SWEEP_SEEDS)]
            sweep_rows.append({"exp": exp_name, "mode": mode, "lambda": lam,
                               "R2_mean": np.mean(r2s), "R2_std": np.std(r2s)})
sweep_df = pd.DataFrame(sweep_rows); sweep_df.to_csv("results_pinn_lambda_theta_sweep.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, exp_name in zip(axes, ["E2", "E5"]):
    for mode in SWEEP_MODES:
        s = sweep_df[(sweep_df.exp == exp_name) & (sweep_df["mode"] == mode)].sort_values("lambda")
        ax.plot(s["lambda"], s["R2_mean"], marker="o", label=f"θ={mode}")
        ax.fill_between(s["lambda"], s["R2_mean"]-s["R2_std"], s["R2_mean"]+s["R2_std"], alpha=0.2)
    ax.axhline(0, color="k", lw=0.6); ax.set_title(f"{exp_name}: soft-PINN R2 vs λ")
    ax.set_xlabel("λ"); ax.set_ylabel("cross-regime R2"); ax.legend()
plt.tight_layout(); plt.show()
