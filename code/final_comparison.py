import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pd.set_option("display.float_format", lambda x: f"{x:.3f}")
plt.rcParams["figure.figsize"] = (11, 4)

DATA_PATH = "kospi200_dataset.csv"
H, WTRAIL, SEQ_LEN = 21, 21, 20
EMBARGO   = max(WTRAIL, SEQ_LEN) + H
GAP_DAYS  = 10
THETA_WIN = 63
THETA_MODE = "adaptive"
LAMBDA, ADD_LAMBDA = 0.3, 0.1
D_MODEL, N_HEADS, N_BLOCKS, FF_DIM = 64, 4, 2, 128
EPOCHS, BATCH, PATIENCE = 50, 32, 8
N_SEEDS = 5
BOOT_B, BOOT_BLOCK = 1000, 21

EXPERIMENTS = [
    ("E1", ["low"],         "high"),
    ("E2", ["high"],        "low"),
    ("E3", ["mid"],         "high"),
    ("E4", ["mid"],         "low"),
    ("E5", ["low", "high"], "mid"),
]
print("TF", tf.__version__)


data = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date").sort_index()
FEATURES = ([f"ret_lag_{l}" for l in range(1, 6)]
            + [f"rvol_{w}" for w in (5, 10, 21)]
            + [f"ma_ratio_{w}" for w in (5, 10, 20)]
            + ["hl_range", "vol_ratio"])
TARGET = "target_vol"; DIDX = data.index; F = len(FEATURES)

FEAT = data[FEATURES].values.astype("float32")
YALL = data[TARGET].values.astype("float32")
RV21 = data["rvol_21"].values.astype("float64")
LV   = 2 * np.log(RV21)
daydiff = DIDX.to_series().diff().dt.days.fillna(9999).values

valid = np.zeros(len(DIDX), dtype=bool)
for i in range(SEQ_LEN - 1, len(DIDX)):
    if not (daydiff[i - SEQ_LEN + 2: i + 1] > GAP_DAYS).any():
        valid[i] = True

THETA_BROAD = float(np.nanmean(LV))
block = pd.Series((daydiff > GAP_DAYS).cumsum(), index=DIDX)
_a = (pd.Series(LV, index=DIDX).groupby(block)
      .transform(lambda s: s.rolling(THETA_WIN, min_periods=10).mean()).values)
THETA_ADAPT = np.where(np.isfinite(_a), _a, THETA_BROAD)

print(f"{len(data)} rows | regimes: {data['regime'].value_counts().to_dict()}")
print(f"valid target-days (common eval set): {int(valid.sum())}")


def pos_of(df):
    p = DIDX.get_indexer(df.index); return p[valid[p]]

def seqs_for(df):
    p = pos_of(df)
    if len(p) == 0:
        return np.empty((0, SEQ_LEN, F), "float32"), np.empty((0,), "float32"), p
    return np.stack([FEAT[q - SEQ_LEN + 1: q + 1] for q in p]), YALL[p], p

def rows_for(df):
    """Tabular rows on the SAME valid day set the sequence models use."""
    p = pos_of(df); return FEAT[p], YALL[p], p

def y_valid(df):
    return YALL[pos_of(df)]

def embargo(train_df, test_df, span=EMBARGO):
    banned = np.zeros(len(DIDX), dtype=bool)
    for p in DIDX.get_indexer(test_df.index):
        banned[max(0, p - span): min(len(DIDX), p + span + 1)] = True
    return train_df[~banned[DIDX.get_indexer(train_df.index)]]

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)),
            "MAPE": float(np.mean(np.abs((y - p) / y)) * 100),
            "R2": float(r2_score(y, p))}


def phi_from(tp):
    sp = np.sort(tp); c = np.where(np.diff(sp) == 1)[0]
    if len(c) > 5:
        return float(np.clip(np.corrcoef(LV[sp[c]], LV[sp[c + 1]])[0, 1], 0.0, 0.999))
    return 0.9

def theta_for(pos, tp, mode=THETA_MODE):
    if mode == "fixed_train": return np.full(len(pos), float(np.nanmean(LV[tp])))
    if mode == "broad":       return np.full(len(pos), THETA_BROAD)
    return THETA_ADAPT[pos]

def vphys(pos, th, phi):
    pb = phi * (1 - phi ** H) / (H * (1 - phi)) if phi < 1 else 1.0
    return np.exp(0.5 * (th + (LV[pos] - th) * pb)).astype("float32")


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

def pred_tree(tr, ev, kind, seed=42):
    Xtr, ytr, _ = rows_for(tr); Xev, _, _ = rows_for(ev)
    if kind == "xgboost":
        m = xgb.XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             n_jobs=-1, random_state=seed)
    else:
        m = RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                                  max_features="sqrt", n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict(Xev)

def pred_anchor(tr, ev):
    tp, ep = pos_of(tr), pos_of(ev)
    return vphys(ep, theta_for(ep, tp), phi_from(tp))

def pred_net(tr, ev, arch, lam, seed):
    """arch='soft' (lam=0 -> Transformer) or 'additive'."""
    X, y, pos = seqs_for(tr)
    phi = phi_from(pos); anchor = vphys(pos, theta_for(pos, pos), phi)
    mu, sd = X.reshape(-1, F).mean(0), X.reshape(-1, F).std(0) + 1e-8
    Xs = (X - mu) / sd
    model = _backbone(seed)
    if arch == "additive":
        resid = y - anchor; r_sd = resid.std() + 1e-8
        Yt = (resid / r_sd).reshape(-1, 1)
        loss = lambda yt, yp: tf.reduce_mean(tf.square(yp - yt)) + lam * tf.reduce_mean(tf.square(yp))
    else:
        ty_mu, ty_sd = y.mean(), y.std() + 1e-8
        Yt = np.concatenate([((y - ty_mu) / ty_sd).reshape(-1, 1),
                             ((anchor - ty_mu) / ty_sd).reshape(-1, 1)], axis=1)
        loss = lambda yt, yp: (tf.reduce_mean(tf.square(yp - yt[:, 0:1]))
                               + lam * tf.reduce_mean(tf.square(yp - yt[:, 1:2])))
    model.compile("adam", loss=loss)
    model.fit(Xs, Yt, epochs=EPOCHS, batch_size=BATCH, verbose=0, validation_split=0.15,
              callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
    Xe, _, pe = seqs_for(ev)
    out = model.predict((Xe - mu) / sd, verbose=0).ravel()
    if arch == "additive":
        return vphys(pe, theta_for(pe, pos), phi) + out * r_sd
    return out * ty_sd + ty_mu


PRED = {}
TRUE = {}

rows = []
for name, trs, te in EXPERIMENTS:
    tr = embargo(data[data["regime"].isin(trs)], data[data["regime"] == te])
    ev = data[data["regime"] == te]
    y = y_valid(ev); TRUE[name] = y
    row = {"exp": name, "train": "+".join(trs), "test": te}

    for kind in ["xgboost", "randomforest"]:
        p = pred_tree(tr, ev, kind); PRED[(name, kind)] = p
        row[kind] = metrics(y, p)["R2"]

    p = pred_anchor(tr, ev); PRED[(name, "anchor")] = p
    row["anchor"] = metrics(y, p)["R2"]

    for label, arch, lam in [("transformer", "soft", 0.0),
                             ("soft_pinn", "soft", LAMBDA),
                             ("additive_pinn", "additive", ADD_LAMBDA)]:
        preds = [pred_net(tr, ev, arch, lam, s) for s in range(N_SEEDS)]
        PRED[(name, label)] = preds
        r2s = [metrics(y, p)["R2"] for p in preds]
        row[label] = float(np.mean(r2s)); row[label + "_sd"] = float(np.std(r2s))
    rows.append(row); print("done", name)

MODELS = ["xgboost", "randomforest", "transformer", "soft_pinn", "additive_pinn", "anchor"]
master = pd.DataFrame(rows).set_index("exp")
master.to_csv("results_master_comparison.csv")
master[["train", "test"] + MODELS]


x = np.arange(len(master)); w = 0.13
fig, ax = plt.subplots(figsize=(13, 5))
colors = dict(xgboost="tab:red", randomforest="tab:orange", transformer="tab:blue",
              soft_pinn="tab:purple", additive_pinn="tab:green", anchor="black")
for i, m in enumerate(MODELS):
    err = master[m + "_sd"] if m + "_sd" in master.columns else None
    ax.bar(x + (i - 2.5) * w, master[m], w, yerr=err, capsize=2,
           label=m.replace("_", "-"), color=colors[m], alpha=0.85)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(master.index)
ax.set_ylabel("cross-regime R²"); ax.set_title("Cross-regime R² by model (higher = better)")
ax.legend(ncol=3); plt.tight_layout()
fig.savefig("fig2_cross_regime_r2.png", dpi=150, bbox_inches="tight")
plt.show()


fig, ax = plt.subplots(figsize=(13, 4))
for i, m in enumerate(MODELS):
    ax.bar(x + (i - 2.5) * w, np.clip(master[m], -3, None), w, label=m.replace("_", "-"),
           color=colors[m], alpha=0.85)
ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-3, 1)
ax.set_xticks(x); ax.set_xticklabels(master.index)
ax.set_ylabel("cross-regime R² (clipped at −3)")
ax.set_title("Same, clipped — only the anchor and additive-PINN clear zero")
ax.legend(ncol=3); plt.tight_layout()
fig.savefig("fig3_cross_regime_r2_clipped.png", dpi=150, bbox_inches="tight")
plt.show()


def block_boot_diff(y, pa, pb, B=BOOT_B, block=BOOT_BLOCK, seed=0):
    rng = np.random.default_rng(seed); n = len(y); nb = int(np.ceil(n / block))
    d = []
    for _ in range(B):
        starts = rng.integers(0, max(1, n - block), nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        d.append(r2_score(y[idx], pa[idx]) - r2_score(y[idx], pb[idx]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def get(name, model):
    p = PRED[(name, model)]
    return np.mean(p, axis=0) if isinstance(p, list) else p

CONTRASTS = [("additive_pinn", "transformer"),
             ("additive_pinn", "soft_pinn"),
             ("anchor", "xgboost"),
             ("anchor", "transformer"),
             ("additive_pinn", "xgboost")]
boot_rows = []
for name, _, _ in EXPERIMENTS:
    y = TRUE[name]
    for a, b in CONTRASTS:
        m, lo, hi = block_boot_diff(y, get(name, a), get(name, b))
        boot_rows.append({"exp": name, "contrast": f"{a} − {b}", "diff_R2": m,
                          "ci_lo": lo, "ci_hi": hi,
                          "significant": bool(lo > 0 or hi < 0)})
boot = pd.DataFrame(boot_rows)
boot.to_csv("results_bootstrap_ci.csv", index=False)
boot.set_index(["exp", "contrast"])


def scatter_panel(exp_name, models=("xgboost", "randomforest", "transformer",
                                    "soft_pinn", "additive_pinn", "anchor")):
    y = TRUE[exp_name]
    fig, axes = plt.subplots(1, len(models), figsize=(3.0 * len(models), 3.3), sharex=True, sharey=True)
    lo = float(min(y.min(), min(get(exp_name, m).min() for m in models)))
    hi = float(max(y.max(), max(get(exp_name, m).max() for m in models)))
    for ax, m in zip(axes, models):
        p = get(exp_name, m)
        ax.scatter(y, p, s=6, alpha=0.35, color=colors[m])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{m.replace('_','-')}\nR²={r2_score(y, p):.2f}", fontsize=10)
        ax.set_xlabel("actual vol")
    axes[0].set_ylabel("predicted vol")
    fig.suptitle(f"{exp_name}: predicted vs actual (dashed = perfect)", y=1.04)
    plt.tight_layout()
    fig.savefig(f"fig1_{exp_name.lower()}_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()

scatter_panel("E2")
scatter_panel("E4")


diag = []
for name, _, te in EXPERIMENTS:
    y = TRUE[name]
    r = {"exp": name, "test": te, "true_mean": float(y.mean())}
    for m in MODELS:
        r[m] = float(get(name, m).mean())
    diag.append(r)
pred_levels = pd.DataFrame(diag).set_index("exp")
pred_levels.to_csv("results_prediction_levels.csv")
print("Mean predicted volatility vs truth (over-prediction = the down-shift failure):")
pred_levels
