"""
fuzzy_logic.py - the fuzzy-logic engine of the Intrusion Detection System.

Contents (in order):
  1. Constants            dataset paths, NSL-KDD columns, attack families
  2. Fuzzy variables      inputs, default membership functions, output sets
  3. Rule base            rule constructor, default rules, rule helpers
  4. Membership functions triangular / trapezoidal / Gaussian
  5. Data pipeline        loading and preprocessing (scaling to 0..1)
  6. Inference            Mamdani and Sugeno systems
  7. Evaluation           confusion matrix, metrics, ROC/AUC, threshold tuning
  8. Cached pipeline      configuration <-> JSON, cached evaluation, decisions
  9. Input handling       uploaded CSV files and scoring of records

Only fuzzy logic is used (no machine learning).
The user interface lives in app.py and the look & feel in styles.py.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# 1. CONSTANTS: dataset paths, columns, attack families
# =============================================================================


BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE, "data", "KDDTrain_.txt")
TEST_PATH = os.path.join(BASE, "data", "KDDTest_.txt")


COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]


# Standard NSL-KDD grouping of the attack names into 4 families
FAMILY = {}
for _name in "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb".split():
    FAMILY[_name] = "DoS"
for _name in "ipsweep nmap portsweep satan mscan saint".split():
    FAMILY[_name] = "Probe"
for _name in "buffer_overflow loadmodule perl rootkit httptunnel ps sqlattack xterm".split():
    FAMILY[_name] = "U2R"
for _name in ("ftp_write guess_passwd imap multihop phf spy warezclient warezmaster "
              "snmpguess snmpgetattack named xlock xsnoop sendmail").split():
    FAMILY[_name] = "R2L"


CLASSES = ["Normal", "Suspicious", "Malicious"]


# =============================================================================
# 2. FUZZY INPUT VARIABLES (selected from the dataset analysis)
# =============================================================================


# Every input is first scaled to the range 0..1 (min-max with a cap).
#   cap   : value that maps to 1.0 (larger values are clipped = outlier handling)
#   log   : True -> log10(1+x) is used before scaling (byte counts are very skewed)
# All membership functions below are defined on this normalised 0..1 scale.
VARIABLES = {
    "serror_rate": dict(label="SYN-error rate", cap=1.0, log=False,
                        info="share of connections with SYN errors (flooding / DoS)"),
    "rerror_rate": dict(label="REJ-error rate", cap=1.0, log=False,
                        info="share of connections rejected (port scanning / probing)"),
    "count": dict(label="Connections to same host", cap=511, log=False,
                  info="connections to the same host in the last 2 seconds"),
    "same_srv_rate": dict(label="Same-service rate", cap=1.0, log=False,
                          info="share of those connections that used the same service"),
    "diff_srv_rate": dict(label="Different-service rate", cap=1.0, log=False,
                          info="share of those connections that used different services (scan)"),
    "dst_host_srv_count": dict(label="Dest-host service count", cap=255, log=False,
                               info="connections to the same service on the destination host"),
    "logged_in": dict(label="Logged in", cap=1.0, log=False,
                      info="1 = login succeeded (real sessions), 0 = no login"),
    "dst_host_count": dict(label="Dest-host connection count", cap=255, log=False,
                           info="connections to the destination host (low = one-shot sweeps across many hosts)"),
    "src_bytes": dict(label="Source bytes", cap=1e7, log=True,
                      info="bytes sent from source to destination (log scale)"),
}
# type, parameters.  tri=[a,b,c]  trap=[a,b,c,d]  gauss=[mean,sigma]
DEFAULT_MF = {
    "serror_rate": {"low": ("trap", [0, 0, 0.05, 0.30]), "medium": ("tri", [0.15, 0.5, 0.85]),
                    "high": ("trap", [0.60, 0.85, 1, 1])},
    "rerror_rate": {"low": ("trap", [0, 0, 0.05, 0.30]), "medium": ("tri", [0.15, 0.5, 0.85]),
                    "high": ("trap", [0.50, 0.80, 1, 1])},
    "count": {"low": ("trap", [0, 0, 0.02, 0.08]), "medium": ("gauss", [0.12, 0.07]),
              "high": ("trap", [0.15, 0.40, 1, 1]), "very_high": ("trap", [0.50, 0.85, 1, 1])},
    "same_srv_rate": {"low": ("trap", [0, 0, 0.30, 0.60]), "medium": ("tri", [0.40, 0.65, 0.90]),
                      "high": ("trap", [0.70, 0.95, 1, 1])},
    "diff_srv_rate": {"low": ("trap", [0, 0, 0.03, 0.15]), "medium": ("tri", [0.05, 0.20, 0.40]),
                      "high": ("trap", [0.20, 0.45, 1, 1])},
    "dst_host_srv_count": {"low": ("trap", [0, 0, 0.10, 0.30]), "medium": ("tri", [0.15, 0.45, 0.80]),
                           "high": ("trap", [0.60, 0.90, 1, 1])},
    "dst_host_count": {"low": ("trap", [0, 0, 0.02, 0.10]), "high": ("trap", [0.30, 0.80, 1, 1])},
    "logged_in": {"no": ("trap", [0, 0, 0.30, 0.60]), "yes": ("trap", [0.40, 0.70, 1, 1])},
    "src_bytes": {"tiny": ("trap", [0, 0, 0.12, 0.20]), "normal": ("tri", [0.15, 0.40, 0.65]),
                  "huge": ("trap", [0.55, 0.75, 1, 1])},
}
# Output (risk score 0..100) fuzzy sets used by the Mamdani system
OUT_MF = {"Low": ("trap", [0, 0, 15, 40]), "Medium": ("tri", [25, 50, 75]),
          "High": ("trap", [60, 85, 100, 100])}
UNIVERSE = np.arange(0, 101, 1.0)          # discretised risk axis for centroid


# =============================================================================
# 3. THE FUZZY RULE BASE
# =============================================================================


# Each rule:  IF <conditions joined by AND/OR> THEN risk is <Mamdani set> (Mamdani)
#                                                   THEN risk = <constant>  (Sugeno, zero-order)
# 'why' explains the cyber-security meaning of the rule.
def R(rid, cond, mam, sug, why, op="AND", weight=1.0):
    return dict(id=rid, cond=cond, op=op, mam=mam, sug=float(sug), why=why,
                weight=weight, enabled=True)


DEFAULT_RULES = [
    # ---- evidence of NORMAL traffic -------------------------------------------------
    R("N1", [("logged_in", "yes"), ("serror_rate", "low"), ("same_srv_rate", "high")], "Low", 5,
      "A completed login with clean TCP handshakes and one steady service is typical user activity."),
    R("N2", [("serror_rate", "low"), ("rerror_rate", "low"), ("diff_srv_rate", "low"), ("count", "low")],
      "Low", 10, "Few connections, no handshake or reject errors and no service hopping: quiet normal traffic."),
    R("N3", [("logged_in", "yes"), ("dst_host_srv_count", "high"), ("serror_rate", "low")], "Low", 8,
      "A busy, healthy service that many authenticated users share (web / mail / ftp)."),
    R("N4", [("logged_in", "no"), ("same_srv_rate", "high"), ("dst_host_srv_count", "high"),
             ("serror_rate", "low"), ("rerror_rate", "low"), ("count", "low")], "Low", 20,
      "Short lookups (DNS / ping) to a popular service with no errors and a small connection count. "
      "Weight 0.7 because some ping sweeps look similar.", weight=0.7),
    R("N5", [("logged_in", "no"), ("same_srv_rate", "high"), ("dst_host_srv_count", "high"),
             ("serror_rate", "low"), ("rerror_rate", "low"), ("count", "medium")], "Low", 15,
      "Moderate bursts of clean lookups to a popular service: ordinary name-server / web-browsing activity."),
    R("N6", [("rerror_rate", "high"), ("count", "low"), ("dst_host_srv_count", "high")], "Low", 20,
      "A single rejected connection to a popular web service is a normal closed-port or timeout event."),
    R("B1", [("serror_rate", "low"), ("rerror_rate", "low")], "Low", 20,
      "Baseline: no handshake or reject errors means no flood/scan evidence. Low weight so it never hides other rules.", weight=0.4),
    # ---- DoS / flooding -------------------------------------------------------------
    R("D1", [("serror_rate", "high"), ("count", "high")], "High", 97,
      "Many connections that never complete the TCP handshake: SYN flood (neptune)."),
    R("D2", [("serror_rate", "high"), ("same_srv_rate", "low")], "High", 94,
      "SYN errors spread over random services/ports: spoofed flood traffic."),
    R("D3", [("serror_rate", "high"), ("logged_in", "no")], "High", 92,
      "Handshake failures with no login: nobody is really using the service."),
    R("D4", [("count", "very_high"), ("logged_in", "no"), ("src_bytes", "normal")], "High", 90,
      "Extreme connection bursts to one host with identical payload sizes and no login: smurf / ICMP flood."),
    # ---- Probe / scanning -----------------------------------------------------------
    R("P1", [("diff_srv_rate", "high"), ("rerror_rate", "high")], "High", 92,
      "Many different services hit and rejected: port scan (satan, portsweep)."),
    R("P2", [("rerror_rate", "high"), ("count", "high")], "High", 85,
      "A burst of rejected connections: attacker probing closed ports."),
    R("P3", [("diff_srv_rate", "high"), ("count", "high")], "High", 85,
      "Fast service hopping inside a short time window: scanning tool at work."),
    R("P4", [("src_bytes", "tiny"), ("logged_in", "no"), ("dst_host_count", "low")], "Medium", 70,
      "Empty probe packets, each to a different host, no login: ping sweep (ipsweep, nmap)."),
    R("P5", [("rerror_rate", "high"), ("dst_host_srv_count", "low")], "Medium", 62,
      "Rejects against a service that is almost unused: probing for open ports."),
    # ---- Suspicious sessions --------------------------------------------------------
    R("S1", [("logged_in", "yes"), ("src_bytes", "huge")], "Medium", 45,
      "Authenticated session pushing an unusually large upload: possible data planting or a bulk transfer to review."),
    R("S2", [("same_srv_rate", "low"), ("count", "high"), ("logged_in", "no")], "Medium", 65,
      "Many services touched in a burst without logging in: reconnaissance."),
]


def rules_to_text(rule):
    return (" " + rule["op"] + " ").join(f"{VARIABLES[v]['label']} is {s.upper()}" for v, s in rule["cond"])


def check_rules(rules):
    """Detect duplicate or contradictory rules (same IF part)."""
    msgs, seen = [], {}
    for r in rules:
        key = (r["op"], tuple(sorted(map(tuple, r["cond"]))))
        if key in seen:
            other = seen[key]
            if other["mam"] != r["mam"]:
                msgs.append(f"Contradiction: {other['id']} and {r['id']} have the same IF part but different outputs.")
            else:
                msgs.append(f"Redundant: {other['id']} and {r['id']} have the same IF part and output.")
        seen[key] = r
    return msgs


# =============================================================================
# 4. MEMBERSHIP FUNCTIONS
# =============================================================================


def membership(kind, p, x):
    """Membership degree (0..1) of the values x for one fuzzy set."""
    x = np.asarray(x, dtype=float)
    if kind == "tri":                      # triangular  [a, b, c]
        a, b, c = p
        left = (x - a) / (b - a) if b > a else (x >= b).astype(float)
        right = (c - x) / (c - b) if c > b else (x <= b).astype(float)
        return np.clip(np.minimum(left, right), 0, 1)
    if kind == "trap":                     # trapezoidal [a, b, c, d]
        a, b, c, d = p
        left = (x - a) / (b - a) if b > a else np.ones_like(x)
        right = (d - x) / (d - c) if d > c else np.ones_like(x)
        return np.clip(np.minimum(np.minimum(left, 1), right), 0, 1)
    if kind == "gauss":                    # Gaussian [mean, sigma]
        m, s = p
        return np.exp(-((x - m) ** 2) / (2 * max(s, 1e-6) ** 2))
    raise ValueError("unknown membership type " + str(kind))


# =============================================================================
# 5. DATA LOADING AND PREPROCESSING
# =============================================================================


@st.cache_data(show_spinner=False)
def load_dataset(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["attack_name"] = df["label"].str.strip().str.lower()
    df["actual"] = np.where(df["attack_name"] == "normal", "Normal", "Attack")
    df["family"] = df["attack_name"].map(lambda n: "Normal" if n == "normal" else FAMILY.get(n, "Other"))
    return df


def split_feature_types(df):
    """Detect numerical / categorical features automatically (not hard-coded)."""
    skip = {"label", "attack_name", "actual", "family", "difficulty"}
    cols = [c for c in df.columns if c not in skip]
    cat = [c for c in cols if df[c].dtype == object or str(df[c].dtype).startswith("str")]
    num = [c for c in cols if c not in cat]
    return num, cat


def data_quality(df):
    return dict(rows=len(df), missing=int(df.isna().sum().sum()), duplicates=int(df.duplicated().sum()),
                negatives=int((df.select_dtypes("number") < 0).sum().sum()))


def feature_relevance(df):
    """Single-feature AUC on the TRAINING data: how well one feature alone separates attacks.
    Relevance = |AUC - 0.5| * 2   (0 = useless, 1 = perfect)."""
    y = (df["actual"] == "Attack").values
    n1, n0 = y.sum(), (~y).sum()
    rows = []
    for c in split_feature_types(df)[0]:
        if df[c].nunique() < 2:
            rows.append((c, 0.5, 0.0))
            continue
        ranks = df[c].rank().values
        auc = (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        rows.append((c, auc, abs(auc - 0.5) * 2))
    out = pd.DataFrame(rows, columns=["feature", "auc", "relevance"]).sort_values("relevance", ascending=False)
    out["used_in_fuzzy_model"] = out["feature"].isin(VARIABLES)
    return out.reset_index(drop=True)


def fit_medians(train_df):
    return {v: float(train_df[v].median()) for v in VARIABLES}


def preprocess(df, medians):
    """Raw records -> normalised fuzzy inputs (0..1). Same function is used for both models.
    Steps: numeric conversion, replace missing/invalid with TRAIN median, clip outliers, scale."""
    out = pd.DataFrame(index=df.index)
    report = dict(missing_filled=0, invalid_fixed=0, clipped=0)
    for v, cfg in VARIABLES.items():
        if v in df.columns:
            raw = pd.to_numeric(df[v], errors="coerce")
        else:
            raw = pd.Series(np.nan, index=df.index)
        bad = raw.isna() | np.isinf(raw)
        report["missing_filled"] += int(bad.sum())
        raw = raw.mask(bad, medians[v])
        neg = raw < 0
        report["invalid_fixed"] += int(neg.sum())
        raw = raw.mask(neg, 0)
        over = raw > cfg["cap"]
        report["clipped"] += int(over.sum())
        raw = raw.clip(upper=cfg["cap"])
        if cfg["log"]:
            out[v] = np.log10(1 + raw) / np.log10(1 + cfg["cap"])
        else:
            out[v] = raw / cfg["cap"]
    return out, report


def fuzzify(xn, mfs):
    """Membership degrees of every fuzzy set: dict[(variable,set)] -> array"""
    return {(v, s): membership(k, p, xn[v].values) for v, sets in mfs.items() for s, (k, p) in sets.items()}


# =============================================================================
# 6. THE TWO FUZZY INFERENCE SYSTEMS
# =============================================================================


def rule_strength(rule, mu, how):
    """Firing strength of one rule.
    Mamdani: AND = min, OR = max            Sugeno: AND = product, OR = probabilistic sum"""
    parts = [mu[(v, s)] for v, s in rule["cond"]]
    f = parts[0].copy()
    for m in parts[1:]:
        if how == "mamdani":
            f = np.minimum(f, m) if rule["op"] == "AND" else np.maximum(f, m)
        else:
            f = f * m if rule["op"] == "AND" else f + m - f * m
    return f * rule["weight"] if rule["enabled"] else np.zeros_like(f)


def mamdani_infer(xn, mfs, rules, out_mf=OUT_MF, return_agg=False):
    """Mamdani: fuzzify -> min/max rules -> min IMPLICATION (clip output set) ->
    max AGGREGATION -> CENTROID defuzzification.  Returns (risk 0..100, firing matrix)."""
    mu = fuzzify(xn, mfs)
    fire = np.column_stack([rule_strength(r, mu, "mamdani") for r in rules])      # (N, rules)
    n = len(xn)
    aggregated = np.zeros((n, len(UNIVERSE)))
    for label, (k, p) in out_mf.items():
        idx = [i for i, r in enumerate(rules) if r["mam"] == label]
        if not idx:
            continue
        strength = np.clip(fire[:, idx].max(axis=1), 0, 1)                       # strongest rule for this label
        clipped = np.minimum(strength[:, None], membership(k, p, UNIVERSE)[None, :])   # implication = min
        aggregated = np.maximum(aggregated, clipped)                             # aggregation = max
    area = aggregated.sum(axis=1)
    risk = np.where(area > 1e-9, (aggregated * UNIVERSE).sum(axis=1) / np.maximum(area, 1e-9), 50.0)
    if return_agg:
        return risk, fire, aggregated
    return risk, fire


def sugeno_infer(xn, mfs, rules):
    """Zero-order Sugeno: AND = product -> rule output is a constant z_k ->
    WEIGHTED AVERAGE  sum(w_k*z_k)/sum(w_k).  No output membership functions, no centroid."""
    mu = fuzzify(xn, mfs)
    fire = np.column_stack([rule_strength(r, mu, "sugeno") for r in rules])
    z = np.array([r["sug"] for r in rules])
    total = fire.sum(axis=1)
    risk = np.where(total > 1e-9, (fire * z).sum(axis=1) / np.maximum(total, 1e-9), 50.0)
    return np.clip(risk, 0, 100), fire


def classify(score, t_normal, t_malicious):
    return np.where(score < t_normal, "Normal", np.where(score < t_malicious, "Suspicious", "Malicious"))


def run_models(df, medians, mfs, rules):
    """Run BOTH models on raw records; returns result frame and the two firing matrices."""
    xn, rep = preprocess(df, medians)
    t0 = time.perf_counter()
    m_score, m_fire = mamdani_infer(xn, mfs, rules)
    t1 = time.perf_counter()
    s_score, s_fire = sugeno_infer(xn, mfs, rules)
    t2 = time.perf_counter()
    res = pd.DataFrame({"mamdani_risk": m_score, "sugeno_risk": s_score}, index=df.index)
    timing = dict(mamdani_ms_per_1000=(t1 - t0) / max(len(df), 1) * 1e6,
                  sugeno_ms_per_1000=(t2 - t1) / max(len(df), 1) * 1e6)
    return res, m_fire, s_fire, rep, timing


# =============================================================================
# 7. EVALUATION HELPERS (written by hand: confusion matrix, metrics, ROC, AUC)
# =============================================================================


def split_train(train_df, seed=42, fit_fraction=0.7):
    """Fixed split of the TRAINING file: 'fit' part for studying the rules, 'val' part for threshold
    tuning. The official test file is never used for any tuning (no data leakage)."""
    order = np.random.RandomState(seed).permutation(len(train_df))
    cut = int(fit_fraction * len(train_df))
    return train_df.iloc[order[:cut]], train_df.iloc[order[cut:]]


def confusion(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return dict(TP=int((y & pred).sum()), TN=int((~y & ~pred).sum()),
                FP=int((~y & pred).sum()), FN=int((y & ~pred).sum()))


def metrics_from(c):
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return dict(Accuracy=div(tp + tn, tp + tn + fp + fn), Precision=prec, Recall=rec,
                F1=div(2 * prec * rec, prec + rec), Specificity=div(tn, tn + fp),
                FPR=div(fp, fp + tn), FNR=div(fn, fn + tp))


def roc_curve(y, score):
    """Return fpr, tpr, auc computed from the continuous risk score."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-score, kind="stable")
    ys, ss = y[order], np.asarray(score)[order]
    tps, fps = np.cumsum(ys), np.cumsum(1 - ys)
    keep = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]          # one point per distinct score
    tpr = np.r_[0, tps[keep] / max(tps[-1], 1)]
    fpr = np.r_[0, fps[keep] / max(fps[-1], 1)]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def tune_thresholds(score, y):
    """Choose the two class borders from VALIDATION data only.
    t_normal   : risk above which traffic is 'Suspicious' -> maximises (recall + specificity) / 2
    t_malicious: lowest risk above which at least 97 % of the flagged records are real attacks."""
    y = np.asarray(y).astype(bool)
    best_t, best = 50.0, -1
    for t in np.arange(1, 100, 1.0):
        p = score >= t
        bal = ((p & y).sum() / max(y.sum(), 1) + (~p & ~y).sum() / max((~y).sum(), 1)) / 2
        if bal > best:
            best, best_t = bal, t
    t_mal = None
    for t in np.arange(best_t + 5, 100, 1.0):
        p = score >= t
        if p.sum() >= 50 and y[p].mean() >= 0.97:
            t_mal = t
            break
    if t_mal is None:
        t_mal = best_t + (100 - best_t) / 2
    return float(best_t), float(t_mal)


# =============================================================================
# 8. CONFIGURATION, CACHED PIPELINE AND DECISION HELPERS
# =============================================================================


# ---- configuration (membership functions + rules) is kept in session_state and ------------------
# ---- turned into JSON text, which is used as the cache key for the heavy calculations. ----------
def config_json(mfs, rules):
    slim = [{k: r[k] for k in ("id", "cond", "op", "mam", "sug", "weight", "enabled", "why")} for r in rules]
    return json.dumps({"mfs": mfs, "rules": slim}, sort_keys=True)


def config_from_json(text):
    d = json.loads(text)
    mfs = {v: {s: (k, p) for s, (k, p) in sets.items()} for v, sets in d["mfs"].items()}
    return mfs, d["rules"]


@st.cache_data(show_spinner=False)
def get_data():
    train, test = load_dataset(TRAIN_PATH), load_dataset(TEST_PATH)
    return train, test, fit_medians(train)


@st.cache_data(show_spinner="Running both fuzzy systems on the dataset ...")
def evaluate_config(cfg_text):
    """Heavy part, cached. Tunes thresholds on the validation split and scores the test file."""
    mfs, rules = config_from_json(cfg_text)
    train, test, med = get_data()
    fit, val = split_train(train)
    v_res, _, _, _, _ = run_models(val, med, mfs, rules)
    y_val = (val["actual"] == "Attack").values
    thr = {m: tune_thresholds(v_res[m + "_risk"].values, y_val) for m in ("mamdani", "sugeno")}
    t_res, m_fire, s_fire, rep, _ = run_models(test, med, mfs, rules)
    # timing: median of 3 runs on 5000 test records
    sample = test.sample(5000, random_state=1)
    xn, _ = preprocess(sample, med)
    times = {"Mamdani": [], "Sugeno": []}
    for _ in range(3):
        a = time.perf_counter(); mamdani_infer(xn, mfs, rules)
        b = time.perf_counter(); sugeno_infer(xn, mfs, rules)
        c = time.perf_counter()
        times["Mamdani"].append(b - a); times["Sugeno"].append(c - b)
    ms_per_1000 = {k: float(np.median(v)) / len(sample) * 1000 * 1000 for k, v in times.items()}
    return dict(thr=thr, test=t_res, m_fire=m_fire, s_fire=s_fire, ms_per_1000=ms_per_1000,
                val_auc={m: roc_curve(y_val, v_res[m + "_risk"].values)[2] for m in ("mamdani", "sugeno")})


@st.cache_data(show_spinner="Analysing rules on the training split ...")
def rule_analysis(cfg_text):
    """For every rule: how often does it fire on the FIT part of the training data and how many of
    those records are real attacks? Used to refine rules without touching the test file."""
    mfs, rules = config_from_json(cfg_text)
    train, _, med = get_data()
    fit, _ = split_train(train)
    xn, _ = preprocess(fit, med)
    _, fire = mamdani_infer(xn, mfs, rules)
    y = (fit["actual"] == "Attack").values
    rows = []
    for i, r in enumerate(rules):
        act = fire[:, i] > 0.5 * max(r["weight"], 0.05)
        share = float(y[act].mean()) if act.sum() else np.nan
        if act.sum() == 0:
            verdict = "never fires"
        elif r["mam"] == "Low" and share > 0.20:
            verdict = "review: too many attacks"
        elif r["mam"] == "High" and share < 0.70:
            verdict = "review: too many normals"
        else:
            verdict = "consistent"
        rows.append(dict(rule=r["id"], records_fired=int(act.sum()), attack_share=share,
                         expected=r["mam"], verdict=verdict))
    return pd.DataFrame(rows)


def get_thresholds(model, tuned):
    if st.session_state.get("use_tuned", True):
        return tuned[model]
    return st.session_state["thr_" + model]


def alerts_from_class(cls):
    """Binary decision used for the evaluation: Suspicious counts as an alert (default)."""
    cls = np.asarray(cls)
    return (cls != "Normal") if st.session_state.get("susp_alert", True) else (cls == "Malicious")


def full_metrics(actual, risk, thr):
    y = (np.asarray(actual) == "Attack")
    cls = classify(risk, *thr)
    c = confusion(y, alerts_from_class(cls))
    m = metrics_from(c)
    fpr, tpr, auc = roc_curve(y, risk)
    return dict(m=m, c=c, fpr=fpr, tpr=tpr, auc=auc, cls=cls)


def verdict_text(M):
    """Plain-English comparison generated from the measured numbers (never hard-coded results)."""
    a, b = M["Mamdani"], M["Sugeno"]
    pick = lambda key, hi=True: "Mamdani" if ((a[key] > b[key]) == hi) else "Sugeno"
    lines = [
        f"**Overall (F1-score):** {pick('F1')} is better "
        f"(Mamdani {a['F1']:.3f} vs Sugeno {b['F1']:.3f}). "
        f"Accuracy: Mamdani {a['Accuracy']:.3f} vs Sugeno {b['Accuracy']:.3f}.",
        f"**Attack detection (recall):** {pick('Recall')} detects more attacks "
        f"(Mamdani {a['Recall']:.3f} vs Sugeno {b['Recall']:.3f}).",
        f"**False positives:** {pick('FPR', hi=False)} raises fewer false alarms "
        f"(FPR Mamdani {a['FPR']:.3f} vs Sugeno {b['FPR']:.3f}).",
        f"**Ranking quality (AUC):** {pick('AUC')} separates normal and attack scores better "
        f"(Mamdani {a['AUC']:.3f} vs Sugeno {b['AUC']:.3f}).",
        f"**Speed:** {pick('ms', hi=False)} is faster "
        f"(Mamdani {a['ms']:.1f} ms vs Sugeno {b['ms']:.1f} ms per 1000 records).",
    ]
    return lines


# =============================================================================
# 9. INPUT HANDLING FOR UPLOADED FILES / MANUAL ENTRY
# =============================================================================


def read_uploaded_csv(file):
    """Read a CSV; accept files with a header row or raw NSL-KDD files without header."""
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return None, f"Could not read the file: {exc}"
    if df.empty:
        return None, "The uploaded file is empty."
    if not any(c in df.columns for c in COLUMNS):            # probably no header row
        file.seek(0)
        raw = pd.read_csv(file, header=None)
        if raw.shape[1] in (41, 42, 43):
            raw.columns = COLUMNS[:raw.shape[1]]
            df = raw
        else:
            return None, "No known NSL-KDD column names found and the column count is not 41-43."
    return df, None


def add_labels(df):
    """If the data contains a 'label' column, derive actual class and family."""
    df = df.copy()
    if "label" in df.columns:
        names = df["label"].astype(str).str.strip().str.lower().str.rstrip(".")
        df["actual"] = np.where(names == "normal", "Normal", "Attack")
        df["family"] = names.map(lambda n: "Normal" if n == "normal" else FAMILY.get(n, "Other"))
    return df


def score_records(df, source, tuned):
    """Score raw records with both models and return a tidy frame ready for the history table."""
    _, _, med = get_data()
    mfs, rules = st.session_state["mfs"], st.session_state["rules"]
    res, m_fire, s_fire, rep, _ = run_models(df, med, mfs, rules)
    tm, ts = get_thresholds("mamdani", tuned), get_thresholds("sugeno", tuned)
    out = pd.DataFrame(index=df.index)
    out["source"] = source
    out["actual"] = df["actual"].values if "actual" in df.columns else "unknown"
    out["family"] = df["family"].values if "family" in df.columns else "unknown"
    out["mamdani_risk"] = res["mamdani_risk"].round(2)
    out["sugeno_risk"] = res["sugeno_risk"].round(2)
    out["mamdani_class"] = classify(res["mamdani_risk"].values, *tm)
    out["sugeno_class"] = classify(res["sugeno_risk"].values, *ts)
    out["agree"] = out["mamdani_class"] == out["sugeno_class"]
    for v in VARIABLES:
        out[v] = df[v].values if v in df.columns else np.nan
    return out.reset_index(drop=True), m_fire, s_fire, rep
