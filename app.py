"""
Fuzzy Logic Intrusion Detection System (Mamdani + Sugeno) on NSL-KDD.
Run with:   streamlit run app.py

Only fuzzy logic is used (no machine learning). The project has three files:
  app.py          Streamlit pages and user interaction (this file)
  fuzzy_logic.py  fuzzy variables, rules, inference, data pipeline, metrics
  styles.py       CSS, colours and plotting helpers
"""
import copy
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fuzzy_logic import (
    # constants and fuzzy definitions
    BASE, CLASSES, COLUMNS, DEFAULT_MF, DEFAULT_RULES, R, TEST_PATH, TRAIN_PATH, UNIVERSE, VARIABLES,
    # data pipeline and inference
    get_data, preprocess, split_feature_types, data_quality, feature_relevance,
    mamdani_infer, run_models, rule_analysis, evaluate_config,
    # rules and configuration
    rules_to_text, check_rules, config_json, config_from_json,
    # decisions and evaluation
    get_thresholds, alerts_from_class, confusion, metrics_from, full_metrics, verdict_text,
    # uploaded files and scoring
    read_uploaded_csv, add_labels, score_records,
)

from styles import (
    # palette and CSS
    AMBER, BLUE, CLASS_COLORS, CSS, CYAN, GREEN, PURPLE, RED,
    # HTML and plotting helpers
    card, hero, show, style, gauge, confusion_figure, membership_figure, output_figure,
)


# =============================================================================
# SESSION STATE HELPERS
# =============================================================================


def append_history(frame):
    h = st.session_state["history"]
    frame = frame.copy()
    frame["batch"] = st.session_state["batch_no"] = st.session_state.get("batch_no", 0) + 1
    frame["time"] = pd.Timestamp.now().strftime("%H:%M:%S")
    start = st.session_state.get("event_no", 0)
    frame["event"] = np.arange(start + 1, start + len(frame) + 1)
    st.session_state["event_no"] = start + len(frame)
    h = pd.concat([h, frame], ignore_index=True) if len(h) else frame
    st.session_state["history"] = h.tail(5000).reset_index(drop=True)


def init_state():
    if "mfs" not in st.session_state:
        st.session_state["mfs"] = copy.deepcopy(DEFAULT_MF)
        st.session_state["rules"] = copy.deepcopy(DEFAULT_RULES)
        st.session_state["history"] = pd.DataFrame()


# =============================================================================
# STREAMLIT PAGES
# =============================================================================


# -----------------------------------------------------------------------------
# Page 1: dataset overview
# -----------------------------------------------------------------------------
def page_overview(ev):
    train, test, med = get_data()
    hero("Dataset Inspection & Preprocessing", "What the NSL-KDD files contain and which features the fuzzy systems use")
    num, cat = split_feature_types(train)
    qt, qe = data_quality(train), data_quality(test)
    c = st.columns(5)
    c[0].metric("Training records", f"{qt['rows']:,}")
    c[1].metric("Test records", f"{qe['rows']:,}")
    c[2].metric("Numerical features", len(num))
    c[3].metric("Categorical features", len(cat))
    c[4].metric("Features used by fuzzy rules", len(VARIABLES))
    q = pd.DataFrame({"Training file": qt, "Test file": qe}).T
    q.columns = ["Rows", "Missing values", "Duplicate rows", "Negative values"]
    st.markdown("##### Data quality check")
    st.dataframe(q, width="stretch")
    unseen = sorted(set(test["attack_name"]) - set(train["attack_name"]))
    card(f"<b>Categorical features detected:</b> {', '.join(cat)}.<br>"
         f"<b>Label:</b> the column <i>label</i> holds the traffic type; it is only used to evaluate the system. "
         f"<i>difficulty</i> is a dataset helper column and is ignored.<br>"
         f"<b>Important:</b> the test file contains <b>{len(unseen)} attack types that never appear in the training file</b> "
         f"({', '.join(unseen[:8])}, ...). The test score therefore measures how well the fuzzy rules generalise.")

    left, right = st.columns(2)
    dist = pd.DataFrame({"Training": train["actual"].value_counts(), "Test": test["actual"].value_counts()}).reset_index()
    dist.columns = ["Class", "Training", "Test"]
    dist = dist.melt("Class", var_name="File", value_name="Records")
    fig = px.bar(dist, x="Class", y="Records", color="File", barmode="group",
                 color_discrete_map={"Training": CYAN, "Test": PURPLE}, text_auto=True)
    with left:
        show(style(fig, "Normal vs attack traffic", xtitle="Traffic class", ytitle="Number of records"))
    fam = pd.DataFrame({"Training": train["family"].value_counts(), "Test": test["family"].value_counts()}).fillna(0).reset_index()
    fam.columns = ["Family", "Training", "Test"]
    fam = fam.melt("Family", var_name="File", value_name="Records")
    fig = px.bar(fam, x="Family", y="Records", color="File", barmode="group",
                 color_discrete_map={"Training": CYAN, "Test": PURPLE}, text_auto=True)
    with right:
        show(style(fig, "Attack category distribution", xtitle="Attack family", ytitle="Number of records"))

    rel = feature_relevance(train)
    fig = px.bar(rel.head(20).iloc[::-1], x="relevance", y="feature", orientation="h", color="used_in_fuzzy_model",
                 color_discrete_map={True: CYAN, False: "#4b5a8a"}, hover_data={"auc": ":.3f"})
    show(style(fig, "Feature relevance (training data, single-feature AUC)", 560,
               xtitle="Relevance = |AUC - 0.5| x 2  (0 = useless, 1 = perfect)", ytitle="Feature"))
    card("<b>How features were chosen:</b> every numeric feature was ranked by how well it alone separates normal from attack "
         "traffic. For each group of near-identical features (e.g. <i>serror_rate</i>, <i>srv_serror_rate</i>, "
         "<i>dst_host_serror_rate</i>) only one strong representative was kept so rules stay readable. "
         "Note that <i>num_failed_logins</i> (used in the sample project) has relevance 0.0 in this dataset, so it was dropped.")

    st.markdown("##### Features used by the fuzzy systems")
    t = pd.DataFrame([dict(feature=k, meaning=v["info"], scaling=("log10(1+x) / log10(1+%g)" % v["cap"]) if v["log"]
                           else f"x / {v['cap']:g} (clipped at {v['cap']:g})", train_median=med[k])
                      for k, v in VARIABLES.items()])
    st.dataframe(t, width="stretch", hide_index=True)
    card("<b>Preprocessing (identical for Mamdani and Sugeno):</b> (1) convert to numbers; (2) missing / infinite values "
         "are replaced by the training median; (3) negative values are set to 0; (4) outliers are clipped at the cap shown "
         "above; (5) every feature is scaled to 0..1 (byte counts use a log scale because they are extremely skewed). "
         "The original data had no missing values or duplicates, so steps 2-4 mainly protect manual and uploaded input.")


# -----------------------------------------------------------------------------
# Page 2: membership functions and rules
# -----------------------------------------------------------------------------
def mf_table(mfs):
    rows = []
    for v, sets in mfs.items():
        for s, (k, p) in sets.items():
            q = list(p) + [np.nan] * (4 - len(p))
            rows.append(dict(variable=v, fuzzy_set=s, type=k, p1=q[0], p2=q[1], p3=q[2], p4=q[3]))
    return pd.DataFrame(rows)


def parse_mf_table(df):
    """Validate the edited table; return (new_mfs, error_message)."""
    new = {}
    for _, r in df.iterrows():
        k = r["type"]
        n = {"tri": 3, "trap": 4, "gauss": 2}[k]
        p = [r["p1"], r["p2"], r["p3"], r["p4"]][:n]
        if any(pd.isna(x) for x in p):
            return None, f"{r['variable']}/{r['fuzzy_set']}: a parameter is empty."
        p = [float(x) for x in p]
        if k == "gauss" and p[1] <= 0:
            return None, f"{r['variable']}/{r['fuzzy_set']}: sigma must be > 0."
        if k != "gauss" and p != sorted(p):
            return None, f"{r['variable']}/{r['fuzzy_set']}: parameters must be in increasing order."
        new.setdefault(r["variable"], {})[r["fuzzy_set"]] = (k, p)
    return new, None


def page_rules(ev):
    hero("Fuzzy Membership Functions & Rule Base", "Inspect, analyse and adjust the knowledge of both fuzzy systems")
    mfs, rules = st.session_state["mfs"], st.session_state["rules"]
    t_mf, t_rules, t_an, t_edit = st.tabs(["Membership functions", "Rule base", "Rule analysis", "Adjust & save"])

    with t_mf:
        card("Every input is scaled to 0..1 first. The curves below are drawn in the original units. "
             "<b>Triangular</b> sets describe 'medium' values, <b>trapezoidal</b> sets describe the shoulders (low / high) "
             "and a <b>Gaussian</b> set is used for 'medium' connection counts. Overlapping sets mean a value can belong "
             "to two sets at once (e.g. 0.4 SYN-error rate is partly LOW and partly MEDIUM).")
        names = list(VARIABLES)
        cols = st.columns(2)
        for i, v in enumerate(names):
            with cols[i % 2]:
                show(membership_figure(v, mfs))
        show(output_figure())

    with t_rules:
        card("Both systems share the same <b>IF-part</b> (what traffic looks like) but each has its own <b>THEN-part</b>: "
             "Mamdani uses fuzzy output sets (Low / Medium / High) while Sugeno uses a crisp constant risk value.")
        table = pd.DataFrame([dict(ID=r["id"], IF=rules_to_text(r), Mamdani_THEN=r["mam"] + " risk",
                                   Sugeno_THEN=r["sug"], Weight=r["weight"], Enabled=r["enabled"], Purpose=r["why"])
                              for r in rules])
        st.dataframe(table, width="stretch", hide_index=True, height=640)
        for msg in check_rules(rules):
            st.warning(msg)
        if not check_rules(rules):
            st.success("Rule check: no contradictory or redundant rules found.")

    with t_an:
        card("<b>Rule refinement without data leakage.</b> The rules were studied on 70 % of the <i>training</i> file only. "
             "For each rule we count the records where it fires strongly (> 0.5) and how many of them are real attacks. "
             "A rule that says LOW risk should have a small attack share, a rule that says HIGH risk a large one. "
             "The test file was never used for this.")
        ra = rule_analysis(config_json(mfs, rules))
        fig = px.bar(ra, x="rule", y="attack_share", color="expected", text_auto=".2f",
                     color_discrete_map={"Low": GREEN, "Medium": AMBER, "High": RED},
                     category_orders={"expected": ["Low", "Medium", "High"]})
        show(style(fig, "Share of real attacks among records where each rule fires (training split)",
                   xtitle="Rule", ytitle="Attack share (0 = all normal, 1 = all attacks)"))
        st.dataframe(ra, width="stretch", hide_index=True)

    with t_edit:
        st.markdown("##### Rule editor")
        st.caption("Change the enabled flag, weight (0-1.5), Mamdani output set or Sugeno constant, then press Apply.")
        base = pd.DataFrame([dict(id=r["id"], rule=rules_to_text(r), enabled=r["enabled"], weight=r["weight"],
                                  mamdani_output=r["mam"], sugeno_constant=r["sug"]) for r in rules])
        edited = st.data_editor(base, hide_index=True, width="stretch", key="rule_editor",
                                disabled=["id", "rule"],
                                column_config={"mamdani_output": st.column_config.SelectboxColumn(options=["Low", "Medium", "High"]),
                                               "weight": st.column_config.NumberColumn(min_value=0.0, max_value=1.5, step=0.05),
                                               "sugeno_constant": st.column_config.NumberColumn(min_value=0.0, max_value=100.0)})
        if st.button("Apply rule changes"):
            for r, (_, e) in zip(rules, edited.iterrows()):
                r["enabled"], r["weight"] = bool(e["enabled"]), float(np.clip(e["weight"], 0, 1.5))
                r["mam"], r["sug"] = e["mamdani_output"], float(np.clip(e["sugeno_constant"], 0, 100))
            st.rerun()

        with st.expander("Add a new rule"):
            c = st.columns(3)
            conds = []
            for i in range(3):
                v = c[i].selectbox(f"Condition {i + 1} feature", ["(none)"] + list(VARIABLES), key=f"nv{i}")
                s = c[i].selectbox(f"Condition {i + 1} fuzzy set", list(mfs[v]) if v != "(none)" else ["-"], key=f"ns{i}")
                if v != "(none)":
                    conds.append((v, s))
            d = st.columns(3)
            op = d[0].selectbox("Operator", ["AND", "OR"])
            mam = d[1].selectbox("Mamdani output", ["Low", "Medium", "High"], index=1)
            sug = d[2].number_input("Sugeno constant (0-100)", 0.0, 100.0, 50.0)
            if st.button("Add rule"):
                if not conds:
                    st.error("Choose at least one condition.")
                else:
                    rid = f"U{sum(1 for r in rules if r['id'].startswith('U')) + 1}"
                    rules.append(R(rid, conds, mam, sug, "User-defined rule.", op=op))
                    st.rerun()

        with st.expander("Edit membership function parameters (normalised 0-1 scale)"):
            st.caption("tri = [a, b, c]   trap = [a, b, c, d]   gauss = [mean, sigma]. Parameters must be increasing.")
            ed = st.data_editor(mf_table(mfs), hide_index=True, width="stretch", key="mf_editor",
                                disabled=["variable", "fuzzy_set"],
                                column_config={"type": st.column_config.SelectboxColumn(options=["tri", "trap", "gauss"])})
            if st.button("Apply membership changes"):
                new, err = parse_mf_table(ed)
                if err:
                    st.error(err)
                else:
                    st.session_state["mfs"] = new
                    st.rerun()

        c1, c2, c3 = st.columns(3)
        if c1.button("Reset to default rules and membership functions"):
            st.session_state["mfs"] = copy.deepcopy(DEFAULT_MF)
            st.session_state["rules"] = copy.deepcopy(DEFAULT_RULES)
            st.rerun()
        if c2.button("Save configuration to saved_config/"):
            os.makedirs(os.path.join(BASE, "saved_config"), exist_ok=True)
            with open(os.path.join(BASE, "saved_config", "fuzzy_config.json"), "w") as f:
                f.write(config_json(mfs, rules))
            st.success("Saved to saved_config/fuzzy_config.json")
        if c3.button("Load saved configuration"):
            path = os.path.join(BASE, "saved_config", "fuzzy_config.json")
            if os.path.exists(path):
                m, rl = config_from_json(open(path).read())
                for r in rl:
                    r["cond"] = [tuple(x) for x in r["cond"]]
                st.session_state["mfs"], st.session_state["rules"] = m, rl
                st.rerun()
            else:
                st.error("No saved configuration found.")


# -----------------------------------------------------------------------------
# Page 3: Mamdani vs Sugeno comparison
# -----------------------------------------------------------------------------
def compute_models(ev):
    """Metrics for both models on the TEST file, with the current thresholds."""
    train, test, med = get_data()
    out = {}
    for name, key in (("Mamdani", "mamdani"), ("Sugeno", "sugeno")):
        thr = get_thresholds(key, ev["thr"])
        r = full_metrics(test["actual"].values, ev["test"][key + "_risk"].values, thr)
        r["thr"] = thr
        out[name] = r
    return out


def page_compare(ev):
    hero("Mamdani vs Sugeno Comparison", "Both fuzzy systems evaluated on the official test file with the real dataset labels")
    train, test, med = get_data()
    R_ = compute_models(ev)
    M = {n: dict(r["m"], AUC=r["auc"], ms=ev["ms_per_1000"][n]) for n, r in R_.items()}

    cols = st.columns(2)
    for col, n in zip(cols, ("Mamdani", "Sugeno")):
        with col:
            st.markdown(f"##### {n}")
            a = st.columns(4)
            a[0].metric("Accuracy", f"{M[n]['Accuracy']:.3f}")
            a[1].metric("Recall", f"{M[n]['Recall']:.3f}")
            a[2].metric("False-pos. rate", f"{M[n]['FPR']:.3f}")
            a[3].metric("AUC", f"{M[n]['AUC']:.3f}")

    tab = pd.DataFrame({n: {"Accuracy": M[n]["Accuracy"], "Precision": M[n]["Precision"], "Recall": M[n]["Recall"],
                            "F1-score": M[n]["F1"], "Specificity": M[n]["Specificity"],
                            "False Positive Rate": M[n]["FPR"], "False Negative Rate": M[n]["FNR"],
                            "AUC": M[n]["AUC"], "Time (ms per 1000 rec.)": M[n]["ms"]} for n in M})
    st.dataframe(tab.style.format("{:.4f}"), width="stretch")

    keys = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "FPR", "FNR", "AUC"]
    long = pd.DataFrame([dict(Metric=k, Model=n, Value=M[n][k]) for n in M for k in keys])
    fig = px.bar(long, x="Metric", y="Value", color="Model", barmode="group", text_auto=".3f",
                 color_discrete_map={"Mamdani": CYAN, "Sugeno": PURPLE})
    show(style(fig, "Mamdani vs Sugeno: evaluation metrics (test file)", 420, xtitle="Metric", ytitle="Value (0-1)"))

    c1, c2 = st.columns(2)
    with c1:
        show(confusion_figure(R_["Mamdani"]["c"], "Confusion matrix: Mamdani"))
    with c2:
        show(confusion_figure(R_["Sugeno"]["c"], "Confusion matrix: Sugeno"))

    d1, d2 = st.columns(2)
    fig = go.Figure()
    for n, col in (("Mamdani", CYAN), ("Sugeno", PURPLE)):
        fig.add_trace(go.Scatter(x=R_[n]["fpr"], y=R_[n]["tpr"], mode="lines", line=dict(color=col, width=3),
                                 name=f"{n} (AUC = {R_[n]['auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#667", dash="dash"), name="Random guess"))
    with d1:
        show(style(fig, "ROC curves", 400, xtitle="False positive rate", ytitle="True positive rate (recall)"))
    fig = px.bar(x=list(M), y=[M[n]["ms"] for n in M], color=list(M), text_auto=".1f",
                 color_discrete_map={"Mamdani": CYAN, "Sugeno": PURPLE})
    fig.update_layout(showlegend=False)
    with d2:
        show(style(fig, "Prediction time (measured on 5,000 test records)", 400, xtitle="Model",
                   ytitle="Milliseconds per 1000 records"))

    st.markdown("### Which model is better?")
    card("<br>".join(verdict_text(M)), "good")
    card("<b>Differences between the two inference methods</b><br>"
         "&bull; <b>Mamdani:</b> rule outputs are fuzzy sets (Low / Medium / High). AND = <i>min</i>, OR = <i>max</i>; "
         "each output set is clipped by the rule strength (min implication), clipped sets are combined with <i>max</i> "
         "and the risk is the <i>centroid</i> of the combined shape. Intuitive and smooth, but slower and the score "
         "can never reach exactly 0 or 100.<br>"
         "&bull; <b>Sugeno (zero-order):</b> rule outputs are constants. AND = <i>product</i>, OR = probabilistic sum and "
         "the risk is the <i>weighted average</i> of the constants. No output sets and no centroid, so it is faster, "
         "uses the full 0-100 range and each rule's influence is easy to tune."
         f"<br>&bull; Thresholds (borders Normal | Suspicious | Malicious) - Mamdani: {R_['Mamdani']['thr'][0]:.0f} / "
         f"{R_['Mamdani']['thr'][1]:.0f}, Sugeno: {R_['Sugeno']['thr'][0]:.0f} / {R_['Sugeno']['thr'][1]:.0f}. "
         "They were tuned on a validation split of the training file, never on the test file.")
    if st.session_state.get("susp_alert", True):
        st.caption("Binary evaluation: Suspicious and Malicious both count as an alert (change this in the sidebar).")
    else:
        st.caption("Binary evaluation: only Malicious counts as an alert (change this in the sidebar).")

    st.markdown("##### Detection rate per attack family (test file)")
    rows = []
    for n, r in R_.items():
        alert = alerts_from_class(r["cls"])
        for f_ in ["Normal", "DoS", "Probe", "R2L", "U2R"]:
            m = (test["family"] == f_).values
            if m.sum():
                rows.append(dict(Family=f_, Model=n, Alert_rate=float(alert[m].mean()), Records=int(m.sum())))
    fam = pd.DataFrame(rows)
    fig = px.bar(fam, x="Family", y="Alert_rate", color="Model", barmode="group", text_auto=".2f",
                 hover_data=["Records"], color_discrete_map={"Mamdani": CYAN, "Sugeno": PURPLE})
    show(style(fig, "Share of records flagged as alert, per true family", 400, xtitle="True traffic family",
               ytitle="Alert rate (Normal = false alarms, others = detection)"))


# -----------------------------------------------------------------------------
# Page 4: prediction (manual + CSV)
# -----------------------------------------------------------------------------
def explain_table(m_fire_row, s_fire_row, rules):
    rows = []
    for i, r in enumerate(rules):
        if m_fire_row[i] > 0.001 or s_fire_row[i] > 0.001:
            rows.append(dict(Rule=r["id"], IF=rules_to_text(r), Mamdani_strength=round(float(m_fire_row[i]), 3),
                             Mamdani_output=r["mam"], Sugeno_strength=round(float(s_fire_row[i]), 3),
                             Sugeno_output=r["sug"], Meaning=r["why"]))
    if not rows:
        return pd.DataFrame(columns=["Rule", "IF", "Mamdani_strength", "Mamdani_output", "Sugeno_strength",
                                     "Sugeno_output", "Meaning"])
    return pd.DataFrame(rows).sort_values("Mamdani_strength", ascending=False)


def load_example(family):
    """Callback of the example buttons: copy a real test record into the input widgets."""
    _, test, _ = get_data()
    row = test[test["family"] == family].sample(1, random_state=int(time.time()) % 1000).iloc[0]
    for c in st.session_state["num_cols"]:
        st.session_state[f"in_{c}"] = float(row[c]) if st.session_state["is_float"][c] else int(row[c])
    for c in st.session_state["cat_cols"]:
        st.session_state[f"in_{c}"] = row[c]


def page_predict(ev):
    hero("Intrusion Prediction", "Enter network traffic parameters or upload a CSV file; both fuzzy systems answer instantly")
    train, test, med = get_data()
    num, cat = split_feature_types(train)
    tuned = ev["thr"]
    tab_manual, tab_csv = st.tabs(["Manual input", "CSV batch prediction"])

    # ---------------- manual ----------------
    with tab_manual:
        stats = train[num].agg(["min", "max", "median"])
        is_float = {c: bool((train[c] % 1 != 0).any()) for c in num}
        st.session_state["num_cols"], st.session_state["cat_cols"], st.session_state["is_float"] = num, cat, is_float
        for c in num:
            if f"in_{c}" not in st.session_state:
                st.session_state[f"in_{c}"] = float(stats.loc["median", c]) if is_float[c] else int(stats.loc["median", c])
        for c in cat:
            if f"in_{c}" not in st.session_state:
                st.session_state[f"in_{c}"] = train[c].mode()[0]

        st.markdown("**Load a real example from the test file:**")
        b = st.columns(5)
        for col, fam in zip(b, ["Normal", "DoS", "Probe", "R2L", "U2R"]):
            col.button(fam, key=f"ex_{fam}", on_click=load_example, args=(fam,), width="stretch")

        used = [c for c in num if c in VARIABLES]
        others = [c for c in num if c not in VARIABLES]
        st.markdown('<span class="tag">Used by the fuzzy rules</span>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, c in enumerate(used):
            kw = dict(min_value=0.0 if is_float[c] else 0, step=0.01 if is_float[c] else 1, key=f"in_{c}",
                      help=VARIABLES[c]["info"])
            if is_float[c] and stats.loc["max", c] <= 1:
                kw["max_value"] = 1.0
            cols[i % 4].number_input(f"{c}", **kw)
        with st.expander(f"Other dataset features ({len(others)} numeric + {len(cat)} categorical): "
                         "shown for completeness, not used by the fuzzy rules", expanded=True):
            cols = st.columns(4)
            for i, c in enumerate(others):
                cols[i % 4].number_input(c, min_value=0.0 if is_float[c] else 0, step=0.01 if is_float[c] else 1,
                                         key=f"in_{c}")
            for i, c in enumerate(cat):
                opts = sorted(train[c].unique())
                cols[(len(others) + i) % 4].selectbox(c, opts, key=f"in_{c}")

        record = {c: st.session_state.get(f"in_{c}") for c in num + cat}
        df = pd.DataFrame([record])
        res, m_fire, s_fire, rep, _ = run_models(df, med, st.session_state["mfs"], st.session_state["rules"])
        scored, _, _, _ = score_records(df, "manual", tuned)
        tm, ts = get_thresholds("mamdani", tuned), get_thresholds("sugeno", tuned)
        mc, sc = scored.loc[0, "mamdani_class"], scored.loc[0, "sugeno_class"]

        st.markdown("### Result")
        g1, g2 = st.columns(2)
        with g1:
            show(gauge(res.loc[0, "mamdani_risk"], "Mamdani risk score", *tm))
            st.markdown(f"<h3 style='text-align:center;color:{CLASS_COLORS[mc]}'>Mamdani: {mc}</h3>", unsafe_allow_html=True)
        with g2:
            show(gauge(res.loc[0, "sugeno_risk"], "Sugeno risk score", *ts))
            st.markdown(f"<h3 style='text-align:center;color:{CLASS_COLORS[sc]}'>Sugeno: {sc}</h3>", unsafe_allow_html=True)
        if mc != sc:
            card(f"<b>Disagreement:</b> Mamdani says <b>{mc}</b> but Sugeno says <b>{sc}</b>. "
                 "Check the rule table below to see which rules pull in different directions.", "bad")
        else:
            card(f"<b>Both systems agree:</b> {mc}.", "good")
        if rep["clipped"] or rep["invalid_fixed"] or rep["missing_filled"]:
            st.info(f"Input handling: {rep['clipped']} value(s) clipped to the allowed range, "
                    f"{rep['invalid_fixed']} invalid value(s) set to 0, {rep['missing_filled']} missing value(s) filled with the training median.")

        xn, _ = preprocess(df, med)
        _, _, agg = mamdani_infer(xn, st.session_state["mfs"], st.session_state["rules"], return_agg=True)
        p1, p2 = st.columns(2)
        fig = go.Figure(go.Scatter(x=UNIVERSE, y=agg[0], fill="tozeroy", line=dict(color=CYAN, width=3), name="Aggregated output"))
        fig.add_vline(x=res.loc[0, "mamdani_risk"], line_dash="dash", line_color="#fff",
                      annotation_text=f"centroid = {res.loc[0, 'mamdani_risk']:.1f}")
        with p1:
            show(style(fig, "Mamdani: aggregated output set and centroid", 360, xtitle="Risk score", ytitle="Membership degree"))
        ex = explain_table(m_fire[0], s_fire[0], st.session_state["rules"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ex["Rule"], y=ex["Mamdani_strength"], name="Mamdani strength (min/max)", marker_color=CYAN))
        fig.add_trace(go.Bar(x=ex["Rule"], y=ex["Sugeno_strength"], name="Sugeno strength (product)", marker_color=PURPLE))
        fig.update_layout(barmode="group")
        with p2:
            show(style(fig, "Fuzzy rules activated by this record", 360, xtitle="Rule", ytitle="Firing strength (0-1)"))
        st.markdown("##### Activated rules and their meaning")
        st.dataframe(ex, width="stretch", hide_index=True)

        mem = st.expander("Where does this record sit on the membership functions?")
        with mem:
            cols = st.columns(2)
            for i, v in enumerate(VARIABLES):
                val = float(pd.to_numeric(df.get(v), errors="coerce").fillna(med[v]).iloc[0]) if v in df else med[v]
                val = max(0.0, min(val, VARIABLES[v]["cap"]))
                mark = np.log10(1 + val) if VARIABLES[v]["log"] else val
                with cols[i % 2]:
                    show(membership_figure(v, st.session_state["mfs"], mark))
        if st.button("Submit this record to the Live Monitor"):
            append_history(scored)
            st.success("Record added to the Live Monitor.")

    # ---------------- CSV ----------------
    with tab_csv:
        card("Upload a CSV with NSL-KDD columns (with or without header). Missing columns are filled with training medians, "
             "invalid values are corrected and results can be downloaded. If a <i>label</i> column exists, accuracy is reported.")
        st.download_button("Download a sample CSV (50 test records)", test.head(50)[COLUMNS].to_csv(index=False),
                           "sample_traffic.csv", "text/csv")
        up = st.file_uploader("Upload CSV", type=["csv", "txt"])
        if up is not None:
            df, err = read_uploaded_csv(up)
            if err:
                st.error(err)
            else:
                known = [c for c in VARIABLES if c in df.columns]
                if not known:
                    st.error("None of the fuzzy-model features were found in the file.")
                else:
                    missing = [c for c in VARIABLES if c not in df.columns]
                    if missing:
                        st.warning("Missing columns filled with training medians: " + ", ".join(missing))
                    df = add_labels(df)
                    scored, m_fire, s_fire, rep = score_records(df, "csv", tuned)
                    st.success(f"{len(scored):,} records scored.  Fixed: {rep['missing_filled']} missing, "
                               f"{rep['invalid_fixed']} invalid, {rep['clipped']} clipped values.")
                    cnt = pd.DataFrame({"Mamdani": scored["mamdani_class"].value_counts(),
                                        "Sugeno": scored["sugeno_class"].value_counts()}).reindex(CLASSES).fillna(0).reset_index()
                    cnt.columns = ["Class", "Mamdani", "Sugeno"]
                    cnt = cnt.melt("Class", var_name="Model", value_name="Records")
                    fig = px.bar(cnt, x="Class", y="Records", color="Model", barmode="group", text_auto=True,
                                 color_discrete_map={"Mamdani": CYAN, "Sugeno": PURPLE})
                    a, b = st.columns(2)
                    with a:
                        show(style(fig, "Predicted classes in the uploaded file", 360, xtitle="Predicted class", ytitle="Records"))
                    dis = int((~scored["agree"]).sum())
                    with b:
                        fig = px.histogram(scored.melt(value_vars=["mamdani_risk", "sugeno_risk"], var_name="Model", value_name="Risk"),
                                           x="Risk", color="Model", barmode="overlay", nbins=40, opacity=0.65,
                                           color_discrete_map={"mamdani_risk": CYAN, "sugeno_risk": PURPLE})
                        show(style(fig, "Risk score distribution", 360, xtitle="Risk score", ytitle="Records"))
                    st.metric("Records where the two models disagree", f"{dis:,} ({dis / len(scored):.1%})")
                    if "actual" in df.columns:
                        mm = {}
                        for n, k in (("Mamdani", "mamdani_class"), ("Sugeno", "sugeno_class")):
                            cm = confusion(df["actual"].values == "Attack", alerts_from_class(scored[k].values))
                            mm[n] = metrics_from(cm)
                        st.markdown("##### Accuracy on the uploaded file (using its label column)")
                        st.dataframe(pd.DataFrame(mm).style.format("{:.4f}"), width="stretch")
                    st.markdown("##### Results (first 500 rows)")
                    show_cols = ["actual", "family", "mamdani_risk", "mamdani_class", "sugeno_risk", "sugeno_class", "agree"]
                    st.dataframe(scored[show_cols].head(500), width="stretch")
                    d1, d2 = st.columns(2)
                    d1.download_button("Download all predictions (CSV)", scored.to_csv(index=False), "predictions.csv", "text/csv")
                    if d2.button("Send these records to the Live Monitor"):
                        append_history(scored.head(2000))
                        st.success("Records added to the Live Monitor.")
        else:
            st.info("No file uploaded yet.")


# -----------------------------------------------------------------------------
# Page 5: live-style monitor (simulation / uploaded data)
# -----------------------------------------------------------------------------
def stream_batch(n, tuned):
    _, test, _ = get_data()
    batch = test.sample(n)
    scored, _, _, _ = score_records(batch, "simulation", tuned)
    append_history(scored)


def monitor_view(auto, n, tuned):
    if auto:
        stream_batch(n, tuned)
    h = st.session_state["history"]
    if h is None or len(h) == 0:
        st.info("No traffic yet. Press 'Stream next batch', switch on auto-stream, or send records from the Prediction page.")
        return
    tm, ts = get_thresholds("mamdani", tuned), get_thresholds("sugeno", tuned)
    st.caption("M / S = Mamdani / Sugeno counts")
    k = st.columns(5)
    k[0].metric("Events", f"{len(h):,}")
    for col, n_ in zip(k[1:4], CLASSES):
        col.metric(f"{n_}  (M / S)", f"{(h['mamdani_class'] == n_).sum()} / {(h['sugeno_class'] == n_).sum()}")
    k[4].metric("Model disagreements", f"{(~h['agree']).sum()}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["event"], y=h["mamdani_risk"], mode="lines+markers", name="Mamdani risk",
                             line=dict(color=CYAN)))
    fig.add_trace(go.Scatter(x=h["event"], y=h["sugeno_risk"], mode="lines+markers", name="Sugeno risk",
                             line=dict(color=PURPLE)))
    fig.add_hline(y=tm[1], line_dash="dot", line_color=RED, annotation_text="Mamdani malicious border")
    fig.add_hline(y=tm[0], line_dash="dot", line_color=AMBER, annotation_text="Mamdani suspicious border")
    show(style(fig, "Intrusion risk score of every event", 420, xtitle="Event number", ytitle="Risk score (0-100)"))

    a, b = st.columns(2)
    cnt = pd.DataFrame({"Mamdani": h["mamdani_class"].value_counts(), "Sugeno": h["sugeno_class"].value_counts()}) \
        .reindex(CLASSES).fillna(0).reset_index()
    cnt.columns = ["Class", "Mamdani", "Sugeno"]
    cnt = cnt.melt("Class", var_name="Model", value_name="Events")
    fig = px.bar(cnt, x="Class", y="Events", color="Model", barmode="group", text_auto=True,
                 color_discrete_map={"Mamdani": CYAN, "Sugeno": PURPLE})
    with a:
        show(style(fig, "Traffic classes: Mamdani vs Sugeno", 360, xtitle="Predicted class", ytitle="Events"))
    ag = h["agree"].map({True: "Models agree", False: "Models disagree"}).value_counts().reset_index()
    ag.columns = ["Result", "Events"]
    fig = px.pie(ag, names="Result", values="Events", hole=0.55, color="Result",
                 color_discrete_map={"Models agree": GREEN, "Models disagree": RED})
    with b:
        show(style(fig, "Agreement between the models", 360))

    c, d = st.columns(2)
    known = h[~h["family"].isin(["unknown"])]
    if len(known):
        fam = known["family"].value_counts().reset_index()
        fam.columns = ["Family", "Events"]
        fig = px.bar(fam, x="Family", y="Events", color="Family", text_auto=True,
                     color_discrete_sequence=[GREEN, RED, AMBER, PURPLE, BLUE, CYAN])
        fig.update_layout(showlegend=False)
        with c:
            show(style(fig, "Attack distribution (true family of events with known label)", 360,
                       xtitle="Traffic family", ytitle="Events"))
    else:
        c.info("Events have no label column, so the true attack distribution is unknown.")
    tr = h.groupby("batch").agg(Mamdani=("mamdani_class", lambda s: (s != "Normal").sum()),
                                Sugeno=("sugeno_class", lambda s: (s != "Normal").sum())).reset_index()
    fig = px.line(tr.melt("batch", var_name="Model", value_name="Alerts"), x="batch", y="Alerts", color="Model",
                  markers=True, color_discrete_map={"Mamdani": CYAN, "Sugeno": PURPLE})
    with d:
        show(style(fig, "Detection trend: alerts per batch", 360, xtitle="Batch number", ytitle="Alerts (Suspicious + Malicious)"))

    st.markdown("##### Prediction history (latest 100)")
    cols = ["event", "time", "source", "actual", "family", "mamdani_risk", "mamdani_class", "sugeno_risk", "sugeno_class", "agree"]
    st.dataframe(h[cols].tail(100).iloc[::-1], width="stretch", hide_index=True)


def page_monitor(ev):
    hero("Live-Style Intrusion Monitor", "Replay of test traffic, uploaded files and manual records scored by both fuzzy systems")
    card("<b>Simulation notice:</b> this application is <b>not connected to a real network</b>. The monitor replays random records "
         "from the NSL-KDD test file (or records you upload / enter manually) and scores them with the fuzzy systems. "
         "It does not capture or detect real network traffic.", "sim")
    tuned = ev["thr"]
    c = st.columns([1, 1, 1, 1])
    n = c[0].slider("Records per batch", 1, 50, 10)
    auto = c[1].toggle("Auto-stream every 2 s", value=False)
    if c[2].button("Stream next batch", width="stretch"):
        stream_batch(n, tuned)
    if c[3].button("Clear history", width="stretch"):
        st.session_state["history"] = pd.DataFrame()
        st.session_state["event_no"] = 0
        st.session_state["batch_no"] = 0
    runner = st.fragment(run_every=2)(monitor_view) if auto else st.fragment(monitor_view)
    runner(auto, n, tuned)


# -----------------------------------------------------------------------------
# Page 6: analytics gallery
# -----------------------------------------------------------------------------
def page_analytics(ev):
    hero("Interactive Analytics", "Score distributions, per-model performance, rule activation and trends")
    train, test, med = get_data()
    R_ = compute_models(ev)
    t1, t2, t3, t4, t5 = st.tabs(["Model performance", "Risk score distribution", "Rule activation", "Top attack types",
                                  "Prediction trends"])
    with t1:
        for n in ("Mamdani", "Sugeno"):
            m = R_[n]["m"]
            fig = go.Figure(go.Bar(x=list(m), y=list(m.values()), marker_color=CYAN if n == "Mamdani" else PURPLE,
                                   text=[f"{v:.3f}" for v in m.values()], textposition="outside"))
            show(style(fig, f"{n} model performance (test file)", 360, xtitle="Metric", ytitle="Value (0-1)"))
        a, b = st.columns(2)
        with a:
            show(confusion_figure(R_["Mamdani"]["c"], "Confusion matrix: Mamdani"))
        with b:
            show(confusion_figure(R_["Sugeno"]["c"], "Confusion matrix: Sugeno"))
    with t2:
        long = pd.concat([pd.DataFrame({"Risk": ev["test"][k + "_risk"].values, "Model": n, "Actual": test["actual"].values})
                          for n, k in (("Mamdani", "mamdani"), ("Sugeno", "sugeno"))])
        fig = px.histogram(long, x="Risk", color="Actual", facet_col="Model", barmode="overlay", nbins=50, opacity=0.7,
                           color_discrete_map={"Normal": GREEN, "Attack": RED})
        show(style(fig, "Intrusion risk score distribution by true class (test file)", 450, xtitle="Risk score", ytitle="Records"))
    with t3:
        mfire = pd.DataFrame(ev["m_fire"], columns=[r["id"] for r in st.session_state["rules"]])
        mfire["family"] = test["family"].values
        heat = mfire.groupby("family").mean()
        fig = px.imshow(heat.round(2), text_auto=True, aspect="auto", color_continuous_scale=[[0, "#0b1230"], [1, CYAN]])
        show(style(fig, "Average Mamdani rule activation per true traffic family (test file)", 480,
                   xtitle="Rule", ytitle="True family"))
        st.caption("Bright cells show which rules fire for which kind of traffic: e.g. D-rules for DoS, P-rules for Probe, "
                   "N-rules for Normal.")
    with t4:
        top = test[test["actual"] == "Attack"]["attack_name"].value_counts().head(15).reset_index()
        top.columns = ["Attack", "Records"]
        top["Seen in training"] = top["Attack"].isin(set(train["attack_name"]))
        fig = px.bar(top, x="Attack", y="Records", color="Seen in training", text_auto=True,
                     color_discrete_map={True: CYAN, False: RED})
        show(style(fig, "Most frequent attack types in the test file", 420, xtitle="Attack type", ytitle="Records"))
    with t5:
        h = st.session_state["history"]
        if len(h) == 0:
            st.info("No monitor data yet. Stream records in the Live Monitor or submit predictions first.")
        else:
            h = h.copy()
            h["Mamdani alerts (rolling 20)"] = (h["mamdani_class"] != "Normal").rolling(20, min_periods=1).sum()
            h["Sugeno alerts (rolling 20)"] = (h["sugeno_class"] != "Normal").rolling(20, min_periods=1).sum()
            fig = px.line(h, x="event", y=["Mamdani alerts (rolling 20)", "Sugeno alerts (rolling 20)"],
                          color_discrete_sequence=[CYAN, PURPLE])
            show(style(fig, "Alerts over time (rolling window of 20 events)", 420, xtitle="Event number", ytitle="Alerts in window"))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Fuzzy IDS", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    if not (os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)):
        st.error("Dataset not found. Place KDDTrain_.txt and KDDTest_.txt inside the 'data' folder next to app.py.")
        st.stop()
    cfg = config_json(st.session_state["mfs"], st.session_state["rules"])
    ev = evaluate_config(cfg)

    with st.sidebar:
        st.markdown("## 🛡️ Fuzzy IDS")
        st.caption("Mamdani & Sugeno fuzzy logic · NSL-KDD")
        page = st.radio("Navigation", ["Dataset Overview", "Rules & Membership", "Mamdani vs Sugeno",
                                       "Prediction", "Live Monitor", "Analytics"], label_visibility="collapsed")
        st.divider()
        st.markdown("**Decision borders**")
        st.checkbox("Use borders tuned on validation data", value=True, key="use_tuned")
        if not st.session_state["use_tuned"]:
            for k in ("mamdani", "sugeno"):
                t = ev["thr"][k]
                st.slider(f"{k.title()}: Normal | Suspicious | Malicious", 0, 100, (int(t[0]), int(t[1])), key="thr_" + k)
        else:
            for k in ("mamdani", "sugeno"):
                t = ev["thr"][k]
                st.caption(f"{k.title()}: Normal < {t[0]:.0f} <= Suspicious < {t[1]:.0f} <= Malicious")
        st.checkbox("Count 'Suspicious' as an alert in evaluation", value=True, key="susp_alert")
        st.divider()
        st.caption("Only fuzzy logic is used: no machine-learning models.")

    pages = {"Dataset Overview": page_overview, "Rules & Membership": page_rules, "Mamdani vs Sugeno": page_compare,
             "Prediction": page_predict, "Live Monitor": page_monitor, "Analytics": page_analytics}
    pages[page](ev)


if __name__ == "__main__":
    main()
