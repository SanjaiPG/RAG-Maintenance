"""
End-to-end evaluation: XGBoost prediction -> SHAP -> case-specific retrieval
-> claim-level answer with citations -> claim-level support check.
"""

import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"

TEST_FILE = "data/processed/test.csv"
MODEL_FILE = "models/xgboost.joblib"
VECTORIZER_FILE = "data/knowledge_base/index/tfidf_vectorizer.joblib"
MATRIX_FILE = "data/knowledge_base/index/tfidf_matrix.joblib"
PASSAGES_FILE = "data/knowledge_base/index/passages.joblib"

OUTPUT_FILE = "results/end_to_end_evaluation.csv"

THRESHOLD = 0.50
WATCH_LOW = 0.10          # below threshold but not negligible -> "watch" band

N_FAILED = 10
N_NON_FAILED = 10
RANDOM_STATE = 42

TOP_K_RETRIEVAL = 5
RETRIEVAL_POOL = 50       # candidates examined before filtering
MIN_SCORE = 0.02          # minimum cosine similarity
MAX_PER_SOURCE = 2        # max passages from one source
MIN_ACTION_TERMS = 1      # passage must mention >= this many action terms

N_SHAP_FACTORS = 3        # top SHAP drivers used in query/question

ACTION_TERM_PATTERN = re.compile(
    r"vibration|thermal|temperature|overheat\w*|lubric\w*|wear|torque|"
    r"motor current|inspect\w*|baseline|ultrasonic\w*",
    re.IGNORECASE,
)

# Feature-name prefixes a ColumnTransformer may add
TRANSFORMER_PREFIXES = (
    "categorical__", "numerical__", "cat__", "num__", "remainder__"
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_model():
    return joblib.load(MODEL_FILE)


def load_test_data():
    return pd.read_csv(TEST_FILE)


def load_retrieval_index():
    vectorizer = joblib.load(VECTORIZER_FILE)
    matrix = joblib.load(MATRIX_FILE)
    passages = joblib.load(PASSAGES_FILE)
    return vectorizer, matrix, passages


# --------------------------------------------------------------------------
# Prediction and SHAP
# --------------------------------------------------------------------------

def map_feature(transformed_name):
    """Map a transformed column name back to one of FEATURES. Never skips."""
    base = transformed_name
    for prefix in TRANSFORMER_PREFIXES:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    if base == "Type" or base.startswith("Type_"):
        return "Type"

    if base in FEATURES:
        return base

    raise ValueError(
        f"Cannot map transformed feature '{transformed_name}' to any of "
        f"{FEATURES}. Update TRANSFORMER_PREFIXES / map_feature."
    )


def compute_shap_matrix(model, X):
    """
    Return an (n_cases, len(FEATURES)) array of SHAP values aggregated back to
    the original features. Units are log-odds (XGBoost margin space).
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_t = preprocessor.transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    X_t = np.asarray(X_t, dtype=float)

    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(X_t)

    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    names = list(preprocessor.get_feature_names_out())
    if values.shape[1] != len(names):
        raise RuntimeError(
            f"SHAP columns ({values.shape[1]}) != transformed features "
            f"({len(names)})."
        )

    column_to_feature = [map_feature(n) for n in names]

    aggregated = np.zeros((values.shape[0], len(FEATURES)))
    for col, feature in enumerate(column_to_feature):
        aggregated[:, FEATURES.index(feature)] += values[:, col]

    if np.allclose(aggregated, 0.0):
        raise RuntimeError(
            "All aggregated SHAP values are zero. Check feature mapping:\n"
            f"{names}"
        )

    # Additivity check: base value + sum(SHAP) should reproduce the model
    # probability (via sigmoid).
    base = float(np.ravel(explainer.expected_value)[0])
    prob_from_shap = 1.0 / (1.0 + np.exp(-(base + values.sum(axis=1))))
    prob_model = model.predict_proba(X)[:, 1]
    max_gap = float(np.max(np.abs(prob_from_shap - prob_model)))
    if max_gap > 0.01:
        print(
            f"WARNING: SHAP additivity gap {max_gap:.4f} "
            "(base + sum(SHAP) does not reproduce predict_proba)."
        )

    return aggregated


def rank_factors(shap_row):
    factors = []
    for feature, value in zip(FEATURES, shap_row):
        value = float(value)
        if value > 1e-12:
            direction = "Pushes toward failure"
        elif value < -1e-12:
            direction = "Pushes away from failure"
        else:
            direction = "No contribution"
        factors.append(
            {"feature": feature, "shap": value, "direction": direction}
        )
    factors.sort(key=lambda f: abs(f["shap"]), reverse=True)
    return factors


# --------------------------------------------------------------------------
# Case conditions (screens based on the AI4I failure-mode definitions)
# --------------------------------------------------------------------------

def reference_stats(test_df):
    return {
        "torque_hi": float(test_df["Torque [Nm]"].quantile(0.90)),
        "rpm_lo": float(test_df["Rotational speed [rpm]"].quantile(0.10)),
        "rpm_hi": float(test_df["Rotational speed [rpm]"].quantile(0.90)),
        "air_hi": float(test_df["Air temperature [K]"].quantile(0.90)),
    }


def derive_conditions(row, ref):
    """
    Boolean screens computed from the raw sensor values. The thresholds for
    heat dissipation, power band, overstrain and tool wear mirror the failure
    modes documented for the AI4I 2020 dataset; verify them against your
    dataset documentation.
    """
    air = float(row["Air temperature [K]"])
    proc = float(row["Process temperature [K]"])
    rpm = float(row["Rotational speed [rpm]"])
    torque = float(row["Torque [Nm]"])
    wear = float(row["Tool wear [min]"])

    delta_t = proc - air
    power_w = torque * rpm * 2.0 * np.pi / 60.0
    overstrain_limit = {"L": 11000, "M": 12000, "H": 13000}.get(
        str(row["Type"]), 12000
    )

    flags = {
        "high_torque": torque >= ref["torque_hi"],
        "low_speed": rpm <= ref["rpm_lo"],
        "high_speed": rpm >= ref["rpm_hi"],
        "high_tool_wear": wear >= 200,
        "high_air_temp": air >= ref["air_hi"],
        "low_heat_dissipation": (delta_t < 8.6) and (rpm < 1380),
        "power_out_of_band": (power_w < 3500) or (power_w > 9000),
        "overstrain": (wear * torque) > overstrain_limit,
    }
    flags = {k: bool(v) for k, v in flags.items()}

    derived = {
        "delta_t_K": round(delta_t, 2),
        "power_W": round(power_w, 1),
        "wear_x_torque": round(wear * torque, 1),
    }
    return flags, derived


def active_flags(flags):
    return [name for name, on in flags.items() if on]


def toward_failure_features(factors, n=N_SHAP_FACTORS):
    top = factors[:n]
    return [f["feature"] for f in top if f["shap"] > 1e-12]


# --------------------------------------------------------------------------
# Question and retrieval query
# --------------------------------------------------------------------------

def build_question(row, probability, factors, flags):
    factor_text = ", ".join(
        f"{f['feature']} ({f['direction'].lower()}, "
        f"SHAP={f['shap']:+.3f} log-odds)"
        for f in factors[:N_SHAP_FACTORS]
    )
    flag_text = ", ".join(active_flags(flags)) or "none"

    return (
        "Assess this predictive-maintenance case. "
        f"Type={row['Type']}; "
        f"Air temperature={row['Air temperature [K]']:.2f} K; "
        f"Process temperature={row['Process temperature [K]']:.2f} K; "
        f"Rotational speed={row['Rotational speed [rpm]']:.2f} rpm; "
        f"Torque={row['Torque [Nm]']:.2f} Nm; "
        f"Tool wear={row['Tool wear [min]']:.2f} min. "
        f"The ML model estimates failure probability {probability:.4f} "
        f"using threshold {THRESHOLD:.2f}. "
        f"Top model factors: {factor_text}. "
        f"Condition screens triggered: {flag_text}. "
        "What maintenance monitoring or inspection actions are supported "
        "by the retrieved technical evidence?"
    )


FLAG_QUERY_TERMS = {
    "high_torque": "high torque mechanical load motor current vibration",
    "low_speed": "low rotational speed high load rotating equipment vibration",
    "high_speed": "high rotational speed rotating equipment vibration bearing",
    "high_tool_wear": "tool wear lubricant wear particle analysis inspection",
    "high_air_temp": "elevated ambient temperature thermal imaging",
    "low_heat_dissipation": "overheating heat dissipation thermal imaging motor",
    "power_out_of_band": "motor current analysis electrical load",
    "overstrain": "overload wear torque lubricant inspection",
}

FEATURE_QUERY_TERMS = {
    "Air temperature [K]": "temperature thermal monitoring",
    "Process temperature [K]": "process temperature overheating thermal",
    "Rotational speed [rpm]": "rotational speed vibration rotating equipment",
    "Torque [Nm]": "torque mechanical load motor current",
    "Tool wear [min]": "wear lubricant wear particle analysis",
    "Type": "",
}


def build_retrieval_query(flags, factors):
    parts = ["predictive maintenance condition monitoring baseline"]

    for name in active_flags(flags):
        parts.append(FLAG_QUERY_TERMS[name])

    for feature in toward_failure_features(factors):
        parts.append(FEATURE_QUERY_TERMS.get(feature, ""))

    return " ".join(p for p in parts if p)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def retrieve_passages(query, vectorizer, matrix, passages,
                      top_k=TOP_K_RETRIEVAL):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, matrix)[0]
    ranked = np.argsort(similarities)[::-1][:RETRIEVAL_POOL]

    results = []
    seen_ids = set()
    per_source = {}

    for index in ranked:
        score = float(similarities[index])
        if score < MIN_SCORE:
            break

        passage = passages[index]
        text = passage.get("text", "").strip()
        if not text:
            continue

        passage_id = passage.get("passage_id", f"passage_{index}")
        if passage_id in seen_ids:
            continue

        # Skip passages with no maintenance-action content
        distinct_terms = {
            m.group(0).lower() for m in ACTION_TERM_PATTERN.finditer(text)
        }
        if len(distinct_terms) < MIN_ACTION_TERMS:
            continue

        source_id = passage.get("source_id", "")
        if per_source.get(source_id, 0) >= MAX_PER_SOURCE:
            continue

        results.append(
            {
                "passage_id": passage_id,
                "source_id": source_id,
                "title": passage.get("title", ""),
                "publisher": passage.get("publisher", ""),
                "url": passage.get("url", ""),
                "text": text,
                "score": score,
            }
        )
        seen_ids.add(passage_id)
        per_source[source_id] = per_source.get(source_id, 0) + 1

        if len(results) >= top_k:
            break

    return results


def format_retrieved_evidence(retrieved):
    return "\n\n".join(
        f"[{r['passage_id']}] {r['title']} ({r['publisher']}) "
        f"score={r['score']:.3f}\n{r['text']}"
        for r in retrieved
    )


def get_source_string(retrieved):
    sources, seen = [], set()
    for r in retrieved:
        if r["source_id"] in seen:
            continue
        seen.add(r["source_id"])
        sources.append(f"{r['title']} | {r['publisher']} | {r['url']}")
    return " || ".join(sources)


def get_passage_id_string(retrieved):
    return " || ".join(r["passage_id"] for r in retrieved)


def get_citation_string(retrieved):
    return " || ".join(
        f"[{r['passage_id']}] {r['title']} | {r['publisher']} | {r['url']}"
        for r in retrieved
    )


# --------------------------------------------------------------------------
# Claim-level answer generation
# --------------------------------------------------------------------------

# Each rule: triggered by case flags or by the model's top drivers; supported
# only if a retrieved passage matches `pattern` (whole-word regex).
RULES = [
    {
        "id": "thermal",
        "flags": ["high_air_temp", "low_heat_dissipation"],
        "features": ["Air temperature [K]", "Process temperature [K]"],
        "pattern": r"\bthermal imag\w*|\btemperature\b|\boverheat\w*",
        "text": (
            "Check operating temperatures (for example by thermal imaging) "
            "and compare them with earlier readings of the same unit, "
            "manufacturer standards, or like equipment."
        ),
    },
    {
        "id": "vibration",
        "flags": ["high_torque", "low_speed", "high_speed",
                  "power_out_of_band", "overstrain"],
        "features": ["Rotational speed [rpm]", "Torque [Nm]"],
        "pattern": r"\bvibration\b",
        "text": (
            "Use vibration monitoring on the rotating equipment and trend "
            "the readings over time."
        ),
    },
    {
        "id": "wear",
        "flags": ["high_tool_wear", "overstrain"],
        "features": ["Tool wear [min]"],
        "pattern": r"\bwear particle\b|\blubricant\w*|\bwear\b",
        "text": (
            "Inspect wear condition and, where the equipment is lubricated, "
            "consider lubricant and wear-particle analysis."
        ),
    },
    {
        "id": "electrical_load",
        "flags": ["high_torque", "overstrain", "power_out_of_band"],
        "features": ["Torque [Nm]"],
        "pattern": r"\bmotor current\b|\bcurrent analysis\b",
        "text": (
            "Review motor electrical load using motor current analysis."
        ),
    },
    {
        "id": "baseline",
        "always": True,
        "flags": [],
        "features": [],
        "pattern": (
            r"\bbaselines?\b|\bprevious images\b|\bmanufacturer'?s standards\b"
        ),
        "text": (
            "Compare monitored values against asset baselines to decide "
            "when maintenance is needed."
        ),
    },
]


def rule_triggered(rule, flags, driver_features):
    if rule.get("always"):
        return True
    if any(flags.get(f, False) for f in rule["flags"]):
        return True
    return any(f in driver_features for f in rule["features"])


def best_supporting_passage(pattern, retrieved):
    regex = re.compile(pattern, re.IGNORECASE)
    best, best_hits = None, 0
    for r in retrieved:
        hits = len(regex.findall(r["text"]))
        if hits > best_hits:
            best, best_hits = r, hits
    return best


def build_claims(flags, factors, retrieved):
    driver_features = toward_failure_features(factors)
    supported, unsupported = [], []

    for rule in RULES:
        if not rule_triggered(rule, flags, driver_features):
            continue
        passage = best_supporting_passage(rule["pattern"], retrieved)
        if passage is None:
            unsupported.append(rule["id"])
        else:
            supported.append(
                {
                    "rule": rule["id"],
                    "text": rule["text"],
                    "passage_id": passage["passage_id"],
                }
            )
    return supported, unsupported


def risk_band(probability):
    if probability >= THRESHOLD:
        return "HIGH (at/above threshold)"
    if probability >= WATCH_LOW:
        return "WATCH (below threshold, not negligible)"
    return "LOW"


def generate_answer(probability, prediction, factors, flags, derived,
                    supported, unsupported):
    if prediction == 1:
        text = (
            f"The ML model estimates a failure probability of "
            f"{probability:.3f}, above the {THRESHOLD:.2f} decision "
            "threshold. This is a model prediction, not a confirmed "
            "physical diagnosis."
        )
    elif probability >= WATCH_LOW:
        text = (
            f"The ML model estimates a failure probability of "
            f"{probability:.3f}, below the {THRESHOLD:.2f} threshold but "
            "high enough to keep the unit on a watch list. This does not "
            "prove the machine is healthy."
        )
    else:
        text = (
            f"The ML model estimates a failure probability of "
            f"{probability:.3f}, well below the {THRESHOLD:.2f} threshold. "
            "This does not prove the machine is healthy."
        )

    toward = [f for f in factors[:N_SHAP_FACTORS] if f["shap"] > 1e-12]
    if toward:
        drivers = "; ".join(
            f"{f['feature']} (SHAP {f['shap']:+.2f} log-odds)"
            for f in toward
        )
        text += f" Main drivers pushing toward failure: {drivers}."
    else:
        text += " No feature pushes the prediction toward failure."

    on = active_flags(flags)
    if on:
        text += (
            f" Condition screens triggered: {', '.join(on)} "
            f"(delta T = {derived['delta_t_K']} K, "
            f"power = {derived['power_W']} W)."
        )

    if supported:
        text += " Actions supported by the retrieved evidence: "
        text += " ".join(
            f"({i}) {c['text']} [{c['passage_id']}]"
            for i, c in enumerate(supported, start=1)
        )
    else:
        text += (
            " The retrieved evidence does not support a specific "
            "maintenance action for this case."
        )

    if unsupported:
        text += (
            " No retrieved passage supports these case-triggered topics: "
            f"{', '.join(unsupported)}."
        )

    text += (
        " These are monitoring and inspection suggestions drawn from "
        "general sources, not a confirmed diagnosis."
    )
    return text


def assess_support(supported, unsupported):
    """
    Lexical, claim-level check: of the claims the case triggered, how many
    found a matching passage. This is NOT entailment; confirm with an LLM
    judge or a human reviewer.
    """
    total = len(supported) + len(unsupported)
    if total == 0 or not supported:
        return "NOT_SUPPORTED", len(supported), total
    if unsupported:
        return "PARTIALLY_SUPPORTED", len(supported), total
    return "SUPPORTED", len(supported), total


# --------------------------------------------------------------------------
# Record creation
# --------------------------------------------------------------------------

def create_case_record(case_id, original_index, row, probability, prediction,
                       factors, flags, derived, question, retrieval_query,
                       retrieved, answer, supported, unsupported, support,
                       n_supported, n_total):
    true_label = int(row[TARGET])

    record = {
        "Case ID": case_id,
        "Original Test Row": original_index,
        "Type": row["Type"],
        "Air temperature [K]": row["Air temperature [K]"],
        "Process temperature [K]": row["Process temperature [K]"],
        "Rotational speed [rpm]": row["Rotational speed [rpm]"],
        "Torque [Nm]": row["Torque [Nm]"],
        "Tool wear [min]": row["Tool wear [min]"],
        "True Label": true_label,
        "Failure Probability": probability,
        "Decision Threshold": THRESHOLD,
        "Thresholded Prediction": prediction,
        "Outcome": (
            "TP" if (true_label, prediction) == (1, 1) else
            "TN" if (true_label, prediction) == (0, 0) else
            "FP" if (true_label, prediction) == (0, 1) else "FN"
        ),
        "Risk Band": risk_band(probability),
        "Condition Flags": ", ".join(active_flags(flags)),
        "Derived Values": json.dumps(derived),
        "Question": question,
        "RAG Query": retrieval_query,
        "Retrieved Sources": get_source_string(retrieved),
        "Retrieved Passage IDs": get_passage_id_string(retrieved),
        "Retrieved Evidence": format_retrieved_evidence(retrieved),
        "Answer": answer,
        "Supported Claims": json.dumps(supported),
        "Unsupported Topics": ", ".join(unsupported),
        "Claims Supported": n_supported,
        "Claims Triggered": n_total,
        "Citations": get_citation_string(retrieved),
        "Supported by Retrieved Text": support,
    }

    for i, factor in enumerate(factors, start=1):
        record[f"Top Factor {i}"] = factor["feature"]
        record[f"Top Factor {i} SHAP"] = factor["shap"]
        record[f"Top Factor {i} Direction"] = factor["direction"]

    return record


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("END-TO-END RAG EVALUATION")
    print("=" * 70)

    print("\nLoading model...")
    model = load_model()

    print("Loading test data...")
    test_df = load_test_data()
    ref = reference_stats(test_df)

    print("Loading retrieval index...")
    vectorizer, matrix, passages = load_retrieval_index()
    print(f"Index size: {len(passages)} passages")

    failed = test_df[test_df[TARGET] == 1].sample(
        n=N_FAILED, random_state=RANDOM_STATE
    )
    non_failed = test_df[test_df[TARGET] == 0].sample(
        n=N_NON_FAILED, random_state=RANDOM_STATE
    )
    selected = pd.concat([failed, non_failed]).sample(
        frac=1, random_state=RANDOM_STATE
    )

    print(f"\nCases evaluated : {len(selected)}")
    print(f"Failed cases    : {len(failed)}")
    print(f"Non-failed cases: {len(non_failed)}")

    # Batch computations (done once)
    X = selected[FEATURES]
    probabilities = model.predict_proba(X)[:, 1]
    shap_matrix = compute_shap_matrix(model, X)

    print("\nRunning cases...")
    records = []

    for position, (original_index, row) in enumerate(
        selected.iterrows(), start=0
    ):
        case_id = f"CASE_{position + 1:02d}"
        probability = float(probabilities[position])
        prediction = int(probability >= THRESHOLD)

        factors = rank_factors(shap_matrix[position])
        flags, derived = derive_conditions(row, ref)

        question = build_question(row, probability, factors, flags)
        retrieval_query = build_retrieval_query(flags, factors)

        retrieved = retrieve_passages(
            retrieval_query, vectorizer, matrix, passages
        )

        supported, unsupported = build_claims(flags, factors, retrieved)
        answer = generate_answer(
            probability, prediction, factors, flags, derived,
            supported, unsupported
        )
        support, n_supported, n_total = assess_support(
            supported, unsupported
        )

        record = create_case_record(
            case_id, original_index, row, probability, prediction,
            factors, flags, derived, question, retrieval_query, retrieved,
            answer, supported, unsupported, support, n_supported, n_total
        )
        records.append(record)

        print(
            f"{case_id}: True={int(row[TARGET])}, "
            f"P={probability:.4f}, Pred={prediction}, "
            f"Flags=[{', '.join(active_flags(flags))}], "
            f"Claims={n_supported}/{n_total}, Support={support}"
        )

    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Output file: {OUTPUT_FILE}")

    print(f"\nOutcome counts (threshold {THRESHOLD:.2f}):")
    print(results_df["Outcome"].value_counts().to_string())

    fn = results_df[results_df["Outcome"] == "FN"]
    if len(fn):
        print("\nMissed failures (false negatives):")
        print(
            fn[["Case ID", "Original Test Row", "Failure Probability",
                "Risk Band", "Top Factor 1", "Top Factor 1 SHAP"]]
            .to_string(index=False)
        )

    print("\nEvidence support (claim-level, lexical):")
    print(results_df["Supported by Retrieved Text"]
          .value_counts().to_string())

    unique_sets = results_df["Retrieved Passage IDs"].nunique()
    print(f"\nDistinct retrieved passage sets: {unique_sets} "
          f"of {len(results_df)} cases")

    print("\nFirst case SHAP values (log-odds):")
    first = results_df.iloc[0]
    for i in range(1, len(FEATURES) + 1):
        print(
            f"{i}. {first[f'Top Factor {i}']}: "
            f"{first[f'Top Factor {i} SHAP']:+.4f} "
            f"({first[f'Top Factor {i} Direction']})"
        )


if __name__ == "__main__":
    main()
