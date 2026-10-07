import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.metrics.pairwise import cosine_similarity


FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

TEST_FILE = "data/processed/test.csv"

MODEL_FILE = "models/xgboost.joblib"

VECTORIZER_FILE = (
    "data/knowledge_base/index/tfidf_vectorizer.joblib"
)

MATRIX_FILE = (
    "data/knowledge_base/index/tfidf_matrix.joblib"
)

PASSAGES_FILE = (
    "data/knowledge_base/index/passages.joblib"
)

OUTPUT_FILE = "results/end_to_end_evaluation.csv"

THRESHOLD = 0.50

N_FAILED = 10
N_NON_FAILED = 10

TOP_K_RETRIEVAL = 5

RANDOM_STATE = 42


def load_model():
    return joblib.load(MODEL_FILE)


def load_test_data():
    return pd.read_csv(TEST_FILE)


def load_retrieval_index():

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    matrix = joblib.load(
        MATRIX_FILE
    )

    passages = joblib.load(
        PASSAGES_FILE
    )

    return vectorizer, matrix, passages


def get_failure_probability(model, row):

    X = pd.DataFrame(
        [
            {
                feature: row[feature]
                for feature in FEATURES
            }
        ]
    )

    probability = model.predict_proba(X)[0, 1]

    return float(probability)


def get_shap_factors(model, row):

    X = pd.DataFrame(
        [
            {
                feature: row[feature]
                for feature in FEATURES
            }
        ]
    )

    preprocessor = model.named_steps["preprocessor"]

    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_for_shap = X_transformed.toarray()
    else:
        X_for_shap = np.asarray(
            X_transformed
        )

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = explainer.shap_values(
        X_for_shap
    )

    if isinstance(shap_values, list):

        if len(shap_values) == 1:
            shap_values = shap_values[0]

        else:
            shap_values = shap_values[-1]

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:

        if shap_values.shape[0] == 1:
            shap_values = shap_values[0]

        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, 1]

    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    shap_values = shap_values.flatten()

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    if len(feature_names) != len(shap_values):

        raise RuntimeError(
            "SHAP feature count does not match "
            "transformed feature count."
        )

    aggregated = {
        feature: 0.0
        for feature in FEATURES
    }

    for feature_name, shap_value in zip(
        feature_names,
        shap_values
    ):

        shap_value = float(
            shap_value
        )

        if feature_name.startswith(
            "cat__Type_"
        ):

            original_feature = "Type"

        elif feature_name.startswith(
            "num__Air temperature [K]"
        ):

            original_feature = (
                "Air temperature [K]"
            )

        elif feature_name.startswith(
            "num__Process temperature [K]"
        ):

            original_feature = (
                "Process temperature [K]"
            )

        elif feature_name.startswith(
            "num__Rotational speed [rpm]"
        ):

            original_feature = (
                "Rotational speed [rpm]"
            )

        elif feature_name.startswith(
            "num__Torque [Nm]"
        ):

            original_feature = (
                "Torque [Nm]"
            )

        elif feature_name.startswith(
            "num__Tool wear [min]"
        ):

            original_feature = (
                "Tool wear [min]"
            )

        else:
            continue

        aggregated[
            original_feature
        ] += shap_value

    factors = []

    for feature in FEATURES:

        value = float(
            aggregated[feature]
        )

        if value > 1e-12:

            direction = "Toward failure"

        elif value < -1e-12:

            direction = "Away from failure"

        else:

            direction = "Neutral"

        factors.append(
            {
                "feature": feature,
                "shap": value,
                "direction": direction
            }
        )

    factors.sort(
        key=lambda x: abs(
            x["shap"]
        ),
        reverse=True
    )

    return factors


def build_question(
    row,
    probability,
    factors
):

    top_factors = factors[:3]

    factor_text = ", ".join(
        [
            (
                f"{factor['feature']} "
                f"({factor['direction']}, "
                f"SHAP={factor['shap']:.4f})"
            )
            for factor in top_factors
        ]
    )

    question = (
        "Assess this predictive-maintenance case. "
        f"Type={row['Type']}; "
        f"Air temperature={row['Air temperature [K]']:.2f} K; "
        f"Process temperature={row['Process temperature [K]']:.2f} K; "
        f"Rotational speed={row['Rotational speed [rpm]']:.2f} rpm; "
        f"Torque={row['Torque [Nm]']:.2f} Nm; "
        f"Tool wear={row['Tool wear [min]']:.2f} min. "
        f"The ML model estimates failure probability "
        f"{probability:.4f} using threshold "
        f"{THRESHOLD:.2f}. "
        f"Top model factors are: {factor_text}. "
        "What maintenance monitoring or inspection actions "
        "are supported by the retrieved technical evidence?"
    )

    return question


def build_retrieval_query(
    row,
    probability,
    factors
):

    query_parts = []

    query_parts.append(
        "predictive maintenance condition monitoring"
    )

    query_parts.append(
        "machine failure maintenance decision"
    )

    if (
        row["Air temperature [K]"]
        >= row["Process temperature [K]"]
    ):
        query_parts.append(
            "temperature thermal monitoring overheating"
        )

    query_parts.append(
        "rotating equipment vibration monitoring"
    )

    query_parts.append(
        "torque mechanical load monitoring"
    )

    query_parts.append(
        "tool wear condition monitoring"
    )

    top_factors = factors[:3]

    for factor in top_factors:

        feature = factor["feature"]

        if feature == "Air temperature [K]":

            query_parts.append(
                "air temperature thermal monitoring"
            )

        elif feature == "Process temperature [K]":

            query_parts.append(
                "process temperature thermal monitoring"
            )

        elif feature == "Rotational speed [rpm]":

            query_parts.append(
                "rotational speed vibration rotating equipment"
            )

        elif feature == "Torque [Nm]":

            query_parts.append(
                "torque mechanical load rotating equipment"
            )

        elif feature == "Tool wear [min]":

            query_parts.append(
                "tool wear condition monitoring maintenance"
            )

    return " ".join(
        query_parts
    )


def retrieve_passages(
    query,
    vectorizer,
    matrix,
    passages,
    top_k=5
):

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        matrix
    )[0]

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    results = []

    seen_passages = set()

    for index in ranked_indices:

        passage = passages[index]

        passage_id = passage.get(
            "passage_id",
            f"passage_{index}"
        )

        if passage_id in seen_passages:
            continue

        text = passage.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        score = float(
            similarities[index]
        )

        results.append(
            {
                "passage_id": passage_id,
                "source_id": passage.get(
                    "source_id",
                    ""
                ),
                "title": passage.get(
                    "title",
                    ""
                ),
                "publisher": passage.get(
                    "publisher",
                    ""
                ),
                "url": passage.get(
                    "url",
                    ""
                ),
                "text": text,
                "score": score
            }
        )

        seen_passages.add(
            passage_id
        )

        if len(results) >= top_k:
            break

    return results


def format_retrieved_evidence(
    retrieved
):

    evidence = []

    for item in retrieved:

        evidence.append(
            f"[{item['passage_id']}] "
            f"{item['title']} "
            f"({item['publisher']})\n"
            f"{item['text']}"
        )

    return "\n\n".join(
        evidence
    )


def generate_answer(
    row,
    probability,
    threshold,
    prediction,
    factors,
    retrieved
):

    if prediction == 1:

        prediction_text = (
            f"The ML model estimates a failure probability "
            f"of {probability:.3f}, which is above the "
            f"{threshold:.2f} decision threshold. "
            "This is a model prediction and is not by itself "
            "a confirmed physical failure diagnosis."
        )

    else:

        prediction_text = (
            f"The ML model estimates a failure probability "
            f"of {probability:.3f}, which is below the "
            f"{threshold:.2f} decision threshold. "
            "This is a model prediction and does not prove "
            "that the machine is physically healthy."
        )

    recommendations = []

    evidence_text = " ".join(
        [
            item["text"].lower()
            for item in retrieved
        ]
    )

    if (
        "vibration" in evidence_text
        or "condition monitoring" in evidence_text
    ):

        recommendations.append(
            "Use vibration or condition monitoring "
            "and examine trends over time."
        )

    if (
        "thermal" in evidence_text
        or "temperature" in evidence_text
        or "overheating" in evidence_text
    ):

        recommendations.append(
            "Monitor thermal behavior and compare "
            "temperatures with appropriate baselines, "
            "previous measurements, or similar equipment."
        )

    if (
        "lubric" in evidence_text
        or "oil" in evidence_text
    ):

        recommendations.append(
            "Include lubrication or oil-condition checks "
            "where relevant to the equipment."
        )

    if (
        "tool wear" in evidence_text
        or "wear" in evidence_text
    ):

        recommendations.append(
            "Consider wear-related inspection and "
            "condition monitoring where applicable."
        )

    if not recommendations:

        answer = (
            prediction_text
            + " The retrieved technical evidence does not "
            "provide enough specific information to support "
            "a particular maintenance recommendation."
        )

    else:

        answer = (
            prediction_text
            + " Based on the retrieved technical evidence, "
            "appropriate actions include: "
            + " ".join(recommendations)
            + " These recommendations are maintenance "
            "monitoring actions rather than a confirmed "
            "physical diagnosis."
        )

    return answer


def assess_support(
    answer,
    retrieved
):

    if not retrieved:

        return "NOT_SUPPORTED"

    evidence_text = " ".join(
        [
            item["text"].lower()
            for item in retrieved
        ]
    )

    answer_lower = answer.lower()

    supported_terms = [
        "vibration",
        "thermal",
        "temperature",
        "lubrication",
        "oil",
        "wear",
        "condition monitoring",
        "trend",
        "baseline"
    ]

    matched = 0

    for term in supported_terms:

        if term in answer_lower and term in evidence_text:

            matched += 1

    if matched >= 2:

        return "SUPPORTED"

    if matched == 1:

        return "PARTIALLY_SUPPORTED"

    return "NOT_SUPPORTED"


def get_source_string(
    retrieved
):

    sources = []

    seen = set()

    for item in retrieved:

        source_id = item["source_id"]

        if source_id in seen:
            continue

        source_text = (
            f"{item['title']} | "
            f"{item['publisher']} | "
            f"{item['url']}"
        )

        sources.append(
            source_text
        )

        seen.add(
            source_id
        )

    return " || ".join(
        sources
    )


def get_passage_id_string(
    retrieved
):

    return " || ".join(
        [
            item["passage_id"]
            for item in retrieved
        ]
    )


def get_citation_string(
    retrieved
):

    citations = []

    for item in retrieved:

        citations.append(
            f"[{item['passage_id']}] "
            f"{item['title']} | "
            f"{item['publisher']} | "
            f"{item['url']}"
        )

    return " || ".join(
        citations
    )


def create_case_record(
    case_id,
    original_index,
    row,
    probability,
    prediction,
    factors,
    question,
    retrieval_query,
    retrieved,
    answer,
    support
):

    record = {
        "Case ID": case_id,
        "Original Test Row": original_index,
        "Type": row["Type"],
        "Air temperature [K]": row[
            "Air temperature [K]"
        ],
        "Process temperature [K]": row[
            "Process temperature [K]"
        ],
        "Rotational speed [rpm]": row[
            "Rotational speed [rpm]"
        ],
        "Torque [Nm]": row[
            "Torque [Nm]"
        ],
        "Tool wear [min]": row[
            "Tool wear [min]"
        ],
        "True Label": int(
            row[TARGET]
        ),
        "Failure Probability": probability,
        "Decision Threshold": THRESHOLD,
        "Thresholded Prediction": prediction,
        "Question": question,
        "RAG Query": retrieval_query,
        "Retrieved Sources": get_source_string(
            retrieved
        ),
        "Retrieved Passage IDs": get_passage_id_string(
            retrieved
        ),
        "Retrieved Evidence": format_retrieved_evidence(
            retrieved
        ),
        "Answer": answer,
        "Citations": get_citation_string(
            retrieved
        ),
        "Supported by Retrieved Text": support
    }

    for i, factor in enumerate(
        factors,
        start=1
    ):

        record[
            f"Top Factor {i}"
        ] = factor["feature"]

        record[
            f"Top Factor {i} SHAP"
        ] = factor["shap"]

        record[
            f"Top Factor {i} Direction"
        ] = factor["direction"]

    return record


def main():

    os.makedirs(
        "results",
        exist_ok=True
    )

    print("=" * 70)
    print("END-TO-END RAG EVALUATION")
    print("=" * 70)

    print()
    print("Loading model...")

    model = load_model()

    print("Loading test data...")

    test_df = load_test_data()

    print("Loading retrieval index...")

    (
        vectorizer,
        matrix,
        passages
    ) = load_retrieval_index()

    failed = test_df[
        test_df[TARGET] == 1
    ].sample(
        n=N_FAILED,
        random_state=RANDOM_STATE
    )

    non_failed = test_df[
        test_df[TARGET] == 0
    ].sample(
        n=N_NON_FAILED,
        random_state=RANDOM_STATE
    )

    selected = pd.concat(
        [
            failed,
            non_failed
        ]
    ).sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    print()
    print(
        f"Cases evaluated : {len(selected)}"
    )

    print(
        f"Failed cases    : "
        f"{len(failed)}"
    )

    print(
        f"Non-failed cases: "
        f"{len(non_failed)}"
    )

    print()
    print("Running cases...")

    records = []

    for case_number, (
        original_index,
        row
    ) in enumerate(
        selected.iterrows(),
        start=1
    ):

        case_id = (
            f"CASE_{case_number:02d}"
        )

        probability = get_failure_probability(
            model,
            row
        )

        prediction = int(
            probability >= THRESHOLD
        )

        factors = get_shap_factors(
            model,
            row
        )

        question = build_question(
            row,
            probability,
            factors
        )

        retrieval_query = build_retrieval_query(
            row,
            probability,
            factors
        )

        retrieved = retrieve_passages(
            retrieval_query,
            vectorizer,
            matrix,
            passages,
            TOP_K_RETRIEVAL
        )

        answer = generate_answer(
            row,
            probability,
            THRESHOLD,
            prediction,
            factors,
            retrieved
        )

        support = assess_support(
            answer,
            retrieved
        )

        record = create_case_record(
            case_id,
            original_index,
            row,
            probability,
            prediction,
            factors,
            question,
            retrieval_query,
            retrieved,
            answer,
            support
        )

        records.append(
            record
        )

        print(
            f"{case_id}: "
            f"True={int(row[TARGET])}, "
            f"Probability={probability:.4f}, "
            f"Prediction={prediction}, "
            f"Support={support}"
        )

    results_df = pd.DataFrame(
        records
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print()
    print("Evidence support:")

    print(
        results_df[
            "Supported by Retrieved Text"
        ].value_counts()
    )

    print()
    print("First case SHAP values:")

    first = results_df.iloc[0]

    for i in range(1, 7):

        feature = first[
            f"Top Factor {i}"
        ]

        value = first[
            f"Top Factor {i} SHAP"
        ]

        direction = first[
            f"Top Factor {i} Direction"
        ]

        print(
            f"{i}. {feature}: "
            f"{value:.6f} "
            f"({direction})"
        )


if __name__ == "__main__":
    main()
