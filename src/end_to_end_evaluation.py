"""
End-to-end evaluation (Req 8).

All heavy-lifting — SHAP computation, feature mapping, retrieval, claim-level
answer generation, and support assessment — is imported from explainability.py,
which contains the single, canonical implementation for this project.

Fixed issues resolved here:
  * SHAP values were all-zero because the old per-row mapping used hard-coded
    prefixes ("cat__" / "num__") that did not match the ColumnTransformer's
    actual output names ("categorical__" / "numerical__").  The batch
    compute_shap_matrix() in explainability.py uses map_feature() with
    TRANSFORMER_PREFIXES that covers both naming conventions.
  * RAG answers were static boilerplate because the retrieval query did not
    use real SHAP drivers (which were zeroed out) and the answer was generated
    by simple keyword presence rather than claim-level matching.  The
    claim-level pipeline in explainability.py checks each supported action
    against a matched passage before including it in the answer.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# All logic lives in explainability.py — import everything from there.
# ---------------------------------------------------------------------------
from explainability import (
    FEATURES,
    TARGET,
    THRESHOLD,
    N_FAILED,
    N_NON_FAILED,
    RANDOM_STATE,
    load_model,
    load_test_data,
    load_retrieval_index,
    reference_stats,
    compute_shap_matrix,
    rank_factors,
    derive_conditions,
    active_flags,
    build_question,
    build_retrieval_query,
    retrieve_passages,
    build_claims,
    generate_answer,
    assess_support,
    create_case_record,
)


OUTPUT_FILE = "results/end_to_end_evaluation.csv"


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("END-TO-END RAG EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load artefacts
    # -----------------------------------------------------------------------
    print("\nLoading model...")
    model = load_model()

    print("Loading test data...")
    test_df = load_test_data()
    ref = reference_stats(test_df)

    print("Loading retrieval index...")
    vectorizer, matrix, passages = load_retrieval_index()
    print(f"Index size: {len(passages)} passages")

    # -----------------------------------------------------------------------
    # Select 10 failed + 10 non-failed cases (Req 8 requires ≥ 20 cases
    # including both categories)
    # -----------------------------------------------------------------------
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
    print(f"  Failed         : {len(failed)}")
    print(f"  Non-failed     : {len(non_failed)}")

    # -----------------------------------------------------------------------
    # Batch predictions and SHAP (single pass — correct feature mapping via
    # map_feature() in explainability.py)
    # -----------------------------------------------------------------------
    X = selected[FEATURES]
    probabilities = model.predict_proba(X)[:, 1]
    shap_matrix = compute_shap_matrix(model, X)

    # -----------------------------------------------------------------------
    # Per-case pipeline
    # -----------------------------------------------------------------------
    print("\nRunning cases...")
    records = []

    for position, (original_index, row) in enumerate(
        selected.iterrows(), start=0
    ):
        case_id = f"CASE_{position + 1:02d}"
        probability = float(probabilities[position])
        prediction = int(probability >= THRESHOLD)

        # SHAP factors — aggregated back to original features (log-odds)
        factors = rank_factors(shap_matrix[position])

        # Condition flags derived from sensor values (mirrors AI4I failure
        # mode definitions — heat dissipation, power band, overstrain, wear)
        flags, derived = derive_conditions(row, ref)

        # RAG query driven by real SHAP drivers + active condition flags
        question = build_question(row, probability, factors, flags)
        retrieval_query = build_retrieval_query(flags, factors)

        # Retrieval with action-term filtering and per-source diversity limit
        retrieved = retrieve_passages(
            retrieval_query, vectorizer, matrix, passages
        )

        # Claim-level answer: each action is only included when a retrieved
        # passage lexically supports it
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

    # -----------------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_FILE, index=False)

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Output file : {OUTPUT_FILE}")
    print(f"Total cases : {len(results_df)}")

    print(f"\nOutcome counts (threshold = {THRESHOLD:.2f}):")
    print(results_df["Outcome"].value_counts().to_string())

    fn_rows = results_df[results_df["Outcome"] == "FN"]
    if len(fn_rows):
        print("\nMissed failures (false negatives):")
        print(
            fn_rows[
                [
                    "Case ID",
                    "Original Test Row",
                    "Failure Probability",
                    "Risk Band",
                    "Top Factor 1",
                    "Top Factor 1 SHAP",
                ]
            ].to_string(index=False)
        )

    print("\nEvidence support (claim-level, lexical):")
    print(
        results_df["Supported by Retrieved Text"].value_counts().to_string()
    )

    unique_sets = results_df["Retrieved Passage IDs"].nunique()
    print(
        f"\nDistinct retrieved passage sets: {unique_sets} "
        f"of {len(results_df)} cases"
    )

    print("\nFirst case SHAP values (log-odds):")
    first = results_df.iloc[0]
    for i in range(1, len(FEATURES) + 1):
        print(
            f"  {i}. {first[f'Top Factor {i}']}: "
            f"{first[f'Top Factor {i} SHAP']:+.4f} "
            f"({first[f'Top Factor {i} Direction']})"
        )


if __name__ == "__main__":
    main()
