import os
import json
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PASSAGES_FILE = "data/knowledge_base/passages.json"
INDEX_DIR = "data/knowledge_base/index"

TOP_K = 5


def load_passages():
    if not os.path.exists(PASSAGES_FILE):
        raise FileNotFoundError(
            f"Passages file not found: {PASSAGES_FILE}\n"
            "Run build_kb.py first."
        )

    with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
        passages = json.load(f)

    if not passages:
        raise ValueError("No passages found in passages.json.")

    return passages

def build_index(passages):

    texts = [p["text"] for p in passages]

    print("\nBuilding TF-IDF index...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    os.makedirs(INDEX_DIR, exist_ok=True)

    joblib.dump(
        vectorizer,
        os.path.join(INDEX_DIR, "tfidf_vectorizer.joblib")
    )

    joblib.dump(
        tfidf_matrix,
        os.path.join(INDEX_DIR, "tfidf_matrix.joblib")
    )

    joblib.dump(
        passages,
        os.path.join(INDEX_DIR, "passages.joblib")
    )

    print("TF-IDF index created successfully.")
    print(f"Documents indexed : {len(passages)}")
    print(f"Vocabulary size    : {len(vectorizer.vocabulary_)}")

    return vectorizer, tfidf_matrix

def load_index():

    vectorizer_path = os.path.join(
        INDEX_DIR,
        "tfidf_vectorizer.joblib"
    )

    matrix_path = os.path.join(
        INDEX_DIR,
        "tfidf_matrix.joblib"
    )

    passages_path = os.path.join(
        INDEX_DIR,
        "passages.joblib"
    )

    if not (
        os.path.exists(vectorizer_path)
        and os.path.exists(matrix_path)
        and os.path.exists(passages_path)
    ):
        return None

    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = joblib.load(matrix_path)
    passages = joblib.load(passages_path)

    return vectorizer, tfidf_matrix, passages

def retrieve(query, vectorizer, tfidf_matrix, passages, top_k=TOP_K):

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        passage = passages[index].copy()

        passage["similarity"] = float(similarities[index])

        results.append(passage)

    return results

def display_results(query, results):

    print("\n")
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(f"\nQuery:")
    print(query)

    print("\n" + "-" * 70)

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 70)

        print(f"Similarity : {result['similarity']:.4f}")
        print(f"Source ID  : {result['source_id']}")
        print(f"Title      : {result['title']}")
        print(f"Publisher  : {result['publisher']}")
        print(f"Passage ID : {result['passage_id']}")

        print("\nText:")
        print(result["text"])

        print(f"\nURL:")
        print(result["url"])

    print("\n" + "=" * 70)

def main():

    print("=" * 70)
    print("KNOWLEDGE BASE RETRIEVER")
    print("=" * 70)

    existing_index = load_index()

    if existing_index is None:

        print("\nNo existing index found.")

        passages = load_passages()

        print(f"Passages loaded: {len(passages)}")

        vectorizer, tfidf_matrix = build_index(passages)

    else:

        print("\nExisting TF-IDF index found.")

        vectorizer, tfidf_matrix, passages = existing_index

        print(f"Passages loaded: {len(passages)}")


    query = (
        "What should be monitored when a machine "
        "has high temperature and abnormal torque?"
    )

    results = retrieve(
        query=query,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        passages=passages,
        top_k=TOP_K
    )

    display_results(query, results)


if __name__ == "__main__":
    main()
