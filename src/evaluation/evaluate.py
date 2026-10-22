import json
from pathlib import Path

from src.embeddings.embedder import Embedder
from src.retrieval.search import Search

from src.evaluation.faithfulness import FaithfulnessEvaluator


EVALUATION_PATH = Path(
    "data/processed/evaluation.json"
)


def load_evaluation_dataset():
    return json.loads(
        EVALUATION_PATH.read_text(
            encoding="utf-8"
        )
    )


def recall_at_k(
    retrieved_chunks,
    relevant_chunks,
    k,
):
    retrieved = set(
        retrieved_chunks[:k]
    )

    relevant = set(
        relevant_chunks
    )

    if not relevant:
        return 0.0

    return len(
        retrieved & relevant
    ) / len(relevant)


def precision_at_k(
    retrieved_chunks,
    relevant_chunks,
    k,
):
    retrieved = retrieved_chunks[:k]

    relevant = set(
        relevant_chunks
    )

    if not retrieved:
        return 0.0

    relevant_retrieved = sum(
        chunk_id in relevant
        for chunk_id in retrieved
    )

    return relevant_retrieved / len(
        retrieved
    )


def reciprocal_rank(
    retrieved_chunks,
    relevant_chunks,
):
    relevant = set(
        relevant_chunks
    )

    for rank, chunk_id in enumerate(
        retrieved_chunks,
        start=1,
    ):
        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def semantic_similarity(
    generated_answer,
    expected_answer,
    embedder,
):
    generated_embedding = embedder.embed_text(
        generated_answer
    )

    expected_embedding = embedder.embed_text(
        expected_answer
    )

    dot_product = sum(
        a * b
        for a, b in zip(
            generated_embedding,
            expected_embedding,
        )
    )

    generated_norm = sum(
        a * a
        for a in generated_embedding
    ) ** 0.5

    expected_norm = sum(
        b * b
        for b in expected_embedding
    ) ** 0.5

    if generated_norm == 0 or expected_norm == 0:
        return 0.0

    return dot_product / (
        generated_norm * expected_norm
    )


def evaluate_retrieval(
    dataset,
    retriever,
):
    recall_scores = []
    precision_scores = []
    reciprocal_ranks = []

    for item in dataset:

        question = item["question"]

        relevant_chunks = item[
            "relevant_chunks"
        ]

        results = retriever.search(
            question,
            limit=5,
        )

        retrieved_chunks = [
            result["chunk"]["chunk_id"]
            for result in results
        ]

        recall = recall_at_k(
            retrieved_chunks,
            relevant_chunks,
            k=5,
        )

        precision = precision_at_k(
            retrieved_chunks,
            relevant_chunks,
            k=5,
        )

        rr = reciprocal_rank(
            retrieved_chunks,
            relevant_chunks,
        )

        recall_scores.append(recall)
        precision_scores.append(precision)
        reciprocal_ranks.append(rr)

        print(
            f"\nQuestion: {question}"
        )

        print(
            f"Expected: {relevant_chunks}"
        )

        print(
            f"Retrieved: {retrieved_chunks}"
        )

        print(
            f"Recall@5: {recall:.3f}"
        )

        print(
            f"Precision@5: {precision:.3f}"
        )

        print(
            f"Reciprocal Rank: {rr:.3f}"
        )

    return {
        "recall": sum(recall_scores)
        / len(recall_scores),

        "precision": sum(precision_scores)
        / len(precision_scores),

        "mrr": sum(reciprocal_ranks)
        / len(reciprocal_ranks),
    }


def evaluate_generation(
    dataset,
    retriever,
    embedder,
    faithfulness_evaluator,
):
    from src.generation.generator import Generator

    generator = Generator()

    similarity_scores = []
    faithfulness_scores = []
    correctness_scores = []

    for item in dataset:

        question = item["question"]
        expected_answer = item["expected_answer"]

        results = retriever.search(
            question,
            limit=5,
        )

        context_parts = []

        for result in results:

            chunk = result["chunk"]

            context_parts.append(
                f"[Page {chunk['page_number']}, "
                f"Chunk {chunk['chunk_id']}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(
            context_parts
        )

        generated_answer = generator.generate(
            query=question,
            context=context,
        )

        faithfulness = (
            faithfulness_evaluator.evaluate(
                context,
                generated_answer,
            )
        )

        correctness = (
            faithfulness_evaluator.evaluate_correctness(
                question,
                expected_answer,
                generated_answer,
            )
        )

        similarity = semantic_similarity(
            generated_answer,
            expected_answer,
            embedder,
        )

        similarity_scores.append(
            similarity
        )

        if faithfulness == "SUPPORTED":
            faithfulness_scores.append(
                1.0
            )
        else:
            faithfulness_scores.append(
                0.0
            )

        if correctness == "CORRECT":
            correctness_scores.append(
                1.0
            )
        else:
            correctness_scores.append(
                0.0
            )

        print("\n" + "=" * 70)

        print(
            f"Question: {question}"
        )

        print(
            f"Expected: {expected_answer}"
        )

        print(
            f"Generated: {generated_answer}"
        )

        print(
            f"Semantic Similarity: "
            f"{similarity:.3f}"
        )

        print(
            f"Faithfulness: "
            f"{faithfulness}"
        )

        print(
            f"Correctness: "
            f"{correctness}"
        )


    average_similarity = (
        sum(similarity_scores)
        / len(similarity_scores)
    )

    average_faithfulness = (
        sum(faithfulness_scores)
        / len(faithfulness_scores)
    )

    average_correctness = (
        sum(correctness_scores)
        / len(correctness_scores)
    )

    return (
        average_similarity,
        average_faithfulness,
        average_correctness,                                    
    )


def evaluate():

    dataset = load_evaluation_dataset()

    print("=" * 70)
    print("INITIALIZING BGE DENSE RETRIEVER")
    print("=" * 70)

    embedder = Embedder()

    retriever = Search(
        embedder=embedder
    )

    print("\n")
    print("=" * 70)
    print("BGE DENSE RETRIEVAL EVALUATION")
    print("=" * 70)

    retrieval_metrics = evaluate_retrieval(
        dataset,
        retriever,
    )

    print("\n")
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"Recall@5: "
        f"{retrieval_metrics['recall']:.3f}"
    )

    print(
        f"Precision@5: "
        f"{retrieval_metrics['precision']:.3f}"
    )

    print(
        f"MRR: "
        f"{retrieval_metrics['mrr']:.3f}"
    )

    print("\n")
    print("=" * 70)
    print("INITIALIZING GENERATION EVALUATION")
    print("=" * 70)

    faithfulness_evaluator = (
        FaithfulnessEvaluator()
    )

    generation_score, faithfulness_score, correctness_score = (
        evaluate_generation(
            dataset,
            retriever,
            embedder,
            faithfulness_evaluator,
        )
    )

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Recall@5: "
        f"{retrieval_metrics['recall']:.3f}"
    )

    print(
        f"Precision@5: "
        f"{retrieval_metrics['precision']:.3f}"
    )

    print(
        f"MRR: "
        f"{retrieval_metrics['mrr']:.3f}"
    )

    print(
        f"Average Semantic Similarity: "
        f"{generation_score:.3f}"
    )

    print(
        f"Average Faithfulness: "
        f"{faithfulness_score:.3f}"
    )

    print(
        f"Average Correctness: "
        f"{correctness_score:.3f}"
    )

    retriever.close()


if __name__ == "__main__":
    evaluate()