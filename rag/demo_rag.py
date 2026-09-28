"""Demonstration script executing Hybrid RAG + Cross-Encoder Reranking on Milestone 7 test queries."""

import json
from rag.pipeline import answer_query


def run_demo():
    demos = [
        # Part A: Lexical / Exact Match Advantages
        ("DEMO 1 (Exact Dish Name)", "Street Style Bun Kabab"),
        ("DEMO 2 (Specific Ingredient)", "basmati rice"),
        ("DEMO 3 (Specific Pakistani Term)", "Badami Kheer"),
        # Part B: Hard Constraint Enforcement Upstream
        ("DEMO 4 (Price Constraint Violation Prevention)", "Murgh Makhani Butter Chicken under PKR 500"),
        ("DEMO 5 (Location Constraint Violation Prevention)", "Peshawari Chicken Karahi in Saddar"),
        ("DEMO 6 (Allergen Exclusion Violation Prevention)", "Chicken Satay Skewers without peanuts"),
    ]

    for label, q in demos:
        print("\n" + "=" * 80)
        print(f"{label}: '{q}'")
        print("=" * 80)
        res = answer_query(q, top_k=3)
        print(f"Status: {res['status']}")
        print(f"Parsed Constraints:\n  {json.dumps(res['parsed_constraints'], indent=4)}")
        print(f"Filter Summary: {res['filter_summary']}")
        if res["warnings"]:
            print(f"Warnings: {res['warnings']}")

        # Show Cross-Encoder Reranked Candidate Details
        print("\nFinal Candidates (Cross-Encoder Reranked):")
        for doc in res.get("retrieved_documents", []):
            print(
                f"  - [{doc['id']}] {doc['dish_name']} | Price: PKR {doc['price_pkr']} | "
                f"Location: {doc['location']} | Reranker Score: {doc.get('reranker_score')} | "
                f"RRF Score: {doc.get('rrf_score')} (Dense Rank: {doc.get('dense_rank')}, BM25 Rank: {doc.get('bm25_rank')})"
            )

        print(f"\nGenerated Answer:\n{res['answer']}")
        print(f"\nCited Sources: {res['sources']}")
        print(f"Latency (ms): {res['latency']}")


if __name__ == "__main__":
    run_demo()
