"""Semantic search CLI utility for querying the menu vector database."""

import argparse
import logging
from typing import List, Optional
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.WARNING)


def run_search_demo(queries: Optional[List[str]] = None, top_k: int = 3):
    vstore = VectorStore()
    count = vstore.count()
    print(f"\nConnected to Vector DB at: {vstore.db_path} (Total indexed documents: {count})")

    demo_queries = queries or [
        "chicken meal in Gulberg",
        "spicy Pakistani food",
        "meal under 1500 rupees",
    ]

    for q in demo_queries:
        print(f"\n=======================================================")
        print(f"QUERY: '{q}' (Top {top_k} results)")
        print(f"=======================================================")
        results = vstore.search(q, top_k=top_k)
        if not results:
            print("No matching items found.")
            continue

        for rank, res in enumerate(results, start=1):
            halal_badge = "Halal: Yes" if res["halal"] else "Halal: No"
            allergens_str = ", ".join(res["allergens"]) if res["allergens"] else "None listed"
            print(
                f"{rank}. {res['dish_name']} | {res['restaurant_name']}\n"
                f"   Price: PKR {res['price_pkr']:.1f} | Location: {res['location']} | {halal_badge}\n"
                f"   Spice: {res['spice_level']} | Allergens: {allergens_str} ({res['allergen_status']})\n"
                f"   Cosine Similarity: {res['similarity']:.4f} (Distance: {res['distance']:.4f}) | ID: {res['id']}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Query the persistent menu vector store.")
    parser.add_argument("query", nargs="?", default=None, help="Optional search query string.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to retrieve.")
    args = parser.parse_args()

    queries = [args.query] if args.query else None
    run_search_demo(queries=queries, top_k=args.top_k)


if __name__ == "__main__":
    main()
