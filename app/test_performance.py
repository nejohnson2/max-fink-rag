#!/usr/bin/env python3
"""Performance profiling for RAG system"""

import time
from rag_system_v3 import RAGSystem
from config import logger

def time_it(label, func):
    start = time.time()
    result = func()
    elapsed = time.time() - start
    logger.info(f"⏱️  {label}: {elapsed:.2f}s")
    return result, elapsed

def main():
    logger.info("=== RAG Performance Test ===")

    # Test initialization
    _, init_time = time_it("System initialization", lambda: RAGSystem(
        store_dir="./fink_archive",
        chroma_collection="rag_collection",
        enable_bm25=True,
        k_recall=30,
        k_ensemble=20,
        k_after_rerank=6,
    ))

    rag = RAGSystem(
        store_dir="./fink_archive",
        chroma_collection="rag_collection",
        enable_bm25=True,
        k_recall=30,
        k_ensemble=20,
        k_after_rerank=6,
    )

    # Test query
    test_question = "What was Max Fink's educational background?"

    logger.info(f"\n📝 Testing query: {test_question}")

    # Full query timing
    start_total = time.time()

    # 1. Intent classification
    _, intent_time = time_it("  1. Intent classification",
                              lambda: rag.classify_intent(test_question))

    # 2. Full query (includes retrieval + reranking + answer generation)
    result, query_time = time_it("  2. Full query",
                                  lambda: rag.ask(test_question))

    total_time = time.time() - start_total

    logger.info(f"\n📊 Summary:")
    logger.info(f"  Init time: {init_time:.2f}s")
    logger.info(f"  Intent time: {intent_time:.2f}s ({intent_time/total_time*100:.1f}%)")
    logger.info(f"  Query time: {query_time:.2f}s ({query_time/total_time*100:.1f}%)")
    logger.info(f"  Total: {total_time:.2f}s")

    # Test cache benefit
    logger.info(f"\n🔄 Testing intent cache benefit...")
    _, cached_intent_time = time_it("  Cached intent call",
                                     lambda: rag.classify_intent(test_question))
    logger.info(f"  Speedup: {intent_time/cached_intent_time:.1f}x faster")

if __name__ == "__main__":
    main()
