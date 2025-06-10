import argparse
import json
import logging
import os
import time
from typing import Dict, List
import gc
import glob
import hashlib

import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark model inference time for passage retrieval"
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model name or path",
    )
    parser.add_argument(
        "--filtered-queries",
        type=str,
        required=True,
        help="Path to filtered_queries.dev.tsv file",
    )
    parser.add_argument(
        "--collection", type=str, required=True, help="Path to collection.tsv file"
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./cache", help="Directory to cache embeddings"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for encoding passages"
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=64,
        help="Batch size for encoding queries",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200000,
        help="Chunk size for processing collection",
    )
    parser.add_argument(
        "--query-process-batch-size",
        type=int,
        default=128,
        help="Batch size for processing query results",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU for encoding if available"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float16",
        choices=["float32", "float16"],
        help="Precision for model weights",
    )
    parser.add_argument(
        "--final-k", type=int, default=200, help="Final number of results per query"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=10000,
        help="Maximum number of queries to process (for testing)",
    )
    parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Enable aggressive memory optimization",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable faster processing with vectorized operations",
    )

    return parser.parse_args()


def get_cache_key(model_name, data_source, subset_hash=None):
    """Generate a cache key based on the model and data source."""
    # Create a deterministic but readable key
    model_id = os.path.basename(model_name).replace("/", "_")
    if subset_hash:
        return f"{model_id}_{os.path.basename(data_source)}_{subset_hash}"
    else:
        return f"{model_id}_{os.path.basename(data_source)}"


def compute_file_hash(file_path, max_size=10000):
    """Compute a hash of the first max_size bytes of a file."""
    hash_obj = hashlib.md5()
    with open(file_path, "rb") as f:
        data = f.read(max_size)
        hash_obj.update(data)
    return hash_obj.hexdigest()[:8]  # Use first 8 chars for readability


def load_filtered_queries(args):
    """Load and process filtered queries."""
    logger.info(f"Loading filtered queries from {args.filtered_queries}")
    queries_df = pd.read_csv(
        args.filtered_queries, sep="\t", header=None, names=["query_id", "query_text"]
    )

    # Limit number of queries if specified (for testing)
    if args.max_queries is not None:
        queries_df = queries_df.head(args.max_queries)

    logger.info(f"Loaded {len(queries_df)} queries")
    return queries_df


def load_model(args):
    """Load the model once to be reused."""
    logger.info(f"Loading model: {args.model}")

    # Set device
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"

    # Configure model precision
    fp16 = True if args.precision == "float16" else False

    # Load model
    model = SentenceTransformer(args.model, device=device)
    if fp16:
        model.half()

    return model


def load_embeddings(file_path, device=None):
    """Load embeddings from disk."""
    if os.path.exists(file_path):
        embeddings = torch.load(file_path, map_location=device)
        logger.info(f"Loaded embeddings from {file_path}")
        return embeddings
    return None


def save_embeddings(embeddings, file_path):
    """Save embeddings to disk, without measuring the time."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(embeddings, file_path)
    logger.info(f"Saved embeddings to {file_path}")


def process_collection_and_search(filtered_queries, args, model):
    """Process collection and search for similar passages, timing only the inference parts."""
    # Create cache directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.join(args.cache_dir, "queries"), exist_ok=True)
    os.makedirs(os.path.join(args.cache_dir, "passages"), exist_ok=True)
    os.makedirs(os.path.join(args.cache_dir, "results"), exist_ok=True)

    # Set device
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"

    # Prepare query data
    query_ids = filtered_queries["query_id"].tolist()
    query_texts = filtered_queries["query_text"].tolist()

    # Generate cache path for query embeddings (for optional saving, not loading)
    query_file_hash = compute_file_hash(args.filtered_queries)
    query_cache_key = get_cache_key(args.model, args.filtered_queries, query_file_hash)
    query_cache_path = os.path.join(args.cache_dir, "queries", f"{query_cache_key}.pt")

    inference_times = {
        "query_embedding": 0,
        "similarity_calculation": 0,
        "ranking_and_result_generation": 0,
        "total": 0,
    }

    # Record start time for total inference
    total_start_time = time.time()

    # Always compute query embeddings (don't use cache for benchmarking)
    logger.info(f"Encoding {len(query_texts)} queries in batches")
    query_embedding_start = time.time()

    query_embeddings = model.encode(
        query_texts,
        batch_size=args.query_batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )

    inference_times["query_embedding"] = time.time() - query_embedding_start
    logger.info(
        f"Query encoding completed in {inference_times['query_embedding']:.2f} seconds"
    )

    # Optionally save query embeddings (not counted in benchmark time)
    # save_embeddings(query_embeddings, query_cache_path)

    # Process collection in chunks
    logger.info(f"Processing collection: {args.collection}")
    chunk_size = args.chunk_size

    # Estimate collection size for progress tracking
    with open(args.collection, "r", encoding="utf-8") as f:
        i = 0
        while i < 10:
            line = f.readline()
            if not line:  # End of file
                break
            i += 1
        avg_line_size = f.tell() / (i + 1) if i > 0 else 100  # Default if file is empty

    f_size = os.path.getsize(args.collection)
    total_chunks = int(f_size / (avg_line_size * chunk_size)) + 1
    logger.info(f"Estimated {total_chunks} chunks to process")

    # Create directory for intermediate results
    intermediate_dir = os.path.join(
        args.cache_dir, "results", f"run_{int(time.time())}"
    )
    os.makedirs(intermediate_dir, exist_ok=True)

    # Hold all results in memory for benchmarking
    all_results = []

    # Process each chunk of passages
    total_chunks_processed = 0

    with open(args.collection, "r", encoding="utf-8") as f:
        while True:
            passages = []
            passage_ids = []

            # Read chunk_size lines
            for _ in range(chunk_size):
                line = f.readline().strip()
                if not line:
                    break

                # Parse passage ID and text
                parts = line.split("\t")
                if len(parts) == 2:
                    passage_id, passage_text = parts
                    passage_ids.append(passage_id)
                    passages.append(passage_text)

            if not passages:
                break

            # Generate unique identifier for this chunk
            chunk_id = f"chunk_{total_chunks_processed}"

            # Create cache path for passage embeddings
            passage_cache_path = os.path.join(
                args.cache_dir,
                "passages",
                f"{chunk_id}_{get_cache_key(args.model, args.collection)}.pt",
            )

            # Load cached passage embeddings (this time is not counted)
            passage_embeddings = load_embeddings(passage_cache_path, device)

            # If embeddings are not found, compute them (time not counted in benchmark)
            if passage_embeddings is None:
                logger.info(
                    f"Encoding chunk {total_chunks_processed+1}/{total_chunks} with {len(passages)} passages"
                )

                # This time is not counted in our benchmark
                passage_embeddings = model.encode(
                    passages,
                    batch_size=args.batch_size,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                )

                # Save passage embeddings for future use (not counted in benchmark)
                save_embeddings(passage_embeddings, passage_cache_path)
            else:
                # Validate that embeddings match passage count
                if passage_embeddings.shape[0] != len(passage_ids):
                    logger.warning(
                        f"Mismatch between loaded embeddings ({passage_embeddings.shape[0]}) and passage IDs ({len(passage_ids)}). "
                        f"Regenerating embeddings for consistency."
                    )
                    # Force regeneration of embeddings (not counted in benchmark)
                    passage_embeddings = model.encode(
                        passages,
                        batch_size=args.batch_size,
                        show_progress_bar=True,
                        convert_to_tensor=True,
                    )
                    # Save the regenerated embeddings (not counted)
                    save_embeddings(passage_embeddings, passage_cache_path)

            # Compute similarities (this time is counted)
            logger.info("Computing similarities for all queries against passages")

            similarity_start = time.time()
            similarity_scores = util.cos_sim(query_embeddings, passage_embeddings)
            chunk_similarity_time = time.time() - similarity_start
            inference_times["similarity_calculation"] += chunk_similarity_time

            logger.info(
                f"Similarity computation completed in {chunk_similarity_time:.2f} seconds"
            )

            # Process results (this time is counted)
            ranking_start = time.time()
            chunk_results = []

            if args.fast_mode:
                # GPU optimization: Find top-k directly on GPU
                logger.info("Finding top-k scores on GPU")
                batch_size = args.query_process_batch_size

                for start_idx in tqdm(
                    range(0, len(query_ids), batch_size),
                    desc="Processing query batches",
                ):
                    end_idx = min(start_idx + batch_size, len(query_ids))
                    batch_query_ids = query_ids[start_idx:end_idx]
                    batch_scores = similarity_scores[start_idx:end_idx]

                    # Get top-k directly on GPU (more efficient than argpartition on CPU)
                    k = min(args.final_k, batch_scores.shape[1])
                    batch_top_scores, batch_top_indices = torch.topk(
                        batch_scores, k=k, dim=1, largest=True, sorted=True
                    )

                    # Now transfer only the top-k results to CPU (much less data)
                    batch_top_scores_cpu = batch_top_scores.cpu().numpy()
                    batch_top_indices_cpu = batch_top_indices.cpu().numpy()

                    # Create results for this batch
                    for i, query_id in enumerate(batch_query_ids):
                        for j, (idx, score) in enumerate(
                            zip(batch_top_indices_cpu[i], batch_top_scores_cpu[i])
                        ):
                            chunk_results.append(
                                {
                                    "query_id": query_id,
                                    "passage_id": passage_ids[idx],
                                    "score": float(score),
                                    "chunk": total_chunks_processed,
                                }
                            )

                    # Clean up memory for this batch
                    del batch_top_scores
                    del batch_top_indices
                    del batch_top_scores_cpu
                    del batch_top_indices_cpu

            else:
                # Traditional process - one query at a time
                for q_idx, query_id in tqdm(
                    enumerate(query_ids),
                    total=len(query_ids),
                    desc="Processing query results",
                ):
                    # Stay on GPU for top-k operations too
                    k = min(args.final_k, similarity_scores.shape[1])
                    top_scores, top_indices = torch.topk(
                        similarity_scores[q_idx], k=k, largest=True, sorted=True
                    )

                    # Transfer only selected results to CPU
                    top_scores_cpu = top_scores.cpu().numpy()
                    top_indices_cpu = top_indices.cpu().numpy()

                    # Add to results
                    for idx, score in zip(top_indices_cpu, top_scores_cpu):
                        chunk_results.append(
                            {
                                "query_id": query_id,
                                "passage_id": passage_ids[idx],
                                "score": float(score),
                                "chunk": total_chunks_processed,
                            }
                        )

                    # Clean up per-query results
                    del top_scores
                    del top_indices

            chunk_ranking_time = time.time() - ranking_start
            inference_times["ranking_and_result_generation"] += chunk_ranking_time
            logger.info(
                f"Ranking and result generation completed in {chunk_ranking_time:.2f} seconds"
            )

            # Add current chunk results to overall results (keeping in memory for benchmark)
            all_results.extend(chunk_results)

            # Clean up GPU memory
            if args.optimize_memory:
                del passage_embeddings
                del similarity_scores
                torch.cuda.empty_cache()
                gc.collect()

            total_chunks_processed += 1
            logger.info(f"Processed {total_chunks_processed}/{total_chunks} chunks")

    # Final ranking across all chunks (this time is counted)
    logger.info("Finalizing results for all queries")
    final_ranking_start = time.time()

    # Convert all results to DataFrame
    all_results_df = pd.DataFrame(all_results)

    # Perform final ranking to get top-k across all chunks
    final_results = []

    for query_id, group in all_results_df.groupby("query_id"):
        top_k = group.nlargest(args.final_k, "score")
        top_k["rank"] = range(1, len(top_k) + 1)
        final_results.append(top_k)

    final_df = pd.concat(final_results)
    final_df = final_df[["query_id", "passage_id", "rank", "score"]]
    final_df = final_df.sort_values(["query_id", "rank"])

    final_ranking_time = time.time() - final_ranking_start
    inference_times["ranking_and_result_generation"] += final_ranking_time
    logger.info(f"Final ranking completed in {final_ranking_time:.2f} seconds")

    # Calculate total inference time
    inference_times["total"] = time.time() - total_start_time

    # Output directory for results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save final results (not counted in benchmark time)
    results_path = os.path.join(
        args.output_dir, f"{os.path.basename(args.model)}_inference_run_results.tsv"
    )
    final_df.to_csv(results_path, sep="\t", index=False)

    # Save timing information
    timing_path = os.path.join(
        args.output_dir, f"{os.path.basename(args.model)}_inference_timing.json"
    )
    with open(timing_path, "w") as f:
        json.dump(inference_times, f, indent=2)

    logger.info(f"Inference time breakdown:")
    logger.info(f"  Query embedding: {inference_times['query_embedding']:.2f} seconds")
    logger.info(
        f"  Similarity calculation: {inference_times['similarity_calculation']:.2f} seconds"
    )
    logger.info(
        f"  Ranking and result generation: {inference_times['ranking_and_result_generation']:.2f} seconds"
    )
    logger.info(f"  Total inference time: {inference_times['total']:.2f} seconds")

    return final_df, inference_times


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model
    model = load_model(args)

    # Load pre-filtered queries
    filtered_queries = load_filtered_queries(args)

    # Process collection and search for similar passages (with timing)
    results_df, timing_info = process_collection_and_search(
        filtered_queries, args, model
    )

    logger.info("Benchmark completed successfully")


if __name__ == "__main__":
    main()
