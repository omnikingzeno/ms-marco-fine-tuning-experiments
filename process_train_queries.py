# used to find top 200 similar passages for each query in train set in the training set using sentence-transformers base model
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
        description="Process train queries and find similar passages"
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="model name or path",
    )
    parser.add_argument(
        "--filtered-queries",
        type=str,
        default="/root/4th-sem/datasets/filtered_queries.train.tsv",
        help="Path to filtered_queries.train.tsv file",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="/root/4th-sem/datasets/collection.tsv",
        help="Path to collection.tsv file",
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Directory to save results"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./cache", help="Directory to cache embeddings"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for encoding passages"
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=1024,
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
        default=256,
        help="Batch size for processing query results",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for encoding if available",
        default=True,
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
        default=None,
        help="Maximum number of queries to process (for testing)",
    )
    parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Enable aggressive memory optimization",
        default=True,
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable faster processing with vectorized operations",
        default=True,
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip computation if result file already exists",
        default=True,
    )
    parser.add_argument(
        "--gpu-final-ranking",
        action="store_true",
        help="Use GPU for final ranking phase",
        default=False,  # Changed to default to False as CPU is faster
    )
    parser.add_argument(
        "--chunk-batches",
        type=int,
        default=10,
        help="Number of chunk files to process at once during final ranking",
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


def save_embeddings(embeddings, file_path):
    """Save embeddings to disk."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(embeddings, file_path)
    logger.info(f"Saved embeddings to {file_path}")


def load_embeddings(file_path, device=None):
    """Load embeddings from disk."""
    if os.path.exists(file_path):
        embeddings = torch.load(file_path, map_location=device)
        logger.info(f"Loaded embeddings from {file_path}")
        return embeddings
    return None


def process_collection_and_search(filtered_queries, args, model):
    """Process collection in optimized chunks and search for similar passages."""
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

    # Generate a cache key for query embeddings
    query_file_hash = compute_file_hash(args.filtered_queries)
    query_cache_key = get_cache_key(args.model, args.filtered_queries, query_file_hash)
    query_cache_path = os.path.join(args.cache_dir, "queries", f"{query_cache_key}.pt")

    # Try to load cached query embeddings
    query_embeddings = load_embeddings(query_cache_path, device)

    # If not found, compute and save them
    if query_embeddings is None:
        logger.info(f"Encoding {len(query_texts)} queries in batches")
        start_time = time.time()

        # Process queries in smaller sub-batches for the large train set
        query_batch_size = args.query_batch_size
        num_queries = len(query_texts)
        sub_batch_embeddings = []

        for start_idx in tqdm(
            range(0, num_queries, query_batch_size), desc="Encoding query batches"
        ):
            end_idx = min(start_idx + query_batch_size, num_queries)
            batch_texts = query_texts[start_idx:end_idx]

            batch_embeddings = model.encode(
                batch_texts,
                batch_size=query_batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
            )

            sub_batch_embeddings.append(batch_embeddings)

            # Free up memory after each batch
            if args.optimize_memory and device == "cuda":
                torch.cuda.empty_cache()

        # Combine all embeddings
        query_embeddings = torch.cat(sub_batch_embeddings, dim=0)

        elapsed_time = time.time() - start_time
        logger.info(f"Query encoding completed in {elapsed_time:.2f} seconds")

        # Save query embeddings for future use
        save_embeddings(query_embeddings, query_cache_path)

    # Process collection in chunks
    logger.info(f"Processing collection: {args.collection}")
    chunk_size = args.chunk_size

    # Check for final results file first (if skip_existing is enabled)
    final_output_path = os.path.join(
        args.output_dir, "base_model_train_top200_results.tsv"
    )
    if args.skip_existing and os.path.exists(final_output_path):
        logger.info(
            f"Final results file {final_output_path} already exists. Loading existing results."
        )
        final_df = pd.read_csv(final_output_path, sep="\t")
        return final_df

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
    # Use a fixed directory name to reuse between runs
    intermediate_dir = os.path.join(args.cache_dir, "results", "run_1746103506")
    os.makedirs(intermediate_dir, exist_ok=True)

    # Process each chunk of passages
    total_chunks_processed = 0
    chunk_result_files = []

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

            # Create cache paths for this chunk
            passage_cache_path = os.path.join(
                args.cache_dir,
                "passages",
                f"{chunk_id}_{get_cache_key(args.model, args.collection)}.pt",
            )
            chunk_results_path = os.path.join(
                intermediate_dir, f"{chunk_id}_results.parquet"
            )

            # Skip to writing results if they already exist
            if os.path.exists(chunk_results_path):
                logger.info(
                    f"Found existing results for {chunk_id} at {chunk_results_path}"
                )
                chunk_result_files.append(chunk_results_path)
                total_chunks_processed += 1
                continue

            # Try to load cached passage embeddings
            passage_embeddings = load_embeddings(passage_cache_path, device)

            # If not found, compute and save them
            if passage_embeddings is None:
                logger.info(
                    f"Encoding chunk {total_chunks_processed+1}/{total_chunks} with {len(passages)} passages"
                )
                passage_embeddings = model.encode(
                    passages,
                    batch_size=args.batch_size,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                )

                # Save passage embeddings for future use
                save_embeddings(passage_embeddings, passage_cache_path)

            # Process queries in smaller batches to avoid OOM with large query set
            logger.info("Computing similarities in batches")
            query_process_batch_size = args.query_process_batch_size
            num_queries = query_embeddings.shape[0]

            chunk_results = []

            for start_idx in tqdm(
                range(0, num_queries, query_process_batch_size),
                desc=f"Processing chunk {total_chunks_processed+1}/{total_chunks}",
            ):
                end_idx = min(start_idx + query_process_batch_size, num_queries)
                batch_query_ids = query_ids[start_idx:end_idx]
                batch_query_embeddings = query_embeddings[start_idx:end_idx]

                # Compute similarities for this batch
                similarity_scores = util.cos_sim(
                    batch_query_embeddings, passage_embeddings
                )

                # Process results for this batch
                if args.fast_mode:
                    # Get top-k directly on GPU
                    k = min(args.final_k, similarity_scores.shape[1])
                    batch_top_scores, batch_top_indices = torch.topk(
                        similarity_scores, k=k, dim=1, largest=True, sorted=True
                    )

                    # Process results
                    batch_top_scores_cpu = batch_top_scores.cpu().numpy()
                    batch_top_indices_cpu = batch_top_indices.cpu().numpy()

                    for i, query_id in enumerate(batch_query_ids):
                        for idx, score in zip(
                            batch_top_indices_cpu[i], batch_top_scores_cpu[i]
                        ):
                            chunk_results.append(
                                {
                                    "query_id": query_id,
                                    "passage_id": passage_ids[idx],
                                    "score": float(score),
                                }
                            )

                    # Clean up batch memory
                    del batch_top_scores, batch_top_indices
                    del batch_top_scores_cpu, batch_top_indices_cpu

                else:
                    # Process one query at a time
                    for q_idx, query_id in enumerate(batch_query_ids):
                        k = min(args.final_k, similarity_scores.shape[1])
                        top_scores, top_indices = torch.topk(
                            similarity_scores[q_idx], k=k, largest=True, sorted=True
                        )

                        top_scores_cpu = top_scores.cpu().numpy()
                        top_indices_cpu = top_indices.cpu().numpy()

                        for idx, score in zip(top_indices_cpu, top_scores_cpu):
                            chunk_results.append(
                                {
                                    "query_id": query_id,
                                    "passage_id": passage_ids[idx],
                                    "score": float(score),
                                }
                            )

                        # Clean up per-query results
                        del top_scores, top_indices

                # Clean up batch memory
                del similarity_scores
                if args.optimize_memory and device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

            # Save chunk results to parquet
            if chunk_results:
                chunk_df = pd.DataFrame(chunk_results)
                chunk_df.to_parquet(chunk_results_path, engine="pyarrow", index=False)
                logger.info(
                    f"Saved {len(chunk_results)} results to {chunk_results_path}"
                )
                chunk_result_files.append(chunk_results_path)

                # Clean up memory
                del chunk_results, chunk_df

            # Clean up GPU memory
            if args.optimize_memory and device == "cuda":
                del passage_embeddings
                torch.cuda.empty_cache()
                gc.collect()

            total_chunks_processed += 1
            logger.info(f"Processed {total_chunks_processed}/{total_chunks} chunks")

    # NEW OPTIMIZED FINAL RANKING SECTION
    # Process in efficient batches using pandas for better performance
    logger.info("Finalizing results across all chunks using optimized CPU processing")

    if not chunk_result_files:
        logger.warning("No chunk result files found. Something may have gone wrong.")
        return pd.DataFrame(columns=["query_id", "passage_id", "rank", "score"])

    # Determine number of chunks to process at once
    chunk_batches = args.chunk_batches
    logger.info(f"Processing in batches of {chunk_batches} chunk files at once")

    # Get all unique query IDs first from all chunk files
    logger.info("Scanning for unique query IDs...")
    all_query_ids = set()
    for chunk_path in tqdm(chunk_result_files, desc="Scanning chunks"):
        chunk_df = pd.read_parquet(chunk_path, columns=["query_id"])
        all_query_ids.update(chunk_df["query_id"].unique())
    unique_query_ids = sorted(list(all_query_ids))
    total_queries = len(unique_query_ids)
    logger.info(f"Processing final rankings for {total_queries} unique queries")

    # Final results container
    final_results = []

    # Process chunk files in larger batches

    for i in tqdm(
        range(0, len(chunk_result_files), chunk_batches),
        desc="Processing chunk batches",
    ):
        # Get current batch of chunk files
        batch_files = chunk_result_files[
            i : min(i + chunk_batches, len(chunk_result_files))
        ]

        logger.info(f"Loading {len(batch_files)} chunk files")

        # Process files incrementally to reduce peak memory
        batch_df = None
        for file_idx, file in enumerate(batch_files):
            # Only read required columns
            df = pd.read_parquet(file, columns=["query_id", "passage_id", "score"])

            if file_idx % 5 == 0:
                logger.info(f"Loaded file {file_idx+1}/{len(batch_files)}")

            if batch_df is None:
                batch_df = df
            else:
                batch_df = pd.concat([batch_df, df], ignore_index=True)
            del df  # Free memory immediately

        if batch_df is None or len(batch_df) == 0:
            continue

        logger.info(f"Processing combined dataframe with {len(batch_df)} rows")

        # Process in larger batches to speed up
        query_batch_size = 20000  # Increased from 5000
        for j in tqdm(
            range(0, len(unique_query_ids), query_batch_size),
            desc=f"Processing query batches",
        ):
            end_j = min(j + query_batch_size, len(unique_query_ids))
            batch_query_ids = unique_query_ids[j:end_j]

            # Filter by batch query IDs
            mask = batch_df["query_id"].isin(batch_query_ids)
            filtered_df = batch_df[mask].copy()

            if len(filtered_df) == 0:
                continue

            logger.info(
                f"Processing {len(filtered_df)} rows for {len(batch_query_ids)} queries"
            )

            # Fix the deprecated behavior warning
            batch_results = filtered_df.groupby("query_id", group_keys=False).apply(
                lambda x: x.nlargest(args.final_k, "score")
            )

            # Add ranking
            batch_results["rank"] = batch_results.groupby("query_id").cumcount() + 1

            # Append to results
            final_results.append(batch_results)

            # Clean up memory more aggressively
            del filtered_df, batch_results
            gc.collect()

    # Clean up memory
    del batch_df
    gc.collect()  # Combine all results
    if final_results:
        # Combine all batches into the final dataframe
        final_df = pd.concat(final_results, ignore_index=True)

        # Get the true top-k for each query by combining all batches
        final_df = (
            final_df.groupby("query_id")
            .apply(lambda x: x.nlargest(args.final_k, "score"))
            .reset_index(drop=True)
        )

        # Re-rank to get correct ranks
        final_df["rank"] = final_df.groupby("query_id").cumcount() + 1

        # Sort by query_id and rank
        final_df = final_df.sort_values(["query_id", "rank"])

        # Select only needed columns
        final_df = final_df[["query_id", "passage_id", "rank", "score"]]
    else:
        logger.warning("No results found in final processing")
        final_df = pd.DataFrame(columns=["query_id", "passage_id", "rank", "score"])

    # Save results to tsv file
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "base_model_train_top200_results.tsv")
    final_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved results to {output_path}")

    return final_df


def save_results_to_json(results_df, args):
    """Save results metadata to JSON file"""
    output_path = os.path.join(
        args.output_dir, "base_model_train_top200_evaluation.json"
    )

    # Prepare output dictionary
    output_dict = {
        "model": args.model,
        "parameters": {
            "batch_size": args.batch_size,
            "query_batch_size": args.query_batch_size,
            "chunk_size": args.chunk_size,
            "query_process_batch_size": args.query_process_batch_size,
            "precision": args.precision,
            "use_gpu": args.use_gpu,
            "max_queries": args.max_queries,
            "fast_mode": args.fast_mode,
            "final_k": args.final_k,
            "gpu_final_ranking": args.gpu_final_ranking,
            "chunk_batches": args.chunk_batches,
        },
        "files": {
            "filtered_queries": args.filtered_queries,
            "collection": args.collection,
        },
        "stats": {
            "num_queries": (
                len(results_df["query_id"].unique()) if not results_df.empty else 0
            ),
            "total_results": len(results_df) if not results_df.empty else 0,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save to file
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    logger.info(f"Saved evaluation metadata to {output_path}")


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if final result file exists
    final_output_path = os.path.join(
        args.output_dir, "base_model_train_top200_results.tsv"
    )
    if args.skip_existing and os.path.exists(final_output_path):
        logger.info(
            f"Final results file {final_output_path} already exists. Loading..."
        )
        results_df = pd.read_csv(final_output_path, sep="\t")
        save_results_to_json(results_df, args)
        logger.info("Using existing results. Processing completed successfully.")
        return

    # Load the model just once
    model = load_model(args)

    # Step 1: Load train queries
    filtered_queries = load_filtered_queries(args)

    # Step 2 & 3: Process collection and search for similar passages
    results_df = process_collection_and_search(filtered_queries, args, model)

    # Step 4: Save results metadata in JSON format
    save_results_to_json(results_df, args)

    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
