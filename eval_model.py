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
    parser = argparse.ArgumentParser(description="Evaluate model for passage retrieval")

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
        required=True,
        help="Path to filtered_queries.dev.tsv file",
    )
    parser.add_argument(
        "--collection", type=str, required=True, help="Path to collection.tsv file"
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Directory to save results"
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
        default=None,
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
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip computation if result file already exists",
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

        query_embeddings = model.encode(
            query_texts,
            batch_size=args.query_batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Query encoding completed in {elapsed_time:.2f} seconds")

        # Save query embeddings for future use
        save_embeddings(query_embeddings, query_cache_path)

    # Process collection in chunks
    logger.info(f"Processing collection: {args.collection}")
    chunk_size = args.chunk_size

    # Check for final results file first (if skip_existing is enabled)
    final_output_path = os.path.join(
        args.output_dir, "sbert_finetuned_lora_1m_results.tsv"
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
    intermediate_dir = os.path.join(
        args.cache_dir, "results", f"run_{int(time.time())}"
    )
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
            else:
                # Validate that embeddings match passage count
                if passage_embeddings.shape[0] != len(passage_ids):
                    logger.warning(
                        f"Mismatch between loaded embeddings ({passage_embeddings.shape[0]}) and passage IDs ({len(passage_ids)}). "
                        f"Regenerating embeddings for consistency."
                    )
                    # Force regeneration of embeddings
                    passage_embeddings = model.encode(
                        passages,
                        batch_size=args.batch_size,
                        show_progress_bar=True,
                        convert_to_tensor=True,
                    )
                    # Save the regenerated embeddings
                    save_embeddings(passage_embeddings, passage_cache_path)
            # Compute similarities
            logger.info("Computing similarities for all queries against passages")
            similarity_scores = util.cos_sim(query_embeddings, passage_embeddings)

            # Process results and save to parquet
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
                            }
                        )

                    # Clean up per-query results
                    del top_scores
                    del top_indices

            # Save chunk results to parquet for efficient downstream processing
            if chunk_results:
                chunk_df = pd.DataFrame(chunk_results)
                chunk_df.to_parquet(chunk_results_path, engine="pyarrow", index=False)
                logger.info(
                    f"Saved {len(chunk_results)} results to {chunk_results_path}"
                )
                chunk_result_files.append(chunk_results_path)

                # Clean up memory
                del chunk_results
                del chunk_df

            # Clean up GPU memory
            if args.optimize_memory:
                del passage_embeddings
                del similarity_scores
                torch.cuda.empty_cache()
                gc.collect()

            total_chunks_processed += 1
            logger.info(f"Processed {total_chunks_processed}/{total_chunks} chunks")

    # Final processing using Dask for memory efficiency
    logger.info("Finalizing results across all chunks using Dask")

    if not chunk_result_files:
        logger.warning("No chunk result files found. Something may have gone wrong.")
        return pd.DataFrame(columns=["query_id", "passage_id", "rank", "score"])

    # Use Dask to read and process all intermediate results efficiently
    dask_df = dd.read_parquet(chunk_result_files, engine="pyarrow")

    # Count total number of results (for information only)
    result_count = len(dask_df)
    logger.info(f"Processing {result_count} total results with Dask")

    # Compute top-k per query using Dask
    logger.info("Finding top passages per query")

    # Step 1: Use Dask to pre-select top candidates per query (getting more than we need)
    top_k_multiplier = 2  # Get 2x as many results as needed to ensure we have enough
    pre_top_k = args.final_k * top_k_multiplier

    # We'll save pre-filtered results to a new parquet file
    pre_filtered_path = os.path.join(intermediate_dir, "pre_filtered_results.parquet")

    # Check if we need to compute this or if it already exists
    if not os.path.exists(pre_filtered_path):
        # Process queries in batches and use GPU when possible
        # Get unique query IDs
        unique_query_ids = dask_df["query_id"].unique().compute()
        total_queries = len(unique_query_ids)
        logger.info(
            f"Processing {total_queries} unique queries using GPU-accelerated batching"
        )

        # Define batch size based on available memory
        # Using larger batches for better efficiency
        batch_size = 1000  # Process 1000 queries at once
        num_batches = (total_queries + batch_size - 1) // batch_size

        # Process in batches and write results directly to disk instead of accumulating in memory
        batch_result_files = []

        for batch_idx in tqdm(range(num_batches), desc="Processing query batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_queries)
            batch_query_ids = unique_query_ids[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_query_ids)} queries"
            )

            # Create a mask for this batch (much faster than processing one query at a time)
            batch_mask = dask_df["query_id"].isin(batch_query_ids)
            batch_data = dask_df[batch_mask]

            # Convert to pandas for efficient processing
            logger.info("Computing batch data from Dask")
            batch_df = batch_data.compute()
            logger.info(f"Loaded batch data: {len(batch_df)} rows")

            if len(batch_df) > 0:
                # Check if GPU is available for batch processing
                if torch.cuda.is_available() and args.use_gpu:
                    # Group by query_id first to make processing more efficient
                    batch_groups = batch_df.groupby("query_id")
                    result_dfs = []

                    # Process each query with GPU acceleration
                    for query_id, group in tqdm(
                        batch_groups, desc="GPU-accelerated filtering"
                    ):
                        if len(group) <= pre_top_k:
                            # If fewer results than needed, keep all
                            result_dfs.append(group)
                        else:
                            # Use GPU to find top-k scores
                            scores = torch.tensor(group["score"].values, device="cuda")
                            top_k_values, top_k_indices = torch.topk(
                                scores, k=min(pre_top_k, len(scores)), largest=True
                            )
                            top_k_results = group.iloc[top_k_indices.cpu().numpy()]
                            result_dfs.append(top_k_results)

                    if result_dfs:
                        filtered_batch = pd.concat(result_dfs)
                    else:
                        filtered_batch = pd.DataFrame(columns=batch_df.columns)
                else:
                    # CPU path with optimized pandas operations
                    # Use pandas groupby for top-k which is still quite efficient
                    logger.info("Using pandas for batch filtering (GPU not used)")
                    filtered_batch = (
                        batch_df.groupby("query_id")
                        .apply(lambda x: x.nlargest(pre_top_k, "score"))
                        .reset_index(drop=True)
                    )

                # Save batch results to a temporary file
                batch_file_path = os.path.join(
                    intermediate_dir, f"batch_{batch_idx}_filtered.parquet"
                )
                filtered_batch.to_parquet(
                    batch_file_path, engine="pyarrow", index=False
                )
                logger.info(
                    f"Saved {len(filtered_batch)} filtered results to {batch_file_path}"
                )
                batch_result_files.append(batch_file_path)

            # Force cleanup
            del batch_df
            if "filtered_batch" in locals():
                del filtered_batch
            if "result_dfs" in locals():
                del result_dfs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Now combine all batch results efficiently - read with Dask and write once
        logger.info(f"Combining results from {len(batch_result_files)} batch files")
        combined_dask_df = dd.read_parquet(batch_result_files, engine="pyarrow")

        # Write combined results to final pre-filtered file
        # Using optimize_dataframe_size to prevent memory issues when writing
        combined_dask_df = combined_dask_df.optimize()
        combined_dask_df.to_parquet(
            pre_filtered_path, engine="pyarrow", write_index=False, compute=True
        )

        # Get count of saved results
        pre_filtered_count = combined_dask_df.shape[0].compute()
        logger.info(
            f"Saved {pre_filtered_count} pre-filtered results to {pre_filtered_path}"
        )

        # Remove temporary batch files
        logger.info("Removing temporary batch files")
        for batch_file in batch_result_files:
            if os.path.exists(batch_file):
                os.remove(batch_file)

    # Step 2: Load pre-filtered results efficiently
    logger.info(f"Loading pre-filtered results from {pre_filtered_path}")

    # Use Dask to read the pre-filtered results to avoid memory issues
    pre_filtered_dask = dd.read_parquet(pre_filtered_path, engine="pyarrow")

    # Get count
    pre_filtered_count = pre_filtered_dask.shape[0].compute()
    logger.info(f"Pre-filtered results count: {pre_filtered_count}")

    # Now do the final top-k selection efficiently
    if pre_filtered_count > 0:
        logger.info("Calculating final rankings")

        # Use Dask to group and find top-k for each query
        # This is more memory efficient than bringing all data into pandas at once

        # Process in smaller batches for final ranking
        unique_query_ids = pre_filtered_dask["query_id"].unique().compute()
        final_results = []

        # Define function to get exact top-k with ranking
        def get_top_k_ranked(df):
            top = df.nlargest(args.final_k, "score")
            top = top.reset_index(drop=True)
            top["rank"] = range(1, len(top) + 1)
            return top

        # Process in reasonable batch sizes
        final_batch_size = 1000
        num_batches = (len(unique_query_ids) + final_batch_size - 1) // final_batch_size

        for batch_idx in tqdm(range(num_batches), desc="Final ranking batches"):
            start_idx = batch_idx * final_batch_size
            end_idx = min(start_idx + final_batch_size, len(unique_query_ids))
            batch_query_ids = unique_query_ids[start_idx:end_idx]

            # Filter just the queries in this batch
            batch_mask = pre_filtered_dask["query_id"].isin(batch_query_ids)
            batch_data = pre_filtered_dask[batch_mask].compute()

            if len(batch_data) > 0:
                # Group by query_id and get top-k with ranking
                batch_results = batch_data.groupby("query_id").apply(get_top_k_ranked)
                if len(batch_results) > 0:
                    final_results.append(batch_results.reset_index(drop=True))

            # Clean up
            del batch_data
            gc.collect()

        if final_results:
            # Combine all batches into the final dataframe
            final_df = pd.concat(final_results, ignore_index=True)
            final_df = final_df.sort_values(["query_id", "rank"])

            # Select only needed columns
            final_df = final_df[["query_id", "passage_id", "rank", "score"]]
        else:
            logger.warning("No results found in final processing")
            final_df = pd.DataFrame(columns=["query_id", "passage_id", "rank", "score"])
    else:
        logger.warning("No pre-filtered results found")
        final_df = pd.DataFrame(columns=["query_id", "passage_id", "rank", "score"])

    # Save results to tsv file
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "sbert_finetuned_lora_1m_results.tsv")
    final_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved results to {output_path}")

    return final_df


def save_results_to_json(results_df, args):
    """Save results to JSON file"""
    output_path = os.path.join(
        args.output_dir, "sbert_finetuned_lora_1m_evaluation.json"
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
        },
        "files": {
            "filtered_queries": args.filtered_queries,
            "collection": args.collection,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save to file
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    logger.info(f"Saved evaluation results to {output_path}")


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if final result file exists
    final_output_path = os.path.join(
        args.output_dir, "sbert_finetuned_lora_1m_results.tsv"
    )
    if args.skip_existing and os.path.exists(final_output_path):
        logger.info(
            f"Final results file {final_output_path} already exists. Loading..."
        )
        results_df = pd.read_csv(final_output_path, sep="\t")
        save_results_to_json(results_df, args)
        logger.info("Using existing results. Evaluation completed successfully.")
        return

    # Load the model just once
    model = load_model(args)

    # Step 1: Load pre-filtered queries
    filtered_queries = load_filtered_queries(args)

    # Step 2 & 3: Process collection and search for similar passages
    results_df = process_collection_and_search(filtered_queries, args, model)

    # Step 4: Save results in JSON format
    save_results_to_json(results_df, args)

    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
