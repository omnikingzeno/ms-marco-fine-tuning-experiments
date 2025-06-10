# ceata the training triples from query and passage data for the hard negatives

import pandas as pd
import argparse
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create training triples from query and passage data"
    )

    # Required file paths
    parser.add_argument(
        "--filtered-queries",
        type=str,
        required=True,
        help="Path to filtered_queries.train.tsv (columns: query_id, query_text)",
    )

    parser.add_argument(
        "--qrels",
        type=str,
        required=True,
        help="Path to sorted_qrels.train.tsv (columns: query_id, 0, passage_id, 1)",
    )

    parser.add_argument(
        "--random-samples",
        type=str,
        required=True,
        help="Path to random_samples.tsv (columns: query_id, passage_id)",
    )

    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Path to collection.tsv (columns: passage_id, passage_text)",
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save the output file",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="triples.train.hard_negatives.tsv",
        help="Name of the output file",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Chunk size for processing collection.tsv",
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    # Log configuration
    logger.info("Starting training data creation with parameters:")
    logger.info(f"Filtered queries: {args.filtered_queries}")
    logger.info(f"QRels file: {args.qrels}")
    logger.info(f"Random samples: {args.random_samples}")
    logger.info(f"Collection file: {args.collection}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Chunk size: {args.chunk_size}")

    # Load all the necessary files
    logger.info("Loading query and passage data...")
    filtered_queries = pd.read_csv(
        args.filtered_queries,
        sep="\t",
        header=None,
        names=["query_id", "query_text"],
    )

    qrels = pd.read_csv(
        args.qrels,
        sep="\t",
        header=None,
        names=["query_id", "zero", "passage_id", "one"],
    )

    random_samples = pd.read_csv(args.random_samples, sep="\t")

    # Create lookup dictionaries
    query_id_to_text = dict(
        zip(filtered_queries["query_id"], filtered_queries["query_text"])
    )

    # Get positive passage ID for each query (first one if multiple)
    query_to_positive = {}
    for query_id, group in qrels.groupby("query_id"):
        if query_id in query_id_to_text:
            query_to_positive[query_id] = group.iloc[0]["passage_id"]

    # Get negative passage ID for each query
    query_to_negative = dict(
        zip(random_samples["query_id"], random_samples["passage_id"])
    )

    # Find queries that have both positive and negative passages
    valid_query_ids = set(query_to_positive.keys()).intersection(
        set(query_to_negative.keys())
    )
    logger.info(
        f"Found {len(valid_query_ids)} queries with both positive and negative passages"
    )

    # Collect all passage IDs we need to look up
    passage_ids_to_find = set()
    for query_id in valid_query_ids:
        passage_ids_to_find.add(query_to_positive[query_id])
        passage_ids_to_find.add(query_to_negative[query_id])
    logger.info(f"Need to find {len(passage_ids_to_find)} unique passages")

    # Create a mapping from passage_id to its text
    passage_dict = {}

    # Read collection.tsv in chunks to avoid memory issues
    chunk_size = args.chunk_size
    chunk_count = 0
    logger.info("Processing collection to find passages...")
    for chunk in pd.read_csv(
        args.collection,
        sep="\t",
        header=None,
        names=["passage_id", "passage_text"],
        chunksize=chunk_size,
    ):
        chunk_count += 1
        if chunk_count % 10 == 0:
            logger.info(
                f"Processing chunk {chunk_count}, {len(passage_ids_to_find)} passages still needed"
            )

        # Filter to only the passages we need
        filtered_chunk = chunk[chunk["passage_id"].isin(passage_ids_to_find)]

        # Add to our passage dictionary
        for _, row in filtered_chunk.iterrows():
            passage_dict[row["passage_id"]] = row["passage_text"]
            passage_ids_to_find.discard(row["passage_id"])

        # If we've found all passages, we can stop
        if not passage_ids_to_find:
            logger.info("Found all needed passages!")
            break

    logger.info(f"Found {len(passage_dict)} passage texts")

    # Now create the triples file
    count = 0
    logger.info(f"Creating triples file: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for query_id in valid_query_ids:
            # Get the query text
            query_text = query_id_to_text[query_id]

            # Get passage IDs
            pos_passage_id = query_to_positive[query_id]
            neg_passage_id = query_to_negative[query_id]

            # Skip if we couldn't find the passage text
            if pos_passage_id not in passage_dict or neg_passage_id not in passage_dict:
                continue

            pos_passage_text = passage_dict[pos_passage_id]
            neg_passage_text = passage_dict[neg_passage_id]

            # Write the triple to the file
            f.write(f"{query_text}\t{pos_passage_text}\t{neg_passage_text}\n")
            count += 1

    logger.info(f"Created {output_path} with {count} triples")

    # Return stats for anyone using this as a module
    return {
        "total_queries": len(filtered_queries),
        "valid_queries": len(valid_query_ids),
        "triples_created": count,
        "output_path": output_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


if __name__ == "__main__":
    main()
