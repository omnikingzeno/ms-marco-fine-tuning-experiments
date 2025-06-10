import pandas as pd
import random

# Set random seed for reproducibility
random.seed(42)

# Define file paths
filtered_queries_path = "filtered_queries.dev.tsv"
qrels_path = "qrels.dev.tsv"
collection_path = r"collection\collection.tsv"
output_path = "visualization_data.tsv"

# Read filtered_queries.dev.tsv
print(f"Reading queries from {filtered_queries_path}")
queries_df = pd.read_csv(
    filtered_queries_path, sep="\t", header=None, names=["query_id", "query_text"]
)

# Randomly sample 1000 query_ids (or less if there are fewer than 1000)
num_samples = min(1000, len(queries_df))
print(f"Sampling {num_samples} queries")
sampled_query_ids = random.sample(list(queries_df["query_id"]), num_samples)

# Read qrels.dev.tsv
print(f"Reading qrels from {qrels_path}")
qrels_df = pd.read_csv(
    qrels_path, sep="\t", header=None, names=["query_id", "zero", "passage_id", "one"]
)

# Filter qrels_df to only include sampled query_ids and keep only one passage per query
filtered_qrels_df = qrels_df[
    qrels_df["query_id"].isin(sampled_query_ids)
].drop_duplicates(subset=["query_id"])
print(f"Found {len(filtered_qrels_df)} matching qrels entries")

# Read collection.tsv
print(f"Reading collection from {collection_path}")
collection_df = pd.read_csv(
    collection_path, sep="\t", header=None, names=["passage_id", "passage_text"]
)

# Merge dataframes to get query_text and passage_text
merged_df = filtered_qrels_df.merge(queries_df, on="query_id").merge(
    collection_df, on="passage_id"
)

# Select only the required columns
visualization_df = merged_df[["query_text", "passage_text"]]
print(f"Created {len(visualization_df)} entries for visualization data")

# Write to visualization_data.tsv
visualization_df.to_csv(output_path, sep="\t", index=False, header=False)
print(f"Visualization data saved to {output_path}")
