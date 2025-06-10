# used to find a random sample among the top 200 results of a base model which will be the hard negative, used for building
# the hard negative dataset

import pandas as pd
import numpy as np

# Load base_model_train_top200_results.tsv
base = pd.read_csv("base_model_train_top200_results.tsv", sep="\t")

# Load sorted_qrels.train.tsv (no header)
qrels = pd.read_csv(
    "sorted_qrels.train.tsv",
    sep="\t",
    header=None,
    names=["query_id", "zero", "passage_id", "one"],
)

# Build a set of (query_id, passage_id) pairs from qrels for fast lookup
qrels_set = set(zip(qrels["query_id"], qrels["passage_id"]))

# Filter base to only ranks 50 to 200 (inclusive)
base = base[(base["rank"] >= 50) & (base["rank"] <= 200)]

# Group by query_id and sample
samples = []
for query_id, group in base.groupby("query_id"):
    # Exclude passage_ids present in qrels for this query_id
    filtered = group[
        ~group["passage_id"].isin(qrels[qrels["query_id"] == query_id]["passage_id"])
    ]
    if not filtered.empty:
        row = filtered.sample(n=1, random_state=42)
        samples.append({"query_id": query_id, "column_id": row.iloc[0]["passage_id"]})

# Save to random_samples.tsv
out_df = pd.DataFrame(samples)
out_df.to_csv(
    "random_samples.tsv", sep="\t", index=False, columns=["query_id", "passage_id"]
)
