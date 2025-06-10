import sys
from collections import defaultdict


def calculate_mrr(qrels_filepath, results_filepath):
    """
    Calculates MRR@10 and MRR@100 given qrels and ranked results files.

    Args:
        qrels_filepath (str): Path to the qrels file (TSV: query_id, 0, passage_id, 1).
        results_filepath (str): Path to the results file (TSV: query_id, passage_id, rank, score).

    Returns:
        tuple: A tuple containing (mrr_at_10, mrr_at_100).
               Returns (0.0, 0.0) if no queries are processed.
    """
    # --- 1. Load Qrels (Ground Truth) ---
    qrels = defaultdict(set)
    print(f"Loading qrels from {qrels_filepath}...")
    try:
        with open(qrels_filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    query_id, _, passage_id, _ = line.strip().split("\t")
                    # Ensure IDs are treated consistently (e.g., as strings)
                    qrels[str(query_id)].add(str(passage_id))
                except ValueError:
                    print(
                        f"Skipping malformed line {i+1} in qrels: {line.strip()}",
                        file=sys.stderr,
                    )
                    continue
    except FileNotFoundError:
        print(f"Error: Qrels file not found at {qrels_filepath}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded relevance info for {len(qrels)} queries.")

    # --- 2. Process Results and Calculate Reciprocal Ranks ---
    print(f"Processing results from {results_filepath}...")
    reciprocal_ranks_10 = {}  # Stores RR@10 for each query_id
    reciprocal_ranks_100 = {}  # Stores RR@100 for each query_id
    processed_queries = set()  # Keep track of unique queries encountered in results

    try:
        with open(results_filepath, "r", encoding="utf-8") as f:
            # Optional: Skip header if your results file has one
            # try:
            #     header = next(f)
            #     # You could potentially validate header columns here if needed
            # except StopIteration:
            #     print("Warning: Results file is empty.", file=sys.stderr)
            #     return 0.0, 0.0 # Or handle as appropriate

            for i, line in enumerate(f):
                try:
                    query_id, passage_id, rank_str, score_str = line.strip().split("\t")
                    rank = int(rank_str)
                    # Ensure IDs are treated consistently (e.g., as strings)
                    query_id = str(query_id)
                    passage_id = str(passage_id)

                except ValueError:
                    print(
                        f"Skipping malformed line {i+1} in results: {line.strip()}",
                        file=sys.stderr,
                    )
                    continue

                processed_queries.add(query_id)

                # Check if this passage is relevant for this query
                # Use .get() to handle queries in results that might not be in qrels
                relevant_passages_for_query = qrels.get(query_id, set())

                if passage_id in relevant_passages_for_query:
                    # --- Calculate RR@10 ---
                    # Record the RR only for the *first* relevant passage found <= 10
                    if rank <= 10 and query_id not in reciprocal_ranks_10:
                        reciprocal_ranks_10[query_id] = 1.0 / rank

                    # --- Calculate RR@100 ---
                    # Record the RR only for the *first* relevant passage found <= 100
                    if rank <= 100 and query_id not in reciprocal_ranks_100:
                        reciprocal_ranks_100[query_id] = 1.0 / rank

                # Optimization: If we've found the first relevant item for both cutoffs
                # for a query, and the file is strictly ordered by query_id then rank,
                # we *could* potentially skip ahead. However, the current logic is robust
                # even if ranks > 100 appear before ranks <= 100 for the same query (unlikely).
                # The check `query_id not in reciprocal_ranks_X` handles finding the *first*.

    except FileNotFoundError:
        print(f"Error: Results file not found at {results_filepath}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Calculate MRR ---
    num_queries = len(processed_queries)

    if num_queries == 0:
        print("Warning: No queries found in the results file.", file=sys.stderr)
        return 0.0, 0.0

    # Sum the reciprocal ranks. Queries with no relevant passage within the cutoff
    # will not be in the respective dictionary, effectively contributing 0 to the sum.
    total_rr_10 = sum(reciprocal_ranks_10.values())
    total_rr_100 = sum(reciprocal_ranks_100.values())

    mrr_10 = total_rr_10 / num_queries
    mrr_100 = total_rr_100 / num_queries

    print(f"Processed {num_queries} unique queries from results file.")
    return mrr_10, mrr_100


# --- Main Execution ---
if __name__ == "__main__":
    # Define file paths
    qrels_filepath = r"C:\\Users\\Manu Pande\\Documents\\thesis\\4thsem\\qrels.dev.tsv"
    results_filepath = r"C:\Users\Manu Pande\Documents\thesis\4thsem\sbert_finetuned_lora_1m_results.tsv"

    # Calculate MRR
    mrr_at_10, mrr_at_100 = calculate_mrr(qrels_filepath, results_filepath)

    print(f"\n--- Results ---")
    print(f"MRR@10:  {mrr_at_10:.4f}")
    print(f"MRR@100: {mrr_at_100:.4f}")
