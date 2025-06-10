import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer
import umap
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2D embeddings visualization")
    parser.add_argument(
        "--input-file",
        type=str,
        default="/root/4th-sem/datasets/visualization_data.tsv",
        help="Path to visualization_data.tsv file",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "manupande21/all-MiniLM-L6-v2-finetuned-triplets",
            "manupande21/all-MiniLM-L6-v2-finetuned-triples_hard_negatives",
        ],
        help="HuggingFace model(s) to use for embeddings",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/4th-sem/visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for embedding generation if available",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for UMAP for reproducibility",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter",
    )
    return parser.parse_args()


def generate_visualization(model_name, df, args, device):
    # Clean model name for filenames
    model_filename = model_name.replace("/", "_").replace("-", "_")

    # Load the model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    # Extract text data
    queries = df["query"].tolist()
    passages = df["passage"].tolist()

    # Generate embeddings
    print("Generating embeddings for queries...")
    query_embeddings = model.encode(
        queries, show_progress_bar=True, convert_to_tensor=True
    )

    print("Generating embeddings for passages...")
    passage_embeddings = model.encode(
        passages, show_progress_bar=True, convert_to_tensor=True
    )

    # Convert to numpy for UMAP
    query_embeddings_np = query_embeddings.cpu().numpy()
    passage_embeddings_np = passage_embeddings.cpu().numpy()

    # Combine all embeddings
    all_embeddings = np.vstack([query_embeddings_np, passage_embeddings_np])

    # Create labels (0 for queries, 1 for passages)
    labels = np.array([0] * len(queries) + [1] * len(passages))

    # Apply UMAP
    print(f"Applying UMAP dimensionality reduction for {model_name}...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )
    embedding = reducer.fit_transform(all_embeddings)

    # Create the visualization
    plt.figure(figsize=(12, 10))

    # Create a scatter plot
    plt.scatter(
        embedding[labels == 0, 0],
        embedding[labels == 0, 1],
        c="blue",
        s=10,
        label="Queries",
        alpha=0.7,
    )
    plt.scatter(
        embedding[labels == 1, 0],
        embedding[labels == 1, 1],
        c="orange",
        s=10,
        label="Passages",
        alpha=0.7,
    )

    plt.title(f"UMAP Visualization of Query-Passage Embeddings\nModel: {model_name}")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()

    # Save the visualization
    output_path = os.path.join(
        args.output_dir, f"umap_visualization_{model_filename}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")

    # Optional: save the embedding data
    np.save(
        os.path.join(args.output_dir, f"umap_embeddings_{model_filename}.npy"),
        embedding,
    )
    np.save(os.path.join(args.output_dir, f"umap_labels_{model_filename}.npy"), labels)

    # Close the plot to free memory
    plt.close()

    # Clear GPU memory
    del model
    del query_embeddings
    del passage_embeddings
    if device == "cuda":
        torch.cuda.empty_cache()

    return embedding, labels


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    print(f"Using device: {device}")

    # Load the data
    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file, sep="\t", header=None, names=["query", "passage"])

    print(f"Loaded {len(df)} query-passage pairs")

    # Process each model
    results = {}
    for model_name in args.models:
        print(f"\nProcessing model: {model_name}")
        embedding, labels = generate_visualization(model_name, df, args, device)
        results[model_name] = (embedding, labels)

    # If multiple models are provided, create a comparison visualization
    if len(args.models) > 1:
        print("\nCreating comparison visualization...")
        fig, axes = plt.subplots(1, len(args.models), figsize=(7 * len(args.models), 6))

        for i, model_name in enumerate(args.models):
            embedding, labels = results[model_name]

            # Clean model name for display
            display_name = model_name.split("/")[-1]

            ax = axes[i] if len(args.models) > 1 else axes

            # Plot queries
            ax.scatter(
                embedding[labels == 0, 0],
                embedding[labels == 0, 1],
                c="blue",
                s=10,
                label="Queries",
                alpha=0.7,
            )

            # Plot passages
            ax.scatter(
                embedding[labels == 1, 0],
                embedding[labels == 1, 1],
                c="orange",
                s=10,
                label="Passages",
                alpha=0.7,
            )

            ax.set_title(display_name)
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")

            # Only add legend to the first plot to avoid repetition
            if i == 0:
                ax.legend()

        plt.tight_layout()
        comparison_path = os.path.join(args.output_dir, "umap_model_comparison2.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        print(f"Comparison visualization saved to {comparison_path}")

    print("Done!")


if __name__ == "__main__":
    main()
