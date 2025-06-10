import pandas as pd


def find_passage_faster(collection_path, target_passage_id):
    """Find a passage by ID in a large collection file."""
    target_passage_id_str = str(target_passage_id)  # Convert target ID to string

    # Process in chunks to handle large files
    try:
        for chunk in pd.read_csv(
            collection_path,
            sep="\t",
            header=None,
            names=[0, 1],
            chunksize=100000,
            dtype={0: str},  # Read the first column explicitly as string
        ):
            # Filter using string comparison
            result = chunk[chunk[0] == target_passage_id_str]
            if not result.empty:
                return result.iloc[0][1]  # Return the passage text (column 1)
    except FileNotFoundError:
        print(f"Error: File not found at {collection_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return None


# Try the fixed method
search_id = 4339068  # Define the ID to search for
passage_text = find_passage_faster(
    r"C:\Users\Manu Pande\Documents\thesis\4thsem\collection\collection.tsv", search_id
)
if passage_text:
    print("\nPassage text:")
    print("-" * 80)
    print(passage_text)
    print("-" * 80)
else:
    # Use the actual search_id in the message
    print(f"No passage found with ID {search_id}")
