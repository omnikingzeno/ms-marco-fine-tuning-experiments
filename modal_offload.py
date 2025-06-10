import modal

modal_volume = modal.Volume.from_name("4th-sem", create_if_missing=True)

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "wget")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
        "torch",
        "huggingface-hub==0.23.0",
        "transformers==4.41.0",
        "sentence-transformers==4.1.0",
        "peft>=0.5.0",
        "dask",
        "accelerate",
        "bitsandbytes",
        "einops",
        "datasets",
        "pandas",
        "numpy",
        "fastparquet",
        "pyarrow",
        "tqdm",
        "wandb",
        "scikit-learn",
        "hf_xet",
    )
).add_local_file(
    "C:\\Users\\Manu Pande\\Documents\\thesis\\4thsem\\visualizer.py",
    "/root/",
)


MOUNT_DIR = "/root/4th-sem"


app = modal.App(
    name="thesis-4thsem",
    image=image,
    volumes={MOUNT_DIR: modal_volume},
)


@app.function(
    # gpu="A100-80GB:2",
    timeout=72000,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_ms_marco():

    import os

    COLLECTION_URL = (
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
    )
    QUERIES_URL = (
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"
    )
    QRELS_DEV = "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv"
    TRAINS_TRIPLES_SMALL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.small.tar.gz"

    download_arr = [QUERIES_URL, QRELS_DEV, TRAINS_TRIPLES_SMALL]

    # os.system("mkdir /root/wget_downloads")
    # os.system("mkdir /root/4th-sem/datasets")
    # os.system("mkdir /root/4th-sem/wget_downloads")
    # for file_d in download_arr:
    #     os.system(f"wget {file_d} -P /root/4th-sem/wget_downloads")

    # extract the files if tar else cp them
    # for file_d in os.listdir("/root/4th-sem/wget_downloads"):
    # if file_d.endswith(".tar.gz"):
    os.system(
        f"tar -xvzf /root/4th-sem/wget_downloads/triples.train.small.tar.gz -C /root/4th-sem/datasets"
    )
    # else:
    #     os.system(
    #         f"cp /root/4th-sem/wget_downloads/{file_d} /root/4th-sem/datasets"
    #     )


@app.function(
    gpu="A100-80GB:1",
    timeout=72000,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={MOUNT_DIR: modal_volume},
)
def evaluate():
    """
    Evaluate the performance of a pretrained model on a dataset.

    This function loads a pretrained model, evaluates its performance on a dataset,
    and saves the evaluation results to a specified directory.

    Args:
        None
    """
    import os
    import json

    # Load the model and evaluate it
    model_name = "manupande21/all-MiniLM-L6-v2-finetuned-triples_hard_negatives"
    filtered_queries_path = "/root/4th-sem/datasets/filtered_queries.dev.tsv"
    collection_path = "/root/4th-sem/datasets/collection.tsv"
    # qrels_path = "/root/4th-sem/datasets/qrels.dev.tsv"
    output_dir = "/root/4th-sem/evaluation-results"
    cache_dir = "/root/4th-sem/cache/sbert_finetuned_hard_negatives"
    # embeddings_dir = "/root/4th-sem/embeddings"
    # index_dir = "/root/4th-sem/index"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # run the script
    os.system(
        f"python /root/eval_model.py --cache-dir {cache_dir} --model {model_name} --collection {collection_path} --filtered-queries {filtered_queries_path} --output-dir {output_dir} --batch-size 8192 --query-batch-size 256 --chunk-size 200000 --query-process-batch-size 128 --use-gpu --precision float16 --optimize-memory --fast-mode"
    )


@app.function(
    gpu="A100-80GB:1",
    timeout=72000,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={MOUNT_DIR: modal_volume},
)
def finetune():
    import os
    import json

    # os.system("pip install --upgrade peft transformers")
    os.system(f"python /root/finetune.py")


@app.function(
    gpu="A100-80GB:1",
    timeout=72000,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={MOUNT_DIR: modal_volume},
)
def train():
    """
    Train model on given dataset.

    Args:
        None
    """
    import os
    import json

    os.system("pip install dask")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    filtered_queries_path = "/root/4th-sem/datasets/filtered_queries.train.tsv"
    collection_path = "/root/4th-sem/datasets/collection.tsv"
    output_dir = "/root/4th-sem/evaluation-results/train_results"
    cache_dir = "/root/4th-sem/cache/sbert"
    # embeddings_dir = "/root/4th-sem/embeddings"
    # qrels_path = "/root/4th-sem/datasets/qrels.train.tsv"
    # index_dir = "/root/4th-sem/index"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # run the script
    os.system(
        f"python /root/process_train_queries.py --cache-dir {cache_dir} --model {model_name} --collection {collection_path} --filtered-queries {filtered_queries_path} --output-dir {output_dir} --batch-size 16384 --query-batch-size 2048 --chunk-size 200000 --query-process-batch-size 65536 --use-gpu --precision float16 --final-k 200 --optimize-memory --fast-mode --gpu-final-ranking --chunk-batches 30"
    )


@app.function(
    gpu="A100-80GB:1",
    timeout=72000,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={MOUNT_DIR: modal_volume},
)
def create_train_data():
    import os

    filtered_queries_path = "/root/4th-sem/datasets/filtered_queries.train.tsv"
    collection_path = "/root/4th-sem/datasets/collection.tsv"
    output_dir = "/root/4th-sem/datasets"
    qrels_path = "/root/4th-sem/datasets/sorted_qrels.train.tsv"
    random_samples_path = "/root/4th-sem/datasets/random_samples.tsv"
    output_file = "triples.train.hard_negatives.tsv"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # run the script
    os.system(
        f"python /root/train_data_creator.py --collection {collection_path} --filtered-queries {filtered_queries_path} --qrels {qrels_path} --random-samples {random_samples_path} --output-dir {output_dir} --output-file {output_file} --chunk-size 200000 "
    )


@app.function(
    gpu="A100-80GB:1",
    timeout=72000,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={MOUNT_DIR: modal_volume},
)
def evaluate_lora():
    """
    Evaluate the performance of a pretrained model on a dataset.

    This function loads a pretrained model, evaluates its performance on a dataset,
    and saves the evaluation results to a specified directory.

    Args:
        None
    """
    import os
    import json

    os.system("mkdir -p /root/model_debug")
    os.system(
        f"huggingface-cli download manupande21/all-MiniLM-L6-v2-LoRA-finetuned-1M --local-dir /root/model_debug"
    )
    adapter_config_path = "/root/model_debug/adapter_config.json"
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            config = json.load(f)

        # Remove problematic parameter if present
        problematic_params = [
            "corda_config",
            "eva_config",
            "exclude_modules",
            "lora_bias",
            "trainable_token_indices",
        ]
        for param in problematic_params:
            if param in config:
                del config[param]
        with open(adapter_config_path, "w") as f:
            json.dump(config, f, indent=2)
    # Load the model and evaluate it
    model_name = "/root/model_debug"
    filtered_queries_path = "/root/4th-sem/datasets/filtered_queries.dev.tsv"
    collection_path = "/root/4th-sem/datasets/collection.tsv"
    # qrels_path = "/root/4th-sem/datasets/qrels.dev.tsv"
    output_dir = "/root/4th-sem/evaluation-results"
    cache_dir = "/root/4th-sem/cache/sbert_finetuned_lora_1M"
    # embeddings_dir = "/root/4th-sem/embeddings"
    # index_dir = "/root/4th-sem/index"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # run the script
    os.system(
        f"python /root/eval_model.py --cache-dir {cache_dir} --model {model_name} --collection {collection_path} --filtered-queries {filtered_queries_path} --output-dir {output_dir} --batch-size 2048 --query-batch-size 128 --chunk-size 100000 --query-process-batch-size 64 --use-gpu --precision float16 --optimize-memory --fast-mode"
    )


@app.function(
    gpu="A100-80GB:1",
    timeout=72000,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={MOUNT_DIR: modal_volume},
)
def inference_times():
    """
    Find inference times for a pretrained model on a dataset.
    """
    import os
    import json

    # Load the model and evaluate it
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    filtered_queries_path = "/root/4th-sem/datasets/queries.eval.tsv"
    collection_path = "/root/4th-sem/datasets/collection.tsv"
    # qrels_path = "/root/4th-sem/datasets/qrels.dev.tsv"
    output_dir = "/root/4th-sem/evaluation-results"
    cache_dir = "/root/4th-sem/cache/sbert"
    # embeddings_dir = "/root/4th-sem/embeddings"
    # index_dir = "/root/4th-sem/index"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # run the script
    os.system(
        f"python /root/inference_times.py --cache-dir {cache_dir} --model {model_name} --collection {collection_path} --filtered-queries {filtered_queries_path} --output-dir {output_dir} --batch-size 8192 --query-batch-size 256 --chunk-size 200000 --query-process-batch-size 128 --use-gpu --precision float16 --optimize-memory --fast-mode"
    )


@app.function(
    gpu="A100-80GB:1",
    timeout=72000,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={MOUNT_DIR: modal_volume},
)
def inference_times_lora():
    """
    Find inference times for a LoRA finetuned model on a dataset.
    """
    import os
    import json

    os.system("mkdir -p /root/model_debug")
    os.system(
        f"huggingface-cli download manupande21/all-MiniLM-L6-v2-LoRA-finetuned-1M --local-dir /root/model_debug"
    )
    # os.system("ls -la /root/model_debug")
    adapter_config_path = "/root/model_debug/adapter_config.json"
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            config = json.load(f)

        # Remove problematic parameter if present
        problematic_params = [
            "corda_config",
            "eva_config",
            "exclude_modules",
            "lora_bias",
            "trainable_token_indices",
        ]
        for param in problematic_params:
            if param in config:
                del config[param]
        with open(adapter_config_path, "w") as f:
            json.dump(config, f, indent=2)
    # Load the model and evaluate it
    model_name = "/root/model_debug"
    model_name_for_output = "all-MiniLM-L6-v2-LoRA-finetuned-1M"
    filtered_queries_path = "/root/4th-sem/datasets/queries.eval.tsv"
    collection_path = "/root/4th-sem/datasets/collection.tsv"
    # qrels_path = "/root/4th-sem/datasets/qrels.dev.tsv"
    output_dir = "/root/4th-sem/evaluation-results"
    cache_dir = "/root/4th-sem/cache/sbert_finetuned_lora_1M"
    # embeddings_dir = "/root/4th-sem/embeddings"
    # index_dir = "/root/4th-sem/index"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # run the script
    os.system(
        f"python /root/inference_times_lora.py --cache-dir {cache_dir} --model {model_name} --collection {collection_path} --filtered-queries {filtered_queries_path} --output-dir {output_dir} --batch-size 8192 --query-batch-size 256 --chunk-size 100000 --query-process-batch-size 128 --use-gpu --precision float16 --optimize-memory --fast-mode --output-name {model_name_for_output}"
    )


@app.function(
    gpu="A100-80GB:1",
    timeout=7200,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={MOUNT_DIR: modal_volume},
)
def generate_embedding_visualization():
    """
    Generate 2D embedding space visualization for queries and passages.
    """
    import os
    import json

    # Install UMAP and matplotlib if needed
    os.system("pip install umap-learn matplotlib")

    # Define parameters
    input_file = "/root/4th-sem/datasets/visualization_data.tsv"
    output_dir = "/root/4th-sem/visualizations"

    # Regular models - can be loaded directly
    regular_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "manupande21/all-MiniLM-L6-v2-finetuned-triplets",
        "manupande21/all-MiniLM-L6-v2-finetuned-triples_hard_negatives",
    ]

    # LoRA models - need special handling
    lora_models = {
        "manupande21/all-MiniLM-L6-v2-LoRA-finetuned-1M": "/root/lora_model_1m",
        "manupande21/all-MiniLM-L6-v2-LoRA-finetuned-triplets_hard_negatives": "/root/lora_model_hn",
    }

    # Download and fix LoRA models
    for model_name, local_dir in lora_models.items():
        os.system(f"mkdir -p {local_dir}")
        os.system(f"huggingface-cli download {model_name} --local-dir {local_dir}")

        # Fix adapter_config.json
        adapter_config_path = f"{local_dir}/adapter_config.json"
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                config = json.load(f)

            # Remove problematic parameters
            problematic_params = [
                "corda_config",
                "eva_config",
                "exclude_modules",
                "lora_bias",
                "trainable_token_indices",
            ]
            for param in problematic_params:
                if param in config:
                    del config[param]

            with open(adapter_config_path, "w") as f:
                json.dump(config, f, indent=2)

    # Combined model list - use local paths for LoRA models
    all_models = regular_models + list(lora_models.values())

    # Format models for command line
    models_arg = " ".join([f'"{model}"' for model in all_models])

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run the visualization script with all models
    os.system(
        f"python /root/visualizer.py --input-file {input_file} --output-dir {output_dir} --use-gpu --models {models_arg}"
    )


@app.local_entrypoint()
def main():
    """
    Main entry point for the evaluation script.

    This function defines a list of pretrained models to be evaluated and calls the evaluate function
    to perform the evaluation.

    Args:
        None
    """
    # download_models_huggingface.remote()
    # infer.remote()
    # evaluate.remote()
    # evaluate_lora.remote()
    # finetune.remote()
    # download_ms_marco.remote()
    # train.remote()
    # create_train_data.remote()
    # inference_times.remote()
    # inference_times_lora.remote()
    generate_embedding_visualization.remote()
