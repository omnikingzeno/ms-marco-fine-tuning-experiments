# import logging
# import math
# import os
# from datetime import datetime

# import pandas as pd
# import torch
# import wandb
# from huggingface_hub import HfApi
# from sentence_transformers import (
#     InputExample,
#     SentenceTransformer,
#     losses,
#     models,
# )
# from sentence_transformers.evaluation import TripletEvaluator
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader

# # --- Configuration ---
# dataset_path = "/root/4th-sem/datasets/sampled_training_data_2M.tsv"
# # Base model identifier from Hugging Face Hub
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# # Your Hugging Face username (replace!)
# hf_username = "manupande21"
# new_model_name = (
#     "all-MiniLM-L6-v2-finetuned-triplets_hard_negatives"
# )
# hf_repo_id = f"{hf_username}/{new_model_name}"
# # Local path to save the final model and checkpoints
# output_path = (
#     f"/root/4th-sem/{new_model_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
# )
# # WandB project name
# wandb_project_name = "sentence-transformer-finetune"

# # Training parameters
# train_batch_size = 1024
# eval_batch_size = 1024
# num_epochs = 5
# max_seq_length = 256  # Max sequence length for the transformer
# warmup_steps_ratio = 0.1  # 10% of training steps for warmup
# evaluation_steps = 1000  # Evaluate every N steps
# checkpoint_save_steps = 2000  # Save checkpoint every N steps
# learning_rate = 2e-5
# train_test_split_ratio = 0.2  # 20% for testing, 80% for training
# random_seed = 42  # For reproducible splits

# # --- Setup Logging ---
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=logging.INFO,
# )
# logger = logging.getLogger(__name__)

# # --- Initialize WandB ---
# try:
#     wandb.init(
#         project=wandb_project_name,
#         config={
#             "model_name": model_name,
#             "dataset_path": dataset_path,
#             "output_path": output_path,
#             "epochs": num_epochs,
#             "train_batch_size": train_batch_size,
#             "eval_batch_size": eval_batch_size,
#             "learning_rate": learning_rate,
#             "warmup_steps_ratio": warmup_steps_ratio,
#             "max_seq_length": max_seq_length,
#             "loss": "TripletLoss",
#         },
#     )
#     wandb_is_available = True
#     logger.info("Weights & Biases initialized successfully.")
# except Exception as e:
#     logger.warning(
#         f"Could not initialize Weights & Biases: {e}. Training will proceed without WandB."
#     )
#     wandb_is_available = False


# # --- Load Dataset ---
# logger.info(f"Loading dataset from: {dataset_path}")
# try:
#     # Read TSV without header, naming columns 0, 1, 2
#     df = pd.read_csv(dataset_path, sep="\t", header=None, on_bad_lines="skip")
#     # Check if we have exactly 3 columns as expected
#     if df.shape[1] != 3:
#         raise ValueError(
#             f"Expected 3 columns (query, positive, negative), but found {df.shape[1]}"
#         )
#     logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
# except FileNotFoundError:
#     logger.error(f"Dataset file not found at {dataset_path}. Please check the path.")
#     exit()
# except Exception as e:
#     logger.error(f"Error loading or parsing dataset: {e}")
#     exit()

# # --- Prepare InputExamples ---
# logger.info("Preparing InputExamples...")
# all_examples = []
# for index, row in df.iterrows():
#     # Ensure data are strings
#     query = str(row[0])
#     positive = str(row[1])
#     negative = str(row[2])
#     # Create InputExample for triplet loss
#     all_examples.append(InputExample(texts=[query, positive, negative]))

# logger.info(f"Created {len(all_examples)} InputExamples.")

# # --- Split Data ---
# logger.info(
#     f"Splitting data into train ({1-train_test_split_ratio:.0%}) and test ({train_test_split_ratio:.0%}) sets..."
# )
# train_examples, test_examples = train_test_split(
#     all_examples,
#     test_size=train_test_split_ratio,
#     random_state=random_seed,
#     shuffle=True,
# )
# logger.info(f"Train set size: {len(train_examples)}")
# logger.info(f"Test set size: {len(test_examples)}")

# # --- Load Model ---
# logger.info(f"Loading base model: {model_name}")
# # We need WordEmbedding model to set max_seq_length
# word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# logger.info(f"Base model {model_name} loaded.")

# # --- Prepare Training ---
# logger.info("Setting up training components...")
# # Triplet loss
# train_loss = losses.TripletLoss(model=model)

# # Data loader
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)

# logger.info("Preparing evaluation data...")
# eval_anchors = [ex.texts[0] for ex in test_examples]
# eval_positives = [ex.texts[1] for ex in test_examples]
# eval_negatives = [ex.texts[2] for ex in test_examples]

# evaluator = TripletEvaluator(
#     anchors=eval_anchors,
#     positives=eval_positives,
#     negatives=eval_negatives,
#     name="test-eval",
#     batch_size=eval_batch_size,
#     show_progress_bar=True,
# )

# # Calculate warmup steps
# num_training_steps = len(train_dataloader) * num_epochs
# warmup_steps = math.ceil(num_training_steps * warmup_steps_ratio)
# logger.info(f"Total training steps: {num_training_steps}")
# logger.info(f"Warmup steps: {warmup_steps}")

# # Checkpoint path
# checkpoint_path = os.path.join(output_path, "checkpoints")
# os.makedirs(checkpoint_path, exist_ok=True)

# # --- Fine-tune the model ---
# logger.info("Starting model fine-tuning...")
# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     evaluator=evaluator,
#     epochs=num_epochs,
#     evaluation_steps=evaluation_steps,
#     warmup_steps=warmup_steps,
#     output_path=output_path,  # Saves the final best model here
#     checkpoint_path=checkpoint_path,  # Saves intermediate checkpoints here
#     checkpoint_save_steps=checkpoint_save_steps,
#     checkpoint_save_total_limit=3,  # Keep only the last 3 checkpoints
#     optimizer_params={"lr": learning_rate},
#     use_amp=True,  # Use Automatic Mixed Precision for faster training (requires PyTorch >= 1.6)
#     # WandB integration is often handled automatically if wandb.init was successful
#     # Alternatively, you might need a callback depending on the sentence-transformers version
#     # from sentence_transformers.callbacks import WandbCallback
#     # callback=WandbCallback(project=wandb_project_name) if wandb_is_available else None
# )
# logger.info(f"Training finished. Best model saved to: {output_path}")


# # --- Save and Push to Hub---
# # The best model is already saved by model.fit to output_path
# # We just need to push it

# logger.info(f"Pushing final model to Hugging Face Hub repository: {hf_repo_id}")
# try:
#     # Ensure the output path contains the final model files
#     if os.path.exists(output_path):
#         # Load the best model saved by .fit() before pushing
#         final_model = SentenceTransformer(output_path)
#         # You can add a model card description
#         final_model.push_to_hub(
#             repo_id=hf_repo_id,
#             commit_message="Fine-tuned all-MiniLM-L6-v2 with triplet loss on custom dataset.",
#             private=False,  # Set to True if you want a private repo
#         )
#         logger.info(f"Model successfully pushed to {hf_repo_id}")
#     else:
#         logger.error(f"Output path {output_path} not found. Cannot push model.")

# except Exception as e:
#     logger.error(f"Error pushing model to Hugging Face Hub: {e}")
#     logger.warning("Model was saved locally but not pushed to the Hub.")


# # --- Finish WandB Run ---
# if wandb_is_available:
#     wandb.finish()
#     logger.info("Weights & Biases run finished.")

# logger.info("Script finished.")


import logging
import math
import os
from datetime import datetime

import pandas as pd
import torch
import wandb
from huggingface_hub import HfApi
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
    models,
)
from sentence_transformers.evaluation import TripletEvaluator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Add LoRA imports
from peft import LoraConfig, TaskType, get_peft_model

# --- Configuration ---
dataset_path = "/root/4th-sem/datasets/sampled_training_data.tsv"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_username = "manupande21"
new_model_name = "all-MiniLM-L6-v2-LoRA-finetuned-1M"
hf_repo_id = f"{hf_username}/{new_model_name}"
output_path = (
    f"/root/4th-sem/{new_model_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
)
wandb_project_name = "sentence-transformer-finetune"

# LoRA params
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

# Training parameters (adjusted for A100 80GB)
train_batch_size = 512
eval_batch_size = 512
num_epochs = 3
max_seq_length = 512
warmup_steps_ratio = 0.1
evaluation_steps = 1000
checkpoint_save_steps = 2000
learning_rate = 2e-5
train_test_split_ratio = 0.2
random_seed = 42

# --- Setup Logging ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Initialize WandB ---
try:
    wandb.init(
        project=wandb_project_name,
        config={
            "model_name": model_name,
            "dataset_path": dataset_path,
            "output_path": output_path,
            "epochs": num_epochs,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "learning_rate": learning_rate,
            "warmup_steps_ratio": warmup_steps_ratio,
            "max_seq_length": max_seq_length,
            "loss": "TripletLoss",
            "LoRA": True,
        },
    )
    wandb_is_available = True
    logger.info("Weights & Biases initialized successfully.")
except Exception as e:
    logger.warning(
        f"Could not initialize Weights & Biases: {e}. Training will proceed without WandB."
    )
    wandb_is_available = False

# --- Load Dataset ---
logger.info(f"Loading dataset from: {dataset_path}")
try:
    df = pd.read_csv(dataset_path, sep="\t", header=None, on_bad_lines="skip")
    if df.shape[1] != 3:
        raise ValueError(
            f"Expected 3 columns (query, positive, negative), but found {df.shape[1]}"
        )
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    logger.error(f"Dataset file not found at {dataset_path}. Please check the path.")
    exit()
except Exception as e:
    logger.error(f"Error loading or parsing dataset: {e}")
    exit()

# --- Prepare InputExamples ---
logger.info("Preparing InputExamples...")
all_examples = []
for index, row in df.iterrows():
    query = str(row[0])
    positive = str(row[1])
    negative = str(row[2])
    all_examples.append(InputExample(texts=[query, positive, negative]))

logger.info(f"Created {len(all_examples)} InputExamples.")

# --- Split Data ---
logger.info(
    f"Splitting data into train ({1-train_test_split_ratio:.0%}) and test ({train_test_split_ratio:.0%}) sets..."
)
train_examples, test_examples = train_test_split(
    all_examples,
    test_size=train_test_split_ratio,
    random_state=random_seed,
    shuffle=True,
)
logger.info(f"Train set size: {len(train_examples)}")
logger.info(f"Test set size: {len(test_examples)}")

# --- Load Base Model & Apply LoRA ---
logger.info(f"Loading base model: {model_name}")
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply LoRA to the underlying model
transformer_encoder = word_embedding_model.auto_model
peft_model = get_peft_model(transformer_encoder, lora_config)
word_embedding_model.auto_model = peft_model

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
logger.info(f"Base model {model_name} with LoRA modifications loaded.")

# --- Prepare Training ---
logger.info("Setting up training components...")
train_loss = losses.TripletLoss(model=model)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)

logger.info("Preparing evaluation data...")
eval_anchors = [ex.texts[0] for ex in test_examples]
eval_positives = [ex.texts[1] for ex in test_examples]
eval_negatives = [ex.texts[2] for ex in test_examples]

evaluator = TripletEvaluator(
    anchors=eval_anchors,
    positives=eval_positives,
    negatives=eval_negatives,
    name="test-eval",
    batch_size=eval_batch_size,
    show_progress_bar=True,
)

num_training_steps = len(train_dataloader) * num_epochs
warmup_steps = math.ceil(num_training_steps * warmup_steps_ratio)
logger.info(f"Total training steps: {num_training_steps}")
logger.info(f"Warmup steps: {warmup_steps}")

checkpoint_path = os.path.join(output_path, "checkpoints")
os.makedirs(checkpoint_path, exist_ok=True)

# --- Fine-tune the Model with LoRA ---
logger.info("Starting model fine-tuning (LoRA)...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=evaluation_steps,
    warmup_steps=warmup_steps,
    output_path=output_path,
    checkpoint_path=checkpoint_path,
    checkpoint_save_steps=checkpoint_save_steps,
    checkpoint_save_total_limit=3,
    optimizer_params={"lr": learning_rate},
    use_amp=True,
)

logger.info(f"Training finished. Best model saved to: {output_path}")

# --- Save and Push to Hub ---
logger.info(f"Pushing final model to Hugging Face Hub repository: {hf_repo_id}")
try:
    if os.path.exists(output_path):
        final_model = SentenceTransformer(output_path)
        final_model.push_to_hub(
            repo_id=hf_repo_id,
            commit_message="LoRA fine-tuned all-MiniLM-L6-v2 with triplet loss on custom dataset.",
            private=False,
        )
        logger.info(f"Model successfully pushed to {hf_repo_id}")
    else:
        logger.error(f"Output path {output_path} not found. Cannot push model.")
except Exception as e:
    logger.error(f"Error pushing model to Hugging Face Hub: {e}")
    logger.warning("Model was saved locally but not pushed to the Hub.")

if wandb_is_available:
    wandb.finish()
    logger.info("Weights & Biases run finished.")

logger.info("Script finished.")
