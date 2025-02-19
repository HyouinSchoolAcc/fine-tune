import os
import json
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# ==========================
# Environment & Debugging
# ==========================

torch.cuda.empty_cache()

# Enable NCCL debug logs
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand issues
os.environ["NCCL_SHM_DISABLE"] = "1"  # Avoid shared memory issues
os.environ["NCCL_P2P_DISABLE"] = "0"  # Enable peer-to-peer GPU comms

# Use only GPUs 0 and 1 (your H100s)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Disable tokenizers parallelism to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Debug memory before model loads
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Before model loading:", torch.cuda.memory_allocated() / 1e9, "GB")

# ==========================
# Load Model & Tokenizer
# ==========================
model_id = "Sao10K/L3-8B-Lunaris-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16  # Match precision with DeepSpeed config
).to("cuda")

# Use EOS token as padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("After model loading:", torch.cuda.memory_allocated() / 1e9, "GB")

# ==========================
# Load & Process Dataset
# ==========================
def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    conversation_texts = []
    for conv_id, messages in data.items():
        conversation = "\n".join([f"{msg.get('role', 'system').capitalize()}: {msg.get('content', '')}" for msg in messages])
        conversation_texts.append(conversation.strip())
    
    return Dataset.from_dict({"text": conversation_texts})

json_file = "all_conversations.json"
dataset = load_data(json_file)

# def tokenize_function(example):
#     # Explicitly specify device mapping for tokenization
#     return tokenizer(
#         example["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=1024,
#         return_tensors="pt"  # Return PyTorch tensors directly
#     )

def tokenize_function(example):
    # Explicitly specify device mapping for tokenization
    encoding = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt"  # Return PyTorch tensors directly
    )
    encoding["labels"] = encoding["input_ids"].clone()
    return encoding

# Convert to PyTorch format and keep tensors on GPU
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    batch_size=4  # Reduce memory pressure
).with_format("torch", device="cuda")  # Keep dataset on GPU
# ==========================
# DeepSpeed Configuration
# ==========================
ds_config = {
  "fp16": {
    "enabled": True
  },
  "zero_optimization": {
    "stage": 2,
  },
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": "auto",
  "train_batch_size": "auto",
  "steps_per_print": 10
}

# Save DeepSpeed config to file
with open("ds_config.json", "w") as f:
    json.dump(ds_config, f, indent=4)

# ==========================
# Training Arguments
# ==========================
torch.cuda.empty_cache()

training_args = TrainingArguments(
    output_dir="./lunaris_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,  # Mixed precision
    optim="adamw_torch_fused",  # Faster fused AdamW
    evaluation_strategy="no",
    deepspeed="ds_config.json"  # Enable DeepSpeed
)

# ==========================
# Trainer & Training
# ==========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# ==========================
# Save Model
# ==========================
trainer.save_model("./lunaris_finetuned")
print("Training complete. Model saved to './lunaris_finetuned'.")
