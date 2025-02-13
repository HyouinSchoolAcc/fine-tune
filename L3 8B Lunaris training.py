import os
import json
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# ==========================
# Environment & Debugging
# ==========================

# Enable NCCL debug logs
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand issues
os.environ["NCCL_SHM_DISABLE"] = "1"  # Avoid shared memory issues
os.environ["NCCL_P2P_DISABLE"] = "0"  # Enable peer-to-peer GPU comms

# Use only GPUs 0 and 1 (your H100s)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
model = AutoModelForCausalLM.from_pretrained(model_id)

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

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ==========================
# DeepSpeed Configuration
# ==========================
ds_config = {
    "train_batch_size": 2,  # Smaller batch size for memory efficiency
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,  # Offloads optimizer states to CPU
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "optimizer": {
        "type": "adamw",
        "params": {
            "lr": 2e-5,
            "weight_decay": 0.01
        }
    },
    "gradient_clipping": 1.0,  # Prevent gradient explosion
    "steps_per_print": 10
}

# Save DeepSpeed config to file
with open("ds_config.json", "w") as f:
    json.dump(ds_config, f, indent=4)

# ==========================
# Training Arguments
# ==========================
training_args = TrainingArguments(
    output_dir="./lunaris_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
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
