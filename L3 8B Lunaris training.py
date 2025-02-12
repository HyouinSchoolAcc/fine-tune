import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import os
import torch
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# Enable NCCL debug logging for more info and try to disable some P2P features if needed
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Exclude GPU 2 by exposing only GPU 0 and GPU 1 (the high-performance NVIDIA H100 NVL GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_and_preprocess_data(json_file):
    """
    Loads conversation data from a JSON file.
    The file is assumed to be a dictionary mapping conversation IDs (e.g. URLs)
    to a list of messages, each with a "role" and "content" field.
    
    This function concatenates each conversation into a single string with
    role markers for each turn. For example:
    
        System: <content>
        User: <content>
        Assistant: <content>
    
    Adjust the formatting if needed.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    conversation_texts = []
    for conv_id, messages in data.items():
        conversation = ""
        for msg in messages:
            # Capitalize the role for consistency
            role = msg.get("role", "system").capitalize()
            content = msg.get("content", "")
            conversation += f"{role}: {content}\n"
        conversation_texts.append(conversation.strip())
    return conversation_texts

def tokenize_function(example, tokenizer, max_length=1024):
    outputs = tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)
    # Set the labels equal to the input_ids so that the model can compute LM loss.
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

def main():
    # Define paths and model identifier.
    json_file = "all_conversations.json"  # Path to your JSON training file
    model_id = "Sao10K/L3-8B-Lunaris-v1"  # Lunaris-v1 model merge based on Llama-3

    # Load tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Set a padding token if the tokenizer doesn't have one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as the padding token
    
    # Load and preprocess the conversation data.
    conversation_texts = load_and_preprocess_data(json_file)
    
    # Create a Hugging Face dataset from the conversation texts.
    # Each training example is stored under the key "text".
    dataset = Dataset.from_dict({"text": conversation_texts})
    
    # Tokenize the dataset.
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Define training arguments.
    training_args = TrainingArguments(
        output_dir="./lunaris_finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp8=True,               # Enable mixed precision if available
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="no"       # Use 'eval_strategy' instead of the deprecated 'evaluation_strategy'
    )
    
    # Create the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    # Start training.
    trainer.train()
    
    # Save the fine-tuned model.
    trainer.save_model("./lunaris_finetuned")
    print("Training complete. Model saved to './lunaris_finetuned'.")

if __name__ == "__main__":
    main()
