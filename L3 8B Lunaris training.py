import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

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
    # Tokenizes a single example, truncating to max_length tokens.
    return tokenizer(example["text"], truncation=True, max_length=max_length)

def main():
    # Define paths and model identifier.
    json_file = "all_conversations.json"  # Path to your JSON training file
    model_id = "Sao10K/L3-8B-Lunaris-v1"  # Lunaris-v1 model merge based on Llama-3

    # Load tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
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
        fp16=True,               # Enable mixed precision if available
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="no"  # Change to "steps" or "epoch" to add evaluation
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
