from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer from the saved directory
MODEL_DIR = "./lunaris_finetuned"  # change this if your path is different
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

def generate_response(prompt, max_length=256, temperature=1.0, num_return_sequences=1):
    """
    Generates a response from the fine-tuned model given a prompt.
    
    Parameters:
        prompt (str): The input text prompt.
        max_length (int): Maximum length of the generated sequence.
        temperature (float): Sampling temperature.
        num_return_sequences (int): Number of output sequences to generate.
        
    Returns:
        list[str]: A list containing the generated response(s).
    """
    # Encode the prompt text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Move model and input_ids to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = input_ids.to(device)
    
    # Generate response(s)
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode the generated tokens to text
    responses = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    return responses

# A simple command-line interface for interactive testing.
if __name__ == "__main__":
    print("Interactive Lunaris-v1 Model")
    print("Enter 'quit' to exit.")
    while True:
        prompt = input("\nEnter your prompt: ")
        if prompt.lower().strip() == "quit":
            break
        responses = generate_response("user:" + prompt)
        print("\nGenerated response:")
        print(responses[0])
