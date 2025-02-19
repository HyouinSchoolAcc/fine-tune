import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up argument parsing
parser = argparse.ArgumentParser(description="Text Generation with Sao10K/L3-8B-Lunaris-v1 Model")
parser.add_argument('-p', '--prompt', type=str, help="Input text prompt for the model")
args = parser.parse_args()

# Determine the input prompt
if args.prompt:
    prompt = args.prompt
else:
    prompt = input("Please enter your prompt: ")

# Load the tokenizer and model
model_id = "Sao10K/L3-8B-Lunaris-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16  # Utilize float16 for efficient inference
).to("cuda")  # Deploy the model to the GPU

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate text
output = model.generate(
    **inputs,
    max_new_tokens=100,  # Limit the generation to 100 new tokens
    temperature=0.7,     # Adjust the randomness of predictions
    top_p=0.9,           # Implement nucleus sampling
    do_sample=True       # Enable sampling for diversity
)

# Decode and display the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
