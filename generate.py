import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_poem(prompt: str, model_dir: str, max_length: int, temperature: float, top_p: float, top_k: int):
    """
    Generates a poem from a given prompt using a fine-tuned GPT-2 model.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load the tokenizer and model from the specified directory
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
        model.eval() # Set the model to evaluation mode

    except OSError:
        print(f"Error: Model or tokenizer not found at '{model_dir}'.")
        print("Please ensure you have trained the model and specified the correct path.")
        return

    # Encode the prompt, adding the <BOS> token for consistency with training
    # The space after <BOS> is important if the tokenizer was trained that way.
    inputs = tokenizer.encode("<BOS> " + prompt, return_tensors="pt").to(device)

    print("\n" + "="*20)
    print(f"Prompt: {prompt}")
    print("="*20)
    print("Generating poem...")
    print("...\n")

    # Generate text using the model's generate method
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            do_sample=True,             # Activates sampling-based generation
            max_length=max_length,      # Maximum length of the generated sequence
            temperature=temperature,    # Controls randomness: higher is more random
            top_p=top_p,                # Nucleus sampling: keeps the most probable tokens with cumulative probability p
            top_k=top_k,                # Keeps only the top k most likely tokens for sampling
            pad_token_id=tokenizer.eos_token_id, # Prevents warnings
            num_return_sequences=1
        )
    
    # Decode the output and skip special tokens for a clean result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a poem using a fine-tuned GPT-2 model.")
    
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to start the poem.")
    
    # Updated default path to match the output of the improved training script
    parser.add_argument("--model", type=str, default="models/poem-gpt2-best", help="Directory of the trained model and tokenizer.")
    
    # --- NEW: Command-line arguments for generation parameters ---
    parser.add_argument("--max_length", type=int, default=150, help="Maximum length of the generated poem.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Controls creativity. Higher values (e.g., 1.2) are more random, lower values (e.g., 0.7) are more deterministic.")
    parser.add_argument("--top_k", type=int, default=50, help="Filters the vocabulary to the top K most likely tokens at each step.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling parameter.")

    args = parser.parse_args()

    generate_poem(
        prompt=args.prompt,
        model_dir=args.model,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
