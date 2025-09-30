import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_poem(prompt, model_dir, max_length=200, temperature=1.0, top_p=0.95, top_k=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    model.eval()

    inputs = tokenizer.encode("<BOS> " + prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        do_sample=True,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for poem generation")
    parser.add_argument("--model", type=str, default="models/poem-gpt2", help="Trained model directory")
    args = parser.parse_args()

    print(generate_poem(args.prompt, args.model))
