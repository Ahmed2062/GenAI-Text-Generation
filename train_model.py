import argparse
import json
import torch
import time, datetime, os
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup

class PoemDataset(Dataset):
    def __init__(self, poems, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.input_ids, self.attn_masks = [], []
        for poem in poems:
            encodings = tokenizer("<BOS>"+poem["poem"]+"<EOS>",
                                  truncation=True,
                                  max_length=max_length,
                                  padding="max_length",
                                  return_tensors="pt")
            self.input_ids.append(encodings["input_ids"].squeeze(0))
            self.attn_masks.append(encodings["attention_mask"].squeeze(0))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def train(data_file, output_dir, epochs=3, batch_size=8, lr=3e-4):
    with open(data_file, "r", encoding="utf-8") as f:
        poems = json.load(f)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", bos_token="<BOS>", eos_token="<EOS>", pad_token="<PAD>")
    dataset = PoemDataset(poems, tokenizer)

    train_size = int(0.85*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader)*epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_masks = [x.to(device) for x in batch]
            model.zero_grad()
            outputs = model(b_input_ids, labels=b_input_ids, attention_mask=b_masks)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_train_loss / len(train_dataloader)
        print(f"  Training Loss: {avg_loss:.3f} | Time: {format_time(time.time()-t0)}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/cleaned.json", help="Input cleaned data file")
    parser.add_argument("--output", type=str, default="models/poem-gpt2", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    train(args.data, args.output, args.epochs, args.batch_size, args.lr)
