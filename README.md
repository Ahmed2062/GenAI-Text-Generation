# Poem Generator with GPT-2

This project scrapes poems from https://poets.org/poems, cleans and preprocesses them, and then fine-tunes a **GPT-2 (124M)** model to generate new poems based on a user prompt.

---

### Project Structure
```
poem-generator/
│
├── data/              # Datasets
│   ├── poems.json     # Raw scraped poems
│   └── cleaned.json   # Cleaned & deduplicated poems
│
├── models/            # Trained models
│   └── poem-gpt2-best # Fine-tuned model and tokenizer
│
├── src/               # Source scripts
│   ├── scrape_poems.py
│   ├── clean_poems.py
│   ├── train_model.py
│   └── generate.py
│
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

---

### Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/Ahmed2062/poem-generator.git](https://github.com/Ahmed2062/poem-generator.git)
cd poem-generator
```

**2. Create and activate a virtual environment (recommended):**
```bash
# Create the environment
python -m venv venv

# Activate it
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

### Usage

You can either generate poems immediately with our pre-trained model or reproduce the entire training process from scratch.

#### Option 1: Quick Start (Generate Poems Now)

A pre-trained model is available in the `models/poem-gpt2-best` directory. You can use it directly to generate poems.

```bash
python src/generate.py --prompt "A rainy night in Delhi"
```

To get more creative or varied results, try experimenting with the parameters:

```bash
# Generate a shorter, more focused poem
python src/generate.py --prompt "The first sign of spring" --max_length 75 --temperature 0.8

# Generate a longer, more unpredictable poem
python src/generate.py --prompt "A forgotten dream" --max_length 200 --temperature 1.2
```

#### Option 2: Reproduce from Scratch

Follow these steps to scrape the data and train the model yourself.

**1. Scrape Poems**
This command scrapes the first 5 pages from poets.org.
```bash
python src/scrape_poems.py --pages 5 --output data/poems.json
```

**2. Clean Poems**
This removes duplicates and cleans up formatting issues.
```bash
python src/clean_poems.py --input data/poems.json --output data/cleaned.json
```

**3. Train Model**
This fine-tunes the base GPT-2 model on the cleaned poems.
* **Note:** Training requires a GPU for a reasonable runtime. On an NVIDIA RTX 3060, this process takes approximately 45-60 minutes.
```bash
python src/train_model.py --data data/cleaned.json --output models/poem-gpt2-best --epochs 3 --batch_size 8
```

**4. Generate from Your Trained Model**
Once training is complete, you can generate poems using your own model.
```bash
python src/generate.py --prompt "The view from my window" --model models/poem-gpt2-best
```
---

### Example Output

**Prompt:** `A rainy night in Delhi`

**Generated Poem:**
> A rainy night in Delhi, the streets hum with shadows,
> Neon lights flicker in puddles where slick auto-rickshaws drift slow.
> The air tastes of jasmine and thunder's confession,
> As the city folds gently into a midnight of poems.

---

### Requirements

* **Python 3.9+** (Developed and tested with Python 3.9.7)
* **Hardware:** A GPU with at least 6GB of VRAM is highly recommended for training.

Contributing

Feel free to fork the repo, open issues, or submit pull requests to improve scraping, cleaning, or model training.

License

MIT License – you’re free to use, modify, and share.
