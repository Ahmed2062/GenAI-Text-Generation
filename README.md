# Poem Generator with GPT-2

This project scrapes poems from  https://poets.org/poems, cleans and preprocesses them, and then fine-tunes a GPT-2 model to generate new poems.

📂 Project Structure
poem-generator/

│
├── data/                   # datasets

│          └── poems.json          # raw scraped poems

│    └── cleaned.json        # cleaned & deduplicated poems

│
├── models/                 # trained models

│   └── poem-gpt2/          

│
├── src/                    # source scripts

│   └── scrape_poems.py     # scrape poems from poets.org

│   └── clean_poems.py      # clean & deduplicate poems

│   └── train_model.py      # fine-tune GPT-2

│   └── generate.py         # generate poems from a trained model

│
├── requirements.txt        # dependencies

└── README.md               # project documentation

⚙️ Installation

Clone the repository:

git clone https://github.com/Ahmed2062/poem-generator.git

cd poem-generator


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # On Linux/Mac

venv\Scripts\activate      # On Windows


Install dependencies:

    pip install -r requirements.txt

📝 Usage
1. Scrape Poems

Scrape poems from poets.org (default: 5 pages):

    python src/scrape_poems.py --pages 5 --output data/poems.json

2. Clean Poems

Remove duplicates and formatting issues:

    python src/clean_poems.py --input data/poems.json --output data/cleaned.json

3. Train Model

Fine-tune GPT-2 on the cleaned poems:

    python src/train_model.py --data data/cleaned.json --output models/poem-gpt2 --epochs 3 --batch_size 8

4. Generate Poems

Generate a poem from a trained model:

    python src/generate.py --prompt "A rainy night in Delhi" --model models/poem-gpt2

🎯 Example Output

Prompt:

A rainy night in Delhi


Generated Poem:

A rainy night in Delhi, the streets hum with shadows,  
Lanterns flicker in puddles where silence drifts slow.  
The air tastes of jasmine and thunder’s confession,  
As the city folds gently into a midnight of poems.  

⚡ Features

Scrapes real poems from poets.org

Cleans and deduplicates the dataset

Fine-tunes GPT-2 on poetry

CLI-based workflow (easy to reproduce)

Generates new, creative poems from prompts

📌 Requirements

Python 3.8+

GPU recommended (for training)

Install all dependencies via:

pip install -r requirements.txt

🤝 Contributing

Feel free to fork the repo, open issues, or submit pull requests to improve scraping, cleaning, or model training.

📜 License

MIT License – you’re free to use, modify, and share.
