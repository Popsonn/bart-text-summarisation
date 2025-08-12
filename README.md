# BART Text Summarization Fine-tuning

A fine-tuned BART model for generating concise summaries from longer text inputs. This project demonstrates how to fine-tune Facebook's BART-large-CNN model on custom data for text summarization tasks.

## 🚀 Features

- Fine-tune BART-large-CNN on custom datasets
- GPU-accelerated training and inference
- Configurable summary length (8-12 tokens)
- Batch processing for efficient inference
- CSV export for model evaluation

## 📋 Requirements

- torch>=1.9.0
- transformers>=4.21.0
- datasets>=2.0.0
- pandas>=1.3.0
- numpy>=1.21.0
- openpyxl>=3.0.9

## 🛠️ Installation

Clone the repository:
git clone https://github.com/yourusername/bart-text-summarization.git
cd bart-text-summarization

Install dependencies:
pip install -r requirements.txt


## 📊 Dataset

### Source Data

This project uses a processed version of the [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews/version/1) dataset from Kaggle.

### Processed Dataset

The provided `text_summarised.xlsx` file contains:

- `Processed_Text12`: Processed clothing review text (source text for summarization)
- `Processed_Title9`: Processed review titles/summaries (target summaries)

**Dataset Statistics:**

- Processed reviews from women's e-commerce clothing
- Text length: Up to 80 tokens (input)
- Summary length: Up to 12 tokens (target)

### Using Your Own Data

To use your own dataset, ensure your Excel file contains columns:

| Column Name         | Description                      |
|---------------------|---------------------------------|
| Processed_Text12    | Your source text here...         |
| Processed_Title9    | Target summary                   |

## 🔧 Usage

### Quick Start

Use the provided dataset:

The dataset is already included in the repository
path = 'data/text_summarised.xlsx' # Update path as needed


Or load your own data:

Load your custom data
import pandas as pd
df = pd.read_excel('your_custom_data.xlsx')


Run the training script:

python train_bart_summarizer.py


### Configuration

Key parameters you can modify:

Training parameters
num_train_epochs=1
per_device_train_batch_size=4
max_input_length=80
max_output_length=12

Generation parameters
max_length=12
min_length=8
length_penalty=2.0
num_beams=4


## 📈 Model Training

The script performs the following steps:

- **Data Loading:** Loads Excel data and removes invalid entries
- **Preprocessing:** Tokenizes input text and target summaries
- **Fine-tuning:** Trains BART model for 1 epoch
- **Evaluation:** Generates summaries for 100 random samples
- **Export:** Saves results to CSV for analysis

## 🎯 Results

After training on the clothing reviews dataset, the model generates summaries like:

| Original Review                                                                                      | Original Summary               | Generated Summary                   |
|----------------------------------------------------------------------------------------------------|-------------------------------|-----------------------------------|
| "I love this dress! The fit is perfect and the material is so comfortable. Great for both casual and dressy occasions. Would definitely recommend..." | "Perfect dress, comfortable material" | "Love dress perfect fit comfortable material" |

The model generates a CSV file (`random_summaries.csv`) containing:

- Original review text
- Original summaries
- Generated summaries

Use this for evaluation and comparison.

## 🔍 Example Output

- **Original Review:** "This sweater is amazing quality and fits perfectly. The color is exactly as shown and shipping was fast..."
- **Original Summary:** "Amazing quality sweater"
- **Generated Summary:** "Sweater amazing quality fits perfectly"

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the BART implementation
- [Facebook AI Research](https://github.com/pytorch/fairseq) for the original BART model
- BART paper: [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)


⭐ Star this repository if you find it helpful!
