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

1. Clone the repository:
```bash
git clone https://github.com/Popsonn/bart-text-summarization.git
cd bart-text-summarization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## 📊 Dataset

### Source Data

This project uses a processed version of the [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews/version/1) dataset from Kaggle.

### Processed Dataset

The provided `text_summarised.xlsx` file contains:

- `Processed_Review_Text`: Processed clothing review text (source text for summarization)
- `Processed_Title`: Processed review titles/summaries (target summaries)

**Dataset Statistics:**

- Processed reviews from women's e-commerce clothing
- Text length: Up to 80 tokens (input)
- Summary length: Up to 12 tokens (target)

### Sample Data
A sample of 100 randomly selected processed reviews is provided in `random_summaries.csv` to demonstrate the model's output quality and help with evaluation.

### Using Your Own Data

To use your own dataset, ensure your Excel file contains columns:

| Column Name         | Description                      |
|---------------------|---------------------------------|
| Processed_Review_Text  | Your source text here...         |
| Processed_Title   | Target summary                   |

## 🔧 Usage

### Quick Start

1. Use the provided dataset (pre-processed and ready for training):
```python
# The dataset is already included: text_summarised.xlsx
```

2. Run the training script:
```bash
python train_bart_summarizer.py
```

### Repository Structure
```
bart-text-summarization/
├── data/
│   ├── text_summarised.xlsx          # Full processed dataset
│   └── random_summaries.csv          # Sample results (100 entries)
├── train_bart_summarizer.py
├── requirements.txt
└── README.md
```

### Configuration

Key parameters you can modify in the training script:

```python
# Training parameters
num_train_epochs=1
per_device_train_batch_size=4
max_input_length=80
max_output_length=12

# Generation parameters
max_length=12
min_length=8
length_penalty=2.0
num_beams=4
```


## 📈 Model Training

The script performs the following steps:

- **Data Loading:** Loads Excel data and removes invalid entries
- **Preprocessing:** Tokenizes input text and target summaries
- **Fine-tuning:** Trains BART model for 1 epoch
- **Evaluation:** Generates summaries for 100 random samples
- **Export:** Saves results to CSV for analysis

## 🎯 Results

After training on the clothing reviews dataset, the model generates concise summaries from longer review text:

| Original Review                                                                                      | Original Summary               | Generated Summary                   |
|----------------------------------------------------------------------------------------------------|-------------------------------|-----------------------------------|
| "I love this dress! The fit is perfect and the material is so comfortable. Great for both casual and dressy occasions. Would definitely recommend..." | "Perfect dress, comfortable material" | "Love dress perfect fit comfortable material" |

### Output Files
- The training process generates a new CSV file with your results
- A sample file (`random_summaries.csv`) is included showing 100 examples with:
  - Original review text
  - Original summaries
  - Generated summaries

Use these files for evaluation and comparison of model performance.

## 🔍 Example Output

- **Original Review:** "This sweater is amazing quality and fits perfectly. The color is exactly as shown and shipping was fast..."
- **Original Summary:** "Amazing quality sweater"
- **Generated Summary:** "Sweater amazing quality fits perfectly"

## ⚙️ Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only training (slower)
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **GPU Support**: CUDA-compatible GPU for faster training

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the BART implementation
- [Facebook AI Research](https://github.com/pytorch/fairseq) for the original BART model
- BART paper: [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)


⭐ Star this repository if you find it helpful!
