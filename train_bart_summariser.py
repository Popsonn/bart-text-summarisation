from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import numpy as np  
import pandas as pd
import torch
import random
import os

# Load model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Load and prepare data
path = '/kaggle/input/text-summarised/text summarised.xlsx'
df = pd.read_excel(path)
df = df.dropna()

# Remove rows with non-string values
df = df[df['Processed_Title9'].apply(lambda x: isinstance(x, str))].reset_index(drop=True)

titles = df['Processed_Text12'].tolist()
summaries = df['Processed_Title9'].tolist()

def preprocess_data(titles, summaries):
    inputs = tokenizer(titles, max_length=80, truncation=True, padding="max_length", return_tensors="pt")
    targets = tokenizer(summaries, max_length=12, truncation=True, padding="max_length", return_tensors="pt")
    
    labels = targets["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": inputs["input_ids"].tolist(),
        "attention_mask": inputs["attention_mask"].tolist(),
        "labels": labels.tolist()
    }

# Preprocess dataset
train_data = preprocess_data(titles, summaries)
dataset = Dataset.from_dict(train_data)

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Training setup
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate summaries for 100 random titles
random_titles = random.sample(titles, 100)
inputs = tokenizer(random_titles, return_tensors="pt", truncation=True, padding=True).to(device)

summary_ids = model.generate(
    inputs['input_ids'],
    max_length=12,
    min_length=8,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

generated_summaries = [tokenizer.decode(summary_id, skip_special_tokens=True) for summary_id in summary_ids]

# Get corresponding original summaries
random_summaries = [summaries[titles.index(title)] for title in random_titles]

# Create and save results
df_results = pd.DataFrame({
    'Original Title': random_titles,
    'Original Summary': random_summaries,
    'Generated Summary': generated_summaries
})

df_results.to_csv('random_summaries.csv', index=False)
print("CSV file saved as 'random_summaries.csv'.")