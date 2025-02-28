
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ Define constants
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"  # Better pretrained model
OUTPUT_DIR = "models"
DATA_PATH = "../data/sentiment_dataset.csv"  # Update with dataset choice
USE_HUGGINGFACE_DATASET = False  # Toggle between CSV dataset and Hugging Face dataset
HUGGINGFACE_DATASET_NAME = "imdb"  # Example dataset
DATASET_SAMPLE_SIZE = 10000  # Limit dataset size for faster training

# ✅ Load dataset dynamically
def load_data():
    if USE_HUGGINGFACE_DATASET:
        dataset = load_dataset(HUGGINGFACE_DATASET_NAME)
        dataset = dataset.map(lambda x: {"text": x["text"], "labels": 2 if x["label"] == 1 else 0})
        dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in ["text", "labels"]])
        dataset = DatasetDict({
            "train": dataset["train"].shuffle(seed=42).select(range(min(DATASET_SAMPLE_SIZE, len(dataset["train"])))) ,
            "validation": dataset["test"].shuffle(seed=42).select(range(min(DATASET_SAMPLE_SIZE // 10, len(dataset["test"]))))
        })
    else:
        df = pd.read_csv(DATA_PATH)
        df = df.rename(columns={"text": "text", "sentiment": "labels"})
        df = df.dropna(subset=["labels"])
        df['labels'] = df['labels'].map({'positive': 2, 'neutral': 1, 'negative': 0})
        df = df.dropna(subset=["labels"])
        df['labels'] = df['labels'].astype(int)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].tolist(), df['labels'].tolist(), test_size=0.2, random_state=42)
        dataset = DatasetDict({
            "train": Dataset.from_dict({"text": train_texts, "labels": train_labels}),
            "validation": Dataset.from_dict({"text": val_texts, "labels": val_labels})
        })
    return dataset

# ✅ Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

# ✅ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

dataset = load_data()
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"]).with_format("torch")

# ✅ Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="none",
    push_to_hub=False,
    fp16=torch.cuda.is_available(),
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=1)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ✅ Train the model
trainer.train()

# ✅ Save model & tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Training complete. Model saved to {OUTPUT_DIR}")

# ✅ Test Sentiment Predictions
sample_texts = dataset["validation"]["text"][:2]  # Take two validation samples
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("🔍 Sample Sentiment Predictions:")
for text in sample_texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=-1).item()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    print(f"Text: {text}\nPredicted Sentiment: {label_map[prediction]}\n")
