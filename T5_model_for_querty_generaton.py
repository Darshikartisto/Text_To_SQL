import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SpiderDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"translate English to SQL: {item['nl_query']}"
        target_text = item['sql_query']

        input_encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding.input_ids.flatten(),
            "attention_mask": input_encoding.attention_mask.flatten(),
            "labels": target_encoding.input_ids.flatten(),
        }
    
def fine_tune_t5_spider():
    with open("processed_spider_data.json", "r") as f:
        data = json.load(f)

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    train_dataset = SpiderDataset(tokenizer, train_data)
    val_dataset = SpiderDataset(tokenizer, val_data)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting fine-tuning...")
    trainer.train()

    model.save_pretrained("./fine_tuned_t5_spider")
    tokenizer.save_pretrained("./fine_tuned_t5_spider")
    print("Fine-tuning complete. Model saved to ./fine_tuned_t5_spider")

if __name__ == "__main__":
    fine_tune_t5_spider()