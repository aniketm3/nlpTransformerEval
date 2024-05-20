# scripts/fine_tune_bert.py

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk

def fine_tune_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    
    dataset = load_from_disk('../data/tokenized_wikipedia')

    training_args = TrainingArguments(
        output_dir='../models/results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='../models/logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained('../models/fine_tuned_bert')

if __name__ == "__main__":
    fine_tune_bert()
