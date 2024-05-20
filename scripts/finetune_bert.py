# scripts/finetune_bert.py

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


def fine_tune_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    
    dataset = load_dataset("wikipedia", "20220301.en", split='train', trust_remote_code=True)
    small_dataset = dataset.take(30) #supposed to shorten the amount of the dataset downloaded to speed up verifying if process works

    training_args = TrainingArguments(
        output_dir='../models/results',
        num_train_epochs=1, #3
        per_device_train_batch_size=8,
        warmup_steps=5, #5
        weight_decay=0.1, #0.01
        logging_dir='../models/logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_dataset,
    )

    for data in small_dataset:
        print(data)
    # trainer.train()
    # model.save_pretrained('../models/fine_tuned_bert')

if __name__ == "__main__":
    fine_tune_bert()
