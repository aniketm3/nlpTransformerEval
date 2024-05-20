# scripts/evaluate_bert.py

from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from datasets import load_from_disk

def evaluate_model(model_path, dataset_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(model_path)

    jeopardy_dataset = load_from_disk(dataset_path)

    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    correct = 0
    total = len(jeopardy_dataset)

    for item in jeopardy_dataset:
        question = item['question']
        context = item['context']

        result = nlp(question=question, context=context)
        if result['answer'] == item['answer']:
            correct += 1

    accuracy = correct / total
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    # Evaluate before fine-tuning
    print("Evaluating pre-trained BERT...")
    evaluate_model('bert-base-uncased', '../data/jeopardy')

    # Evaluate after fine-tuning
    print("Evaluating fine-tuned BERT...")
    evaluate_model('../models/fine_tuned_bert', '../data/jeopardy')
