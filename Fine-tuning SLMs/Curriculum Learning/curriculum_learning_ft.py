#########################
## Curriculum Learning ##
#########################

import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from datasets import load_dataset, concatenate_datasets, Dataset
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def main():
    parser = argparse.ArgumentParser(description='Train a model for text classification')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='linear', help='Learning rate scheduler')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--model_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', help='Model name or path')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    args = parser.parse_args()
    
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    lr_scheduler = args.lr_scheduler
    grad_accum_steps = args.grad_accum_steps
    warmup_ratio = args.warmup_ratio
    model_name = args.model_name
    max_length = args.max_length

    labels = 2
    metric_to_use = 'recall'
    my_tokenizer = AutoTokenizer.from_pretrained(f"{model_name}", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}", num_labels=labels)
    
    my_tokenizer.model_max_length = max_length
    
    current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
    if max_length > current_max_pos:
        new_position_embeddings = nn.Embedding(max_length, embed_size)
        new_position_embeddings.weight.data[:current_max_pos, :] = model.bert.embeddings.position_embeddings.weight.data
        new_position_embeddings.weight.data[current_max_pos:, :] = model.bert.embeddings.position_embeddings.weight.data[-1, :].unsqueeze(0)
        model.bert.embeddings.position_embeddings = new_position_embeddings
        model.config.max_position_embeddings = max_length
        model.bert.embeddings.position_ids = torch.arange(max_length).expand((1, -1)).to(model.device)
        model.bert.embeddings.register_buffer("position_ids", model.bert.embeddings.position_ids)
    
    def loading_data():
        dataset = load_dataset("DKTech/ICSR-data")
        dataset_train = dataset['train']
        dataset_test = dataset['validation']
        dataset_train = dataset_train.shuffle(seed=42)
        return dataset_train, dataset_test
    
    def add_difficulty(example):
        example['difficulty'] = len(example['text'].split())
        return example
    
    dataset_train, dataset_test = loading_data()
    dataset_train = dataset_train.map(add_difficulty)
    dataset_train = dataset_train.sort('difficulty')
    
    def tokenize_function(examples):
        tokenized_inputs = my_tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        tokenized_inputs["labels"] = examples["label"]
        return tokenized_inputs
    
    tokenized_datasets_train = dataset_train.map(
        tokenize_function, batched=True, remove_columns=["text", "difficulty"]
    )
    tokenized_datasets_test = dataset_test.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    
    tokenized_datasets_train.set_format(type='torch')
    tokenized_datasets_test.set_format(type='torch')
    
    tokenized_train = tokenized_datasets_train
    tokenized_test = tokenized_datasets_test
    
    training_args = TrainingArguments(
        output_dir='./OUTPUT/',
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        warmup_ratio=warmup_ratio,
        greater_is_better=True,
        metric_for_best_model=metric_to_use,
        load_best_model_at_end=True,
        learning_rate=lr, 
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler,
        gradient_accumulation_steps=grad_accum_steps,
        fp16=True,
        logging_dir='./logs',
        save_total_limit=2,
        report_to='none',
        push_to_hub=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=my_tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=my_tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        accuracy = accuracy_score(labels, predictions)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    trainer.compute_metrics = compute_metrics
    
    trainer.train()
    
    predict = trainer.predict(tokenized_test)
    
    logits = predict.predictions
    labels = predict.label_ids
    predictions = np.argmax(logits, axis=-1)
    
    print('Classification report on the test set...')
    print(classification_report(labels, predictions))
    
    print('Confusion matrix...')
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

if __name__ == '__main__':
    main()
