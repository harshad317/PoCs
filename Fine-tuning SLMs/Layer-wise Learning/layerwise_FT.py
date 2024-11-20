import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    AdamW
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

def main():
    parser = argparse.ArgumentParser(description='Train a transformer model for sequence classification.')
    parser.add_argument('--model_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                        help='Pre-trained model name or path (e.g., "bert-base-uncased")')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate (e.g., 2e-5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (e.g., 16)')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs (e.g., 3)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (e.g., 0.01)')
    parser.add_argument('--lr_scheduler', type=str, default='linear', help='Learning rate scheduler (e.g., linear)')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps (e.g., 1)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio (e.g., 0.1)')
    args = parser.parse_args()

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    # Unpack arguments
    model_name = args.model_name
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    lr_scheduler = args.lr_scheduler
    grad_accum_steps = args.grad_accum_steps
    warmup_ratio = args.warmup_ratio

    labels = 2
    metric_to_use = 'recall'
    max_length = 1024

    # Load tokenizer and model
    my_tokenizer = AutoTokenizer.from_pretrained(f"{model_name}", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}", num_labels=labels)

    # Update tokenizer's max length
    my_tokenizer.model_max_length = max_length

    # Adjust position embeddings if max_length exceeds current max_pos
    current_max_pos, embed_size = model.config.max_position_embeddings, model.config.hidden_size
    if max_length > current_max_pos:
        print(f"Extending position embeddings from {current_max_pos} to {max_length}")

        max_pos = max_length
        config = model.config

        # Create new position embeddings
        new_pos_embed = nn.Embedding(max_pos, embed_size)
        new_pos_embed.weight.data[:current_max_pos, :] = model.bert.embeddings.position_embeddings.weight.data
        new_pos_embed.weight.data[current_max_pos:, :] = model.bert.embeddings.position_embeddings.weight.data[-1, :].unsqueeze(0).repeat(max_pos - current_max_pos, 1)
        model.bert.embeddings.position_embeddings = new_pos_embed
        model.config.max_position_embeddings = max_pos

    def loading_data():
        dataset = load_dataset("DKTech/ICSR-data")
        dataset_train = dataset['train']
        dataset_test = dataset['validation']
        dataset_train = dataset_train.shuffle(seed=42)
        return dataset_train, dataset_test

    def tokenize_function(examples):
        tokenized_inputs = my_tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        tokenized_inputs["labels"] = examples["label"]
        return tokenized_inputs

    dataset_train, dataset_test = loading_data()

    tokenized_datasets_train = dataset_train.map(
        tokenize_function, batched=True, remove_columns=["text"]
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
        save_total_limit=3,
        logging_dir='./logs',
        report_to='none',
        push_to_hub=False
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

    def get_optimizer_grouped_parameters(model, base_lr, weight_decay):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
        num_layers = len(layers)
        layers = layers[::-1]
        for i, layer in enumerate(layers):
            lr_layer = base_lr * (0.95 ** i)
            params = list(layer.named_parameters())
            for n, p in params:
                if any(nd in n for nd in no_decay):
                    optimizer_grouped_parameters.append({
                        'params': [p],
                        'weight_decay': 0.0,
                        'lr': lr_layer
                    })
                else:
                    optimizer_grouped_parameters.append({
                        'params': [p],
                        'weight_decay': weight_decay,
                        'lr': lr_layer
                    })
        # Classifier parameters
        classifier_params = list(model.classifier.named_parameters())
        for n, p in classifier_params:
            if any(nd in n for nd in no_decay):
                optimizer_grouped_parameters.append({
                    'params': [p],
                    'weight_decay': 0.0,
                    'lr': base_lr
                })
            else:
                optimizer_grouped_parameters.append({
                    'params': [p],
                    'weight_decay': weight_decay,
                    'lr': base_lr
                })
        return optimizer_grouped_parameters

    class CustomTrainer(Trainer):
        def create_optimizer(self):
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(
                self.model, self.args.learning_rate, self.args.weight_decay)
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=my_tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=my_tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics
    )

    trainer.train()

    predict = trainer.predict(tokenized_test)

    logits = predict.predictions
    labels = predict.label_ids
    predictions = np.argmax(logits, axis=-1)

    print('Classification report on the test set...')
    print(classification_report(labels, predictions, digits=4))

    print('Confusion matrix...')
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()
