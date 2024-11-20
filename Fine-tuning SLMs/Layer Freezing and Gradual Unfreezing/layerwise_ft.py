import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from datasets import load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune BioMedBERT with layer freezing and gradual unfreezing.")
    
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of epochs (default: 3)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--lr_scheduler', type=str, default='linear',
                        help='Learning rate scheduler (default: linear)')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='./OUTPUT/',
                        help='Output directory for model checkpoints (default: ./OUTPUT/)')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length (default: 1024)')
    parser.add_argument('--model_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                        help='Model name or path (default: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)')
    parser.add_argument('--labels', type=int, default=2,
                        help='Number of labels for classification (default: 2)')
    parser.add_argument('--metric', type=str, default='recall',
                        help='Metric to use for model evaluation (default: recall)')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of layers in the model (default: 12)')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    return args

args = parse_arguments()
set_seed(args.seed)

model_name = args.model_name
labels = args.labels
metric_to_use = args.metric

my_tokenizer = AutoTokenizer.from_pretrained(f"{model_name}", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}", num_labels=labels)

max_length = args.max_length
my_tokenizer.model_max_length = max_length

# Adjust position embeddings if max_length exceeds model's position embeddings
current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
if max_length > current_max_pos:
    new_position_embeddings = nn.Embedding(max_length, embed_size)
    new_position_embeddings.weight.data[:current_max_pos, :] = model.bert.embeddings.position_embeddings.weight.data
    new_position_embeddings.weight.data[current_max_pos:, :] = model.bert.embeddings.position_embeddings.weight.data[-1, :].unsqueeze(0)
    model.bert.embeddings.position_embeddings = new_position_embeddings
    model.config.max_position_embeddings = max_length
    model.bert.embeddings.position_ids = torch.arange(max_length).expand((1, -1)).to(model.device)
    model.bert.embeddings.register_buffer("position_ids", model.bert.embeddings.position_ids)

# Freeze all layers in the BERT model
for param in model.bert.parameters():
    param.requires_grad = False

# Define Gradual Unfreezing Callback
class GradualUnfreezingCallback(TrainerCallback):
    def __init__(self, model, num_layers):
        self.model = model
        self.num_layers = num_layers
        self.currently_unfrozen = 0
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.currently_unfrozen < self.num_layers:
            layer = self.model.bert.encoder.layer[self.num_layers - 1 - self.currently_unfrozen]
            for param in layer.parameters():
                param.requires_grad = True
            print(f"Unfreezing layer {self.num_layers - self.currently_unfrozen - 1}")
            self.currently_unfrozen += 1

def loading_data():
    dataset = load_dataset("DKTech/ICSR-data")
    dataset_train = dataset['train']
    dataset_test = dataset['validation']
    dataset_train = dataset_train.shuffle(seed=args.seed)
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

def training_arguments():
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        warmup_ratio=args.warmup_ratio,
        greater_is_better=True,
        metric_for_best_model=metric_to_use,
        load_best_model_at_end=True,
        learning_rate=args.learning_rate, 
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        gradient_accumulation_steps=args.grad_accum_steps,
        fp16=True,
        save_total_limit=2,
        logging_dir='./logs',
        report_to='none',
        push_to_hub=False
    )
    return training_args

training_args = training_arguments()

num_layers = args.num_layers
gradual_unfreezing_callback = GradualUnfreezingCallback(model, num_layers)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=my_tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=my_tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience), gradual_unfreezing_callback],
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
