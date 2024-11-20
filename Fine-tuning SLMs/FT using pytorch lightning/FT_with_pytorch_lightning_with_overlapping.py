import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
)
from datasets import load_dataset
import numpy as np
import random
import argparse
import os

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def get_hyperparameters():
    parser = argparse.ArgumentParser(description='Train a BioBERT model with chunk attention.')
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate (e.g., 2e-5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (e.g., 16)')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs (e.g., 3)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (e.g., 0.01)')
    parser.add_argument('--lr_scheduler', type=str, default='linear', help='Learning rate scheduler (e.g., linear)')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps (e.g., 1)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio (e.g., 0.1)')
    # Model parameters
    parser.add_argument('--model_name', type=str, default='michiyasunaga/BioLinkBERT-base',
                        help='Model name (e.g., bert-base-uncased)')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels (e.g., 2)')
    parser.add_argument('--metric_to_use', type=str, default='eval_f1', help='Metric to use for best model selection')
    # Dataset parameters
    parser.add_argument('--dataset_name', type=str, default='DKTech/ICSR-data',
                        help='Name of the dataset to load (e.g., DKTech/ICSR-data)')
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./OUTPUT/testing/', help='Directory to save outputs')
    # Chunking parameters
    parser.add_argument('--chunk_sizes', nargs='+', type=int, default=[512],
                        help='List of chunk sizes to try (e.g., 512 1024)')
    parser.add_argument('--overlap_sizes', nargs='+', type=int, default=[0, 32, 64, 128],
                        help='List of overlap sizes to try (e.g., 0 64 128)')
    args = parser.parse_args()
    return args

args = get_hyperparameters()

# Set variables from args
lr = args.lr
batch_size = args.batch_size
num_epochs = args.num_epochs
weight_decay = args.weight_decay
lr_scheduler = args.lr_scheduler
grad_accum_steps = args.grad_accum_steps
warmup_ratio = args.warmup_ratio
model_name = args.model_name
num_labels = args.num_labels
metric_to_use = args.metric_to_use
dataset_name = args.dataset_name
output_dir = args.output_dir
chunk_sizes = args.chunk_sizes
overlap_sizes = args.overlap_sizes

# Model and tokenizer initialization
my_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Load dataset
def loading_data():
    dataset = load_dataset(dataset_name)
    dataset_train = dataset['train']
    dataset_test = dataset['validation']
    dataset_train = dataset_train.shuffle(seed=42)
    return dataset_train, dataset_test

dataset_train, dataset_test = loading_data()

# Add unique ids to the datasets
def add_example_id(example, idx):
    example['example_id'] = idx
    return example

dataset_train = dataset_train.map(add_example_id, with_indices=True)
dataset_test = dataset_test.map(add_example_id, with_indices=True)

# Function to perform training and evaluation for given chunk and overlap sizes
def train_and_evaluate(chunk_size, overlap_size):
    # Function to tokenize and chunk the text
    def tokenize_and_chunk(examples):
        # Tokenize without truncation
        tokenized_inputs = my_tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
        )
        
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_example_ids = []
        
        for idx in range(len(tokenized_inputs["input_ids"])):
            input_ids = tokenized_inputs["input_ids"][idx]
            attention_mask = tokenized_inputs["attention_mask"][idx]
            label = examples["label"][idx]
            example_id = examples["example_id"][idx]
            
            # Process the first 512 tokens as the initial chunk
            first_chunk_input_ids = input_ids[:512]
            first_chunk_attention_mask = attention_mask[:512]
            all_input_ids.append(first_chunk_input_ids)
            all_attention_masks.append(first_chunk_attention_mask)
            all_labels.append(label)
            all_example_ids.append(example_id)
            
            # Now process remaining tokens with overlapping chunks
            total_length = len(input_ids)
            remainder_start = 512 - overlap_size
            for i in range(remainder_start, total_length, chunk_size - overlap_size):
                if i <= 512:
                    continue
                chunk_input_ids = input_ids[i:i + chunk_size]
                chunk_attention_mask = attention_mask[i:i + chunk_size]
                if len(chunk_input_ids) < 10:
                    continue
                all_input_ids.append(chunk_input_ids)
                all_attention_masks.append(chunk_attention_mask)
                all_labels.append(label)
                all_example_ids.append(example_id)
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
            "example_id": all_example_ids,
        }
    
    # Apply tokenization and chunking
    tokenized_datasets_train = dataset_train.map(
        tokenize_and_chunk, batched=True, remove_columns=dataset_train.column_names
    )
    tokenized_datasets_test = dataset_test.map(
        tokenize_and_chunk, batched=True, remove_columns=dataset_test.column_names
    )
    
    # Set the format to PyTorch tensors
    tokenized_datasets_train.set_format(type='torch')
    tokenized_datasets_test.set_format(type='torch')
    
    # Custom model to handle chunked inputs with attention
    class BioBERTWithChunkAttention(nn.Module):
        def __init__(self, model_name, num_labels):
            super(BioBERTWithChunkAttention, self).__init__()
            self.num_labels = num_labels
            self.bert = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.bert.config.hidden_size
            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 1)
            )
            self.classifier = nn.Linear(self.hidden_size, num_labels)
                
        def forward(self, input_ids=None, attention_mask=None, labels=None, example_id=None):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_outputs = outputs.last_hidden_state[:, 0, :]
            
            example_id_np = example_id.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            unique_ids = np.unique(example_id_np)
            aggregated_outputs = []
            aggregated_labels = []
            for uid in unique_ids:
                indices = np.where(example_id_np == uid)[0]
                embeddings = cls_outputs[indices]
                attn_weights = self.attention(embeddings)
                attn_weights = torch.softmax(attn_weights.squeeze(-1), dim=0)
                agg_embedding = torch.sum(attn_weights.unsqueeze(-1) * embeddings, dim=0)
                aggregated_outputs.append(agg_embedding)
                aggregated_labels.append(labels_np[indices[0]])
            
            aggregated_outputs = torch.stack(aggregated_outputs).to(input_ids.device)
            aggregated_labels = torch.tensor(aggregated_labels).to(input_ids.device)
            
            logits = self.classifier(aggregated_outputs)
            
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, aggregated_labels)
            
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
    
    # Initialize the custom model
    model = BioBERTWithChunkAttention(model_name, num_labels)
    
    # Custom data collator to handle example_id
    class CustomDataCollatorWithPadding(DataCollatorWithPadding):
        def __call__(self, features):
            example_ids = [f['example_id'] for f in features]
            labels = [f['labels'] for f in features]
            batch = super().__call__([{k: v for k, v in f.items() if k != 'example_id'} for f in features])
            batch['example_id'] = torch.tensor(example_ids)
            batch['labels'] = torch.tensor(labels)
            return batch
    
    data_collator = CustomDataCollatorWithPadding(tokenizer=my_tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        save_strategy='no',
        warmup_ratio=warmup_ratio,
        greater_is_better=True,
        metric_for_best_model=metric_to_use,
        load_best_model_at_end=False,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler,
        gradient_accumulation_steps=grad_accum_steps,
        fp16=True,
        report_to='none',
        push_to_hub=False
    )
    
    # Custom Trainer to handle the model inputs and aggregation
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            example_id = inputs.pop("example_id")
            outputs = model(**inputs, labels=labels, example_id=example_id)
            loss = outputs[0]
            return (loss, outputs) if return_outputs else loss
    
        def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
            labels = inputs.pop("labels")
            example_id = inputs.pop("example_id")
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model(**inputs, labels=labels, example_id=example_id)
            loss = outputs[0].mean().detach() if outputs[0] is not None else None
            logits = outputs[1]
    
            logits = logits.detach().cpu()
            labels = labels.detach().cpu()
            example_id = example_id.detach().cpu()
    
            return (loss, (logits, example_id), (labels, example_id))
    
    # Adjust compute_metrics function
    def compute_metrics(eval_pred):
        preds_and_ids, labels_and_ids = eval_pred
        logits, pred_example_ids = preds_and_ids
        labels, label_example_ids = labels_and_ids
    
        # Aggregate logits per example ID
        aggregated_logits = {}
        for logit, eid in zip(logits, pred_example_ids):
            eid = int(eid)
            if eid not in aggregated_logits:
                aggregated_logits[eid] = []
            aggregated_logits[eid].append(logit.numpy())
    
        aggregated_labels = {}
        for label, eid in zip(labels, label_example_ids):
            eid = int(eid)
            aggregated_labels[eid] = int(label.numpy())  # Labels are the same per example
    
        # Compute mean logits per example ID
        final_logits = []
        final_labels = []
        for eid in aggregated_logits:
            mean_logit = np.mean(aggregated_logits[eid], axis=0)
            final_logits.append(mean_logit)
            final_labels.append(aggregated_labels[eid])
    
        final_predictions = np.argmax(final_logits, axis=-1)
    
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            final_labels, final_predictions, average='binary'
        )
        accuracy = accuracy_score(final_labels, final_predictions)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    # Initialize the trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_test,
        tokenizer=my_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    eval_metrics = trainer.evaluate()
    
    return eval_metrics

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store the results
results = {}

# Loop over combinations of chunk_size and overlap_size
for chunk_size in chunk_sizes:
    for overlap_size in overlap_sizes:
        if overlap_size >= chunk_size:
            continue
        print(f"\nTraining with chunk_size={chunk_size} and overlap_size={overlap_size}")
        eval_metrics = train_and_evaluate(chunk_size, overlap_size)
        key = f"chunk_{chunk_size}_overlap_{overlap_size}"
        results[key] = eval_metrics
        print(f"Results for {key}: {eval_metrics}")

# After all combinations, find the best one based on evaluation metric
best_key = max(results, key=lambda k: results[k]['eval_f1'])
print(f"\nBest combination: {best_key} with F1 score: {results[best_key]['eval_f1']:.4f}")

# Print all results
print("\nAll results:")
for key in results:
    print(f"{key}: {results[key]}")

# Save results to a file
results_file = os.path.join(output_dir, 'results.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\nResults have been saved to {results_file}")
