import argparse  # For command-line argument parsing
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary components from transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    AdamW,
    get_scheduler,
)

# Import evaluation metrics from sklearn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)

# Import dataset loader from Hugging Face datasets
from datasets import load_dataset

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_hyperparameters():
    """
    Parse command-line arguments to get hyperparameters.

    Returns:
        args: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Fine-tune BERT model for classification')

    # Add arguments for hyperparameters with default values and help descriptions
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate (default: 2e-5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs (default: 3)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
    parser.add_argument('--lr_scheduler', type=str, default='linear', help='Learning rate scheduler (default: linear)')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length (default: 512)')
    parser.add_argument('--model_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                        help='Model name or path (default: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)')
    parser.add_argument('--output_dir', type=str, default='./OUTPUT/', help='Output directory (default: ./OUTPUT/)')
    parser.add_argument('--save_total_limit', type=int, default=3, help='Limit the total amount of checkpoints (default: 3)')
    args = parser.parse_args()

    return args

def main():
    """
    Main function to run the fine-tuning script.
    """
    # Parse command-line arguments
    args = get_hyperparameters()
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Define the number of labels/classes for classification
    labels = 2
    # Choose the metric to use for selecting the best model
    metric_to_use = 'recall'

    # Load the tokenizer based on the specified model name
    my_tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}", use_fast=True)
    # Load the pre-trained model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(f"{args.model_name}", num_labels=labels)
    
    # Set the maximum sequence length for the tokenizer
    max_length = args.max_length
    my_tokenizer.model_max_length = max_length

    # Adjust the model's position embeddings if the max_length exceeds the original model's max position embeddings
    current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
    if max_length > current_max_pos:
        # Create new position embeddings extending to the new max_length
        new_position_embeddings = nn.Embedding(max_length, embed_size)
        # Copy existing position embeddings into the new embeddings
        new_position_embeddings.weight.data[:current_max_pos, :] = model.bert.embeddings.position_embeddings.weight.data
        # Initialize the remaining position embeddings
        new_position_embeddings.weight.data[current_max_pos:, :] = model.bert.embeddings.position_embeddings.weight.data[-1, :].unsqueeze(0)
        # Replace the model's position embeddings with the new embeddings
        model.bert.embeddings.position_embeddings = new_position_embeddings
        # Update the model configuration
        model.config.max_position_embeddings = max_length
        # Update position ids
        model.bert.embeddings.position_ids = torch.arange(max_length).expand((1, -1)).to(model.device)
        model.bert.embeddings.register_buffer("position_ids", model.bert.embeddings.position_ids)
    
    def loading_data():
        """
        Load and preprocess the dataset.

        Returns:
            dataset_train: Training dataset.
            dataset_test: Testing dataset.
        """
        # Load the dataset from Hugging Face datasets
        dataset = load_dataset("DKTech/ICSR-data")
        # Get the training and validation splits
        dataset_train = dataset['train']
        dataset_test = dataset['validation']
        # Shuffle the training dataset
        dataset_train = dataset_train.shuffle(seed=args.seed)
        return dataset_train, dataset_test
    
    def tokenize_function(examples):
        """
        Tokenize the input texts.

        Args:
            examples: A dictionary containing 'text' and 'label'.

        Returns:
            tokenized_inputs: A dictionary with tokenized inputs and labels.
        """
        # Tokenize the input texts with truncation and padding
        tokenized_inputs = my_tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding='max_length',
        )
        # Include labels in the tokenized inputs
        tokenized_inputs["labels"] = examples["label"]
        return tokenized_inputs

    # Load the datasets
    dataset_train, dataset_test = loading_data()

    # Tokenize the datasets
    tokenized_datasets_train = dataset_train.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_datasets_test = dataset_test.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Set the format of the datasets to PyTorch tensors
    tokenized_datasets_train.set_format(type='torch')
    tokenized_datasets_test.set_format(type='torch')

    # Assign tokenized datasets to variables
    tokenized_train = tokenized_datasets_train
    tokenized_test = tokenized_datasets_test

    # Initialize the optimizer (AdamW is recommended for transformers)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        correct_bias=True
    )

    # Calculate the total number of training steps
    total_steps = len(tokenized_train) // args.batch_size * args.num_epochs

    # Create the learning rate scheduler
    scheduler = get_scheduler(
        name=args.lr_scheduler,  # Scheduler type (e.g., 'linear', 'cosine')
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    # Define the training arguments for the Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # Output directory for checkpoints
        overwrite_output_dir=True,   # Overwrite the output directory
        do_train=True,               # Perform training
        do_eval=True,                # Perform evaluation
        evaluation_strategy='epoch', # Evaluate at the end of each epoch
        save_strategy='epoch',       # Save checkpoints at the end of each epoch
        warmup_ratio=args.warmup_ratio,  # Warmup ratio for the scheduler
        greater_is_better=True,      # Whether a greater metric is better
        metric_for_best_model=metric_to_use,  # Metric to use to select the best model
        load_best_model_at_end=True, # Load the best model at the end of training
        learning_rate=args.learning_rate,  # Learning rate
        per_device_train_batch_size=args.batch_size,  # Training batch size per device
        per_device_eval_batch_size=args.batch_size,   # Evaluation batch size per device
        num_train_epochs=args.num_epochs,             # Number of training epochs
        weight_decay=args.weight_decay,               # Weight decay
        lr_scheduler_type=args.lr_scheduler,          # Learning rate scheduler type
        gradient_accumulation_steps=args.grad_accum_steps,  # Gradient accumulation steps
        fp16=True,                # Use mixed precision (fp16)
        save_total_limit=args.save_total_limit,  # Limit the total number of checkpoints saved
        logging_dir='./logs',     # Directory for storing logs
        report_to='none',         # Reporting to 'none' disables logging to WandB or TensorBoard
        push_to_hub=False,        # Do not push the model to Hugging Face Hub
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,                         # The model to be trained
        args=training_args,                  # Training arguments
        train_dataset=tokenized_train,       # Training dataset
        eval_dataset=tokenized_test,         # Evaluation dataset
        tokenizer=my_tokenizer,              # The tokenizer
        data_collator=DataCollatorWithPadding(tokenizer=my_tokenizer),  # Data collator for dynamic padding
        optimizers=(optimizer, scheduler),   # Optimizer and scheduler
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Early stopping callback
    )

    def compute_metrics(eval_pred):
        """
        Compute evaluation metrics.

        Args:
            eval_pred: A tuple of (logits, labels).

        Returns:
            A dictionary with evaluation metrics.
        """
        # Unpack predictions and true labels
        logits, labels = eval_pred
        # Get the predicted class by selecting the highest logit
        predictions = np.argmax(logits, axis=-1)
        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        # Calculate accuracy
        accuracy = accuracy_score(labels, predictions)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    # Set the compute_metrics function for the Trainer
    trainer.compute_metrics = compute_metrics

    # Start training the model
    trainer.train()

    # Make predictions on the test set
    predict = trainer.predict(tokenized_test)

    # Get logits and labels from predictions
    logits = predict.predictions
    labels = predict.label_ids
    # Convert logits to predicted classes
    predictions = np.argmax(logits, axis=-1)

    # Print the classification report
    print('Classification report on the test set...')
    print(classification_report(labels, predictions))

    # Plot and display the confusion matrix
    print('Confusion matrix...')
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

if __name__ == '__main__':
    main()
