# Import necessary libraries and modules
import argparse  # For command-line argument parsing
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    precision_recall_curve,
)
from datasets import load_dataset
import torch
import numpy as np
import random
from collections import defaultdict
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a text classification model with hyperparameter tuning.")
    parser.add_argument("--model_name", type=str, default="dmis-lab/biobert-v1.1", help="Name of the pretrained model.")
    parser.add_argument("--labels", type=int, default=2, help="Number of classes.")
    parser.add_argument("--metric_to_use", type=str, default="f1", help="Metric to optimize during hyperparameter tuning.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of trials for hyperparameter tuning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--dataset_name", type=str, default="DKTech/ICSR-data", help="Dataset name.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--stride", type=int, default=256, help="Tokenization stride.")
    parser.add_argument("--output_dir", type=str, default="./OUTPUT/testing/", help="Output directory for model.")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Early stopping patience.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps.")
    parser.add_argument("--fp16", action='store_true', help="Whether to use fp16 precision.")
    parser.add_argument("--overwrite_output_dir", action='store_true', help="Whether to overwrite the output directory.")
    # You can add more arguments as needed
    return parser.parse_args()

def main():
    # Parse the command-line arguments
    args = parse_args()

    # Function to set the random seed for reproducibility
    def set_seed(seed=42):
        random.seed(seed)  # Set the seed for the random module
        np.random.seed(seed)  # Set the seed for NumPy
        torch.manual_seed(seed)  # Set the seed for PyTorch CPU
        torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs (if using CUDA)

    # Set the random seed
    set_seed(args.seed)

    # Define the pre-trained model to use
    model_name = args.model_name
    labels = args.labels
    metric_to_use = args.metric_to_use

    # Initialize the tokenizer from the pre-trained model
    my_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Function to load and prepare the dataset
    def loading_data():
        # Load the dataset from the Hugging Face Hub
        dataset = load_dataset(args.dataset_name)
        dataset_train = dataset['train']
        dataset_test = dataset['validation'] if 'validation' in dataset else dataset['test']
        # Shuffle the training data
        dataset_train = dataset_train.shuffle(seed=args.seed)
        return dataset_train, dataset_test

    # Tokenization function to process the text data
    def tokenize_function(examples):
        max_length = args.max_length  # Maximum sequence length
        stride = args.stride  # Overlapping tokens between segments
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        overflow_mapping = []  # Keep track of which samples have been split

        # Loop through each example in the batch
        for i in range(len(examples["text"])):
            # Tokenize the text with truncation and stride
            inputs = my_tokenizer(
                examples["text"][i],
                max_length=max_length,
                truncation=True,  # Truncate sequences longer than max_length
                padding=False,  # Do not pad sequences here
                return_overflowing_tokens=True,  # Return overflowing tokens for long sequences
                stride=stride,  # Number of tokens to shift for the next segment
            )
            num_chunks = len(inputs["input_ids"])  # Number of chunks created
            # Loop through each chunk
            for k in range(num_chunks):
                input_ids_list.append(inputs["input_ids"][k])
                attention_mask_list.append(inputs["attention_mask"][k])
                labels_list.append(examples["label"][i])  # Append the label for each chunk
                overflow_mapping.append(i)  # Map chunk back to the original sample index
        # Return the tokenized inputs as a dictionary
        result = {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "label": labels_list,
            "overflow_to_sample_mapping": overflow_mapping,
        }
        return result

    # Load the dataset
    dataset_train, dataset_test = loading_data()

    # Tokenize the training and test datasets
    tokenized_datasets_train = dataset_train.map(
        tokenize_function, batched=True, remove_columns=dataset_train.column_names
    )
    tokenized_datasets_test = dataset_test.map(
        tokenize_function, batched=True, remove_columns=dataset_test.column_names
    )

    # Set the format of the datasets to PyTorch tensors
    tokenized_datasets_train.set_format(type='torch')
    tokenized_datasets_test.set_format(type='torch')

    # Assign the tokenized datasets to variables
    tokenized_train = tokenized_datasets_train
    tokenized_test = tokenized_datasets_test

    # Function to initialize the model (required for some hyperparameter tuning methods)
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=labels
        )

    # Function to compute metrics during evaluation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred  # Extract the logits and labels
        predictions = np.argmax(logits, axis=-1)  # Get the predicted class indices

        # Aggregate predictions for sequences that were split into multiple chunks
        overflow_mapping = trainer.eval_dataset["overflow_to_sample_mapping"]
        aggregated_predictions = defaultdict(list)
        aggregated_labels = {}
        for pred, label, mapping_idx in zip(predictions, labels, overflow_mapping):
            aggregated_predictions[int(mapping_idx)].append(int(pred))
            aggregated_labels[int(mapping_idx)] = int(label)

        # Final predictions are determined by majority vote
        final_predictions = []
        final_labels = []
        for idx in sorted(aggregated_predictions.keys()):
            preds = aggregated_predictions[idx]
            final_pred = max(set(preds), key=preds.count)  # Majority vote
            final_predictions.append(final_pred)
            final_labels.append(aggregated_labels[idx])

        # Compute precision, recall, F1 score, and accuracy
        precision, recall, f1, _ = precision_recall_fscore_support(
            final_labels, final_predictions, average='binary'
        )
        accuracy = accuracy_score(final_labels, final_predictions)
        # Return the metrics as a dictionary
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Define the hyperparameter search space for Optuna
    def hp_space(trial):
        return {
            # Suggest a learning rate between 5e-6 and 1e-5 on a log scale
            "learning_rate": trial.suggest_float("learning_rate", 5e-6, 1e-5, log=True),
            # Suggest batch size from the list
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [4, 8, 12, 16, 20]
            ),
            # Suggest weight decay between 0 and 0.3
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            # Suggest warmup ratio between 0 and 0.3
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
            # Suggest gradient accumulation steps from the list
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [1, 2, 4, 8]
            ),
        }

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # Directory to save the model outputs
        overwrite_output_dir=args.overwrite_output_dir,  # Overwrite the content of the output directory
        do_train=True,
        do_eval=True,
        num_train_epochs=args.num_train_epochs,  # Number of training epochs
        evaluation_strategy='epoch',  # Evaluate at the end of each epoch
        save_strategy='epoch',  # Save the model at the end of each epoch
        greater_is_better=True,  # For hyperparameter tuning
        metric_for_best_model=metric_to_use,  # Metric to use for model selection
        load_best_model_at_end=True,  # Load the best model found during training
        lr_scheduler_type='linear',  # Learning rate scheduler
        fp16=args.fp16,  # Use mixed precision if specified
        logging_dir='./logs',  # Directory for logs
        report_to='none',  # Disable reporting to external services
        logging_steps=args.logging_steps,  # Log every specified steps
        push_to_hub=False,  # Do not push the model to the Hugging Face Hub
    )

    # Create a Trainer object with the specified arguments
    trainer = Trainer(
        model_init=model_init,  # Model initialization function
        args=training_args,
        train_dataset=tokenized_train,  # Training dataset
        eval_dataset=tokenized_test,  # Evaluation dataset
        tokenizer=my_tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=my_tokenizer),  # Data collator for batching
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],  # Early stopping
        compute_metrics=compute_metrics,  # Metrics function
    )

    # Perform hyperparameter search using Optuna
    best_run = trainer.hyperparameter_search(
        direction="maximize",  # Maximize the metric
        backend="optuna",  # Use Optuna as the backend
        hp_space=hp_space,  # Hyperparameter space
        n_trials=args.n_trials  # Number of trials
    )

    # Print the best hyperparameters found
    print("Best hyperparameters found:")
    print(best_run.hyperparameters)

    # Update the training arguments with the best hyperparameters
    training_args.learning_rate = best_run.hyperparameters['learning_rate']
    training_args.per_device_train_batch_size = int(best_run.hyperparameters['per_device_train_batch_size'])
    training_args.weight_decay = best_run.hyperparameters['weight_decay']
    training_args.warmup_ratio = best_run.hyperparameters['warmup_ratio']
    training_args.gradient_accumulation_steps = int(best_run.hyperparameters['gradient_accumulation_steps'])

    # Create a new Trainer with the best hyperparameters
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=my_tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=my_tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        compute_metrics=compute_metrics,
    )

    # Train the model with the best hyperparameters
    trainer.train()

    # Make predictions on the test dataset
    predict = trainer.predict(tokenized_test)

    # Extract the logits (model outputs before softmax) and labels
    logits = predict.predictions
    labels = predict.label_ids
    overflow_mapping = tokenized_test["overflow_to_sample_mapping"]  # Mapping for aggregation

    # Calculate probabilities using softmax
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    probs = probs.numpy()

    # Compute precision-recall curve to find the best threshold
    precision_vals, recall_vals, thresholds = precision_recall_curve(labels, probs[:, 1])

    # Compute F1 scores for different thresholds
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)  # Avoid division by zero

    # Find the threshold that gives the maximum F1 score
    best_threshold_idx = np.argmax(f1_scores)
    if best_threshold_idx < len(thresholds):
        best_threshold = thresholds[best_threshold_idx]
    else:
        best_threshold = thresholds[-1]

    # Apply the best threshold to convert probabilities to binary predictions
    predictions = (probs[:, 1] >= best_threshold).astype(int)

    # Aggregate predictions for sequences that were split
    aggregated_predictions = defaultdict(list)
    aggregated_labels = {}

    for pred, label, mapping_idx in zip(predictions, labels, overflow_mapping):
        aggregated_predictions[int(mapping_idx)].append(int(pred))
        aggregated_labels[int(mapping_idx)] = int(label)

    # Determine final predictions by majority vote after aggregation
    final_predictions = []
    final_labels = []

    for idx in sorted(aggregated_predictions.keys()):
        preds = aggregated_predictions[idx]
        final_pred = max(set(preds), key=preds.count)  # Majority vote
        final_predictions.append(final_pred)
        final_labels.append(aggregated_labels[idx])

    # Compute final evaluation metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        final_labels, final_predictions, average='binary'
    )
    accuracy = accuracy_score(final_labels, final_predictions)

    # Print the final evaluation metrics
    print("Final evaluation metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    # Print the classification report
    print('Classification report on the test set after aggregation...')
    print(classification_report(final_labels, final_predictions))

    # Plot the confusion matrix using seaborn heatmap
    print('Confusion matrix after aggregation...')
    sns.heatmap(confusion_matrix(final_labels, final_predictions), annot=True, fmt='d')
    plt.show()

if __name__ == "__main__":
    main()

