import torch
import argparse  # For parsing command-line arguments
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import load_dataset
import lightning as L
from torchmetrics.classification import (
    BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score
)
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

class LitTextClassification(L.LightningModule):
    def __init__(self, model_name, learning_rate):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Instantiate the BERT model for sequence classification
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        
        # Initialize metrics for training
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_accuracy = BinaryAccuracy()
        self.train_f1 = BinaryF1Score()
        
        # Initialize metrics for validation
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_accuracy = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()

    def training_step(self, batch, batch_idx):
        # Forward pass and compute loss
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        logits = outputs.logits

        # Get predictions
        preds = torch.argmax(logits, dim=1)
        targets = batch["labels"]

        # Update training metrics
        self.train_precision.update(preds, targets)
        self.train_recall.update(preds, targets)
        self.train_accuracy.update(preds, targets)
        self.train_f1.update(preds, targets)

        # Log loss
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        # Compute metrics at the end of the training epoch
        train_precision = self.train_precision.compute()
        train_recall = self.train_recall.compute()
        train_accuracy = self.train_accuracy.compute()
        train_f1 = self.train_f1.compute()

        # Log metrics
        self.log("train_precision", train_precision, on_epoch=True)
        self.log("train_recall", train_recall, on_epoch=True)
        self.log("train_accuracy", train_accuracy, on_epoch=True)
        self.log("train_f1", train_f1, on_epoch=True)

        # Reset metrics for the next epoch
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_accuracy.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        # Forward pass and compute loss
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        logits = outputs.logits

        # Get predictions
        preds = torch.argmax(logits, dim=1)
        targets = batch["labels"]

        # Update validation metrics
        self.val_precision.update(preds, targets)
        self.val_recall.update(preds, targets)
        self.val_accuracy.update(preds, targets)
        self.val_f1.update(preds, targets)

        # Log loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Compute metrics at the end of the validation epoch
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        val_accuracy = self.val_accuracy.compute()
        val_f1 = self.val_f1.compute()

        # Log metrics
        self.log("val_precision", val_precision, prog_bar=True)
        self.log("val_recall", val_recall, prog_bar=True)
        self.log("val_accuracy", val_accuracy, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)

        # Reset metrics for the next epoch
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_accuracy.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        # Define optimizer with configurable learning rate
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fine-tune BERT model for text classification")
    parser.add_argument("--model_name", type=str, default="dmis-lab/biobert-v1.1",
                        help="Name or path of the pre-trained model")
    parser.add_argument("--dataset_name", type=str, default="DKTech/ICSR-data",
                        help="Name of the Hugging Face dataset")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training and validation")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for the optimizer")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--stride", type=int, default=128, help="Stride size for tokenization")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker threads for data loading")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Type of accelerator ('cpu', 'gpu')")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use for training")
    parser.add_argument("--output_dir", type=str, default="saved_model", help="Directory to save the fine-tuned model")
    parser.add_argument("--monitor_metric", type=str, default="val_recall", help="Metric to monitor for checkpointing")
    parser.add_argument("--monitor_mode", type=str, default="max", help="Mode for monitoring metric ('min' or 'max')")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_name)
    if "validation" in dataset.keys():
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    else:
        # Split dataset into train and validation sets
        split_dataset = dataset["train"].train_test_split(test_size=0.1)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    # Tokenization function
    def tokenize_function(examples):
        # Tokenize the texts
        tokenized_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_overflowing_tokens=True,
            stride=args.stride,
        )
        # Map tokens back to labels
        sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
        labels = []
        for i in range(len(tokenized_inputs["input_ids"])):
            index = sample_mapping[i]
            labels.append(examples["label"][index])
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    # Data collator
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create DataLoader for training
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator,
    )

    # Create DataLoader for validation
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collator,
    )

    # Initialize the LightningModule
    model = LitTextClassification(model_name=args.model_name, learning_rate=args.learning_rate)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor_metric,
        dirpath='checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        mode=args.monitor_mode
    )
    early_stopping = EarlyStopping(
        monitor=args.monitor_metric,
        mode=args.monitor_mode,
        patience=args.patience
    )

    # Initialize the Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True,
        accelerator=args.accelerator,
        devices=args.devices,
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Load the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    best_model = LitTextClassification.load_from_checkpoint(best_model_path)

    # Save the fine-tuned model and tokenizer
    output_dir = args.output_dir  # Use the output directory from arguments
    best_model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Optionally, evaluate the best model
    val_results = trainer.validate(model=best_model, dataloaders=val_loader)
    print(val_results)
    callback_metrics = trainer.callback_metrics
    if 'val_recall' in callback_metrics:
        best_val_recall = callback_metrics['val_recall'].item()
        print(f'Best Validation Recall: {best_val_recall}')
