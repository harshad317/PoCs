import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig
from datasets import load_dataset
import lightning as L
from torchmetrics.classification import (
    BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score
)
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

class LitTextClassification(L.LightningModule):
    def __init__(self, model_name, new_max_positions, learning_rate):
        super().__init__()
        # Load the configuration without modifying max_position_embeddings
        config = BertConfig.from_pretrained(model_name)
        
        # Load the pretrained model with the original configuration
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        
        # Extend the positional embeddings and adjust buffers
        self.extend_position_embeddings(new_max_positions=new_max_positions)
        
        # Define metrics for training and validation
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_accuracy = BinaryAccuracy()
        self.train_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_accuracy = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        
        # Save hyperparameters
        self.save_hyperparameters()

    def extend_position_embeddings(self, new_max_positions=1024):
        # Get the existing position embeddings from the model
        orig_embeds = self.model.bert.embeddings.position_embeddings.weight.data
        orig_max_positions, embedding_dim = orig_embeds.shape

        if orig_max_positions >= new_max_positions:
            print(f"Original position embeddings already have {orig_max_positions} positions, no need to extend.")
            self.model.config.max_position_embeddings = orig_max_positions
            return

        # Create new position embeddings
        new_position_embeddings = torch.nn.Embedding(new_max_positions, embedding_dim)
        # Copy the original embeddings
        new_position_embeddings.weight.data[:orig_max_positions, :] = orig_embeds

        # Initialize the added positions
        num_extra_positions = new_max_positions - orig_max_positions
        new_position_embeddings.weight.data[orig_max_positions:, :] = orig_embeds.mean(dim=0).unsqueeze(0).repeat(num_extra_positions, 1)

        # Assign new embeddings to the model
        self.model.bert.embeddings.position_embeddings = new_position_embeddings

        # Update config
        self.model.config.max_position_embeddings = new_max_positions

        # Adjust position_ids
        self.model.bert.embeddings.position_ids = torch.arange(new_max_positions).unsqueeze(0)
        # Adjust token_type_ids buffer
        self.model.bert.embeddings.register_buffer(
            "token_type_ids", torch.zeros((1, new_max_positions), dtype=torch.long), persistent=False
        )

    def training_step(self, batch, batch_idx):
        # Forward pass through the model
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["label"],
        )
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        targets = batch["label"]
        # Update training metrics
        self.train_precision.update(preds, targets)
        self.train_recall.update(preds, targets)
        self.train_accuracy.update(preds, targets)
        self.train_f1.update(preds, targets)
        # Log training loss
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        # Compute and log training metrics at the end of each epoch
        train_precision = self.train_precision.compute()
        train_recall = self.train_recall.compute()
        train_accuracy = self.train_accuracy.compute()
        train_f1 = self.train_f1.compute()
        self.log("train_precision", train_precision, on_epoch=True)
        self.log("train_recall", train_recall, on_epoch=True)
        self.log("train_accuracy", train_accuracy, on_epoch=True)
        self.log("train_f1", train_f1, on_epoch=True)
        # Reset metrics
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_accuracy.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        # Forward pass through the model
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["label"],
        )
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        targets = batch["label"]
        # Update validation metrics
        self.val_precision.update(preds, targets)
        self.val_recall.update(preds, targets)
        self.val_accuracy.update(preds, targets)
        self.val_f1.update(preds, targets)
        # Log validation loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Compute and log validation metrics at the end of each epoch
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        val_accuracy = self.val_accuracy.compute()
        val_f1 = self.val_f1.compute()
        self.log("val_precision", val_precision, prog_bar=True)
        self.log("val_recall", val_recall, prog_bar=True)
        self.log("val_accuracy", val_accuracy, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)
        # Reset metrics
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_accuracy.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        # Configure the optimizer with the specified learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fine-tune BERT model for text classification.')
    parser.add_argument('--model_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                        help='Pre-trained BERT model name.')
    parser.add_argument('--tokenizer_name', type=str, default='dmis-lab/biobert-v1.1',
                        help='Tokenizer name.')
    parser.add_argument('--dataset_name', type=str, default='DKTech/ICSR-data',
                        help='HuggingFace dataset name.')
    parser.add_argument('--output_dir', type=str, default='saved_model_with_1024_tokens',
                        help='Output directory to save the fine-tuned model.')
    parser.add_argument('--max_seq_length', type=int, default=1024,
                        help='Maximum sequence length.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help='Learning rate for optimizer.')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping.')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading.')
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_name)

    # Split the dataset into training and validation sets
    if "validation" in dataset.keys():
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    else:
        split_dataset = dataset["train"].train_test_split(test_size=0.1)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]

    # Load the tokenizer and set its maximum sequence length
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.model_max_length = args.max_seq_length  # Set tokenizer's max length

    def tokenize_function(sample):
        # Tokenize the input texts
        encoding = tokenizer(
            sample["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length  # Prevent truncation at the original max length
        )
        # Since we're dealing with single sequences, token_type_ids are zeros
        batch_size = len(sample["text"])
        encoding["token_type_ids"] = [[0] * tokenizer.model_max_length for _ in range(batch_size)]
        return encoding

    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Set the format of the datasets to PyTorch tensors
    train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    val_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

    # Create DataLoaders for training and validation
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Initialize the model
    model = LitTextClassification(model_name=args.model_name,
                                  new_max_positions=args.max_seq_length,
                                  learning_rate=args.learning_rate)

    # Define callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(
        monitor='val_recall',
        dirpath='checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        mode='max'
    )

    early_stopping = EarlyStopping('val_recall', mode='max', patience=args.patience)

    # Initialize the trainer
    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Load the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    best_model = LitTextClassification.load_from_checkpoint(best_model_path)

    # Save the fine-tuned model and tokenizer
    output_dir = args.output_dir  # Use the specified output directory
    best_model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate the model on the validation set
    val_results = trainer.validate(model=best_model, dataloaders=val_loader)
    print(val_results)

    # Get the best validation recall
    callback_metrics = trainer.callback_metrics
    if 'val_recall' in callback_metrics:
        best_val_recall = callback_metrics['val_recall'].item()
        print(f'Best Validation Recall: {best_val_recall}')
    else:
        print('Validation recall not found in callback metrics.')
