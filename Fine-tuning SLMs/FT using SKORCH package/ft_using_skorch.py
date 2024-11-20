import numpy as np  # Import NumPy for numerical operations
import torch  # Import PyTorch for building and training neural networks

from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, classification_report
# Import evaluation metrics from scikit-learn

from skorch import NeuralNetClassifier  # Import NeuralNetClassifier from skorch for scikit-learn compatibility
from skorch.callbacks import LRScheduler, ProgressBar, EpochScoring  # Import callbacks for training
from skorch.hf import HuggingfacePretrainedTokenizer  # Import tokenizer for HuggingFace models
from skorch.dataset import Dataset  # Import Dataset for handling data
from skorch.helper import predefined_split  # Import predefined_split for validation data
from skorch.callbacks import EarlyStopping  # Import EarlyStopping for stopping training early

from torch import nn  # Import neural network modules from PyTorch
from torch.optim.lr_scheduler import LambdaLR  # Import learning rate scheduler
from transformers import AutoModelForSequenceClassification  # Import pre-trained model for sequence classification
from datasets import load_dataset  # Import function to load datasets from HuggingFace
import pandas as pd  # Import pandas for data manipulation

# Define constants for model configuration
TOKENIZER = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"  # Name of the tokenizer to use
PRETRAINED_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"  # Name of the pre-trained model to use
OPTIMIZER = torch.optim.AdamW  # Optimizer for training
LR = 3.5e-6  # Learning rate
MAX_EPOCHS = 20  # Maximum number of epochs to train
CRITERION = nn.CrossEntropyLoss  # Loss function
BATCH_SIZE = 50  # Batch size for training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, else CPU

def loading_data():
    """
    Function to load datasets from HuggingFace and shuffle the training data.
    """
    dataset = load_dataset("DKTech/ICSR-data")  # Load the dataset
    dataset_train = dataset['train']  # Training data
    dataset_valid = dataset['validation']  # Validation data
    dataset_test = dataset['test']  # Test data
    dataset_train = dataset_train.shuffle(seed=42)  # Shuffle training data
    return dataset_train, dataset_valid, dataset_test

# Load datasets
dataset_train, dataset_valid, dataset_test = loading_data()

# Convert datasets to pandas DataFrames
train = dataset_train.to_pandas()
valid = dataset_valid.to_pandas()
test = dataset_test.to_pandas()

# Extract features and labels for training, validation, and test sets
X_train = train['text']
y_train = train['label']
X_valid = valid['text']
y_valid = valid['label']
X_test = test['text']
y_test = test['label']

# Calculate the total number of training steps
num_training_steps = MAX_EPOCHS * (len(X_train) // BATCH_SIZE + 1)

def lr_schedule(current_step):
    """
    Learning rate scheduler function that decreases the learning rate linearly.
    """
    factor = float(num_training_steps - current_step) / float(max(1, num_training_steps))
    assert factor > 0  # Ensure the factor is positive
    return factor

class BertModule(nn.Module):
    """
    Neural network module that wraps a pre-trained BERT model for sequence classification.
    """
    def __init__(self, name, num_labels):
        super().__init__()
        self.name = name  # Name of the pre-trained model
        self.num_labels = num_labels  # Number of output labels
        self.reset_weights()  # Initialize weights

    def reset_weights(self):
        # Load the pre-trained model with the specified number of labels
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            self.name, num_labels=self.num_labels
        )

    def forward(self, **kwargs):
        # Define the forward pass
        pred = self.bert(**kwargs)
        return pred.logits  # Return the raw logits

# Define scoring callbacks for precision, recall, and F1 score
precision = EpochScoring(
    make_scorer(precision_score, average='macro'),
    name='valid_precision',
    lower_is_better=False,
    on_train=False,
    use_caching=False,
)

recall = EpochScoring(
    make_scorer(recall_score, average='macro'),
    name='valid_recall',
    lower_is_better=False,
    on_train=False,
    use_caching=False,
)

f1 = EpochScoring(
    make_scorer(f1_score, average='macro'),
    name='valid_f1',
    lower_is_better=False,
    on_train=False,
    use_caching=False,
)

# Initialize the tokenizer with the pre-trained tokenizer
tokenizer = HuggingfacePretrainedTokenizer(TOKENIZER)

# Tokenize and encode the datasets
X_train_transformed = tokenizer.fit_transform(X_train)
X_valid_transformed = tokenizer.transform(X_valid)
X_test_transformed = tokenizer.transform(X_test)

# Create a validation dataset
valid_ds = Dataset(X_valid_transformed, y_valid)

# Define EarlyStopping callback to prevent overfitting
early_stopping = EarlyStopping(
    monitor='valid_recall',  # Monitor validation recall
    patience=3,  # Number of epochs to wait before stopping
    lower_is_better=False,  # Higher recall is better
)

# Define the neural network classifier using skorch
net = NeuralNetClassifier(
    BertModule,  # Neural network module
    module__name=PRETRAINED_MODEL,  # Name of the pre-trained model
    module__num_labels=len(set(y_train)),  # Number of unique labels
    optimizer=OPTIMIZER,  # Optimizer for training
    lr=LR,  # Learning rate
    max_epochs=MAX_EPOCHS,  # Maximum number of epochs
    criterion=CRITERION,  # Loss function
    batch_size=BATCH_SIZE,  # Batch size
    iterator_train__shuffle=True,  # Shuffle training data each epoch
    device=DEVICE,  # Device to train on (CPU or GPU)
    train_split=predefined_split(valid_ds),  # Use predefined validation split
    callbacks=[
        LRScheduler(LambdaLR, lr_lambda=lr_schedule, step_every='batch'),  # Learning rate scheduler
        ProgressBar(),  # Display progress bar during training
        precision,  # Calculate precision each epoch
        recall,  # Calculate recall each epoch
        f1,  # Calculate F1 score each epoch
        early_stopping  # Early stopping callback
    ],
)

# Set random seeds for reproducibility
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
np.random.seed(0)

# Train the model
net.fit(X_train_transformed, y_train)

# Make predictions on the test set
pred = net.predict(X_test_transformed)

# Print the classification report for the test set
print(classification_report(y_test, pred))
