## Hyperparameter tuning

Use this python script to find the best hyperparameters to fine tune any SLMs.

## How to use this script:

### **Install Dependencies**

Install the required packages using `pip`.

```bash
pip install transformers datasets torch optuna scikit-learn matplotlib seaborn
```

**Notes**:

- **PyTorch Installation**: The `torch` package installed via `pip` in the above command is CPU-only by default. If you have a CUDA-compatible GPU and want to leverage it for acceleration, you need to install the GPU version of PyTorch.

  Find the right command for your system on the [PyTorch website](https://pytorch.org/). For example:

  ```bash
  # For CUDA 11.7
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
  ```

- **Check Installations**: Verify that all packages are installed correctly.

  ```bash
  pip list
  ```

---

#### Run the Script from the Terminal

Now that everything is set up, you can run the script using the Python interpreter.

### **Navigate to the Script's Directory**

Open your terminal or command prompt and navigate to the directory where you saved `tuning.py`.

```bash
cd /path/to/your/project
```

### **Run the Script**

You can run the script with default settings or customize it using command-line arguments.

#### **View Help Information**

To see all available command-line arguments and their descriptions, run:

```bash
python tuning.py --help
```

This command will display something like:

```
usage: tuning.py [-h] [--model_name MODEL_NAME] [--labels LABELS]
                      [--metric_to_use METRIC_TO_USE] [--num_train_epochs NUM_TRAIN_EPOCHS]
                      [--n_trials N_TRIALS] [--seed SEED] [--dataset_name DATASET_NAME]
                      [--max_length MAX_LENGTH] [--stride STRIDE]
                      [--output_dir OUTPUT_DIR] [--early_stopping_patience EARLY_STOPPING_PATIENCE]
                      [--logging_steps LOGGING_STEPS] [--fp16] [--overwrite_output_dir]

Train a text classification model with hyperparameter tuning.

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Name of the pretrained model.
  --labels LABELS       Number of classes.
  --metric_to_use METRIC_TO_USE
                        Metric to optimize during hyperparameter tuning.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of training epochs.
  --n_trials N_TRIALS   Number of trials for hyperparameter tuning.
  --seed SEED           Random seed for reproducibility.
  --dataset_name DATASET_NAME
                        Dataset name.
  --max_length MAX_LENGTH
                        Maximum sequence length.
  --stride STRIDE       Tokenization stride.
  --output_dir OUTPUT_DIR
                        Output directory for model.
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Early stopping patience.
  --logging_steps LOGGING_STEPS
                        Logging steps.
  --fp16                Whether to use fp16 precision.
  --overwrite_output_dir
                        Whether to overwrite the output directory.
```

#### **Run with Default Settings**

To run the script with all default parameters:

```bash
python tuning.py
```

This command:

- Uses the default model (`"dmis-lab/biobert-v1.1"`).
- Performs hyperparameter tuning with 30 trials.
- Trains for 1 epoch.
- Uses the default dataset (`"DKTech/ICSR-data"`).

#### **Customize the Script with Arguments**

You can customize various aspects of the training process by providing arguments. For instance:

```bash
python tuning.py \
    --model_name "bert-base-uncased" \
    --num_train_epochs 3 \
    --n_trials 50 \
    --dataset_name "imdb" \
    --output_dir "./model_output/" \
    --max_length 256 \
    --stride 128 \
    --early_stopping_patience 3 \
    --fp16 \
    --overwrite_output_dir
```

**Explanation of the Arguments**:

- `--model_name`: Specify the pre-trained model. Example: `"bert-base-uncased"`.
- `--num_train_epochs`: Number of epochs to train the model. Example: `3`.
- `--n_trials`: Number of trials for hyperparameter tuning with Optuna. Example: `50`.
- `--dataset_name`: Name of the dataset to load. Must be available via the `datasets` library.
- `--output_dir`: Directory where the model outputs and checkpoints will be saved.
- `--max_length`: Maximum sequence length for tokenization. Example: `256`.
- `--stride`: Number of tokens to shift for the next segment during tokenization. Example: `128`.
- `--early_stopping_patience`: Number of evaluation steps with no improvement after which training will be stopped. Example: `3`.
- `--fp16`: Use this flag to enable mixed precision training (FP16). Requires appropriate hardware (GPU with FP16 support).
- `--overwrite_output_dir`: If this flag is set, the contents of the output directory will be overwritten.
