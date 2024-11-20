To run your fine-tuning script, you'll need to follow several steps to ensure everything is set up correctly. Below, I've provided a step-by-step guide to help you run the code successfully:

### **1. Set Up Your Environment**

Before running the script, make sure you have all the necessary packages installed. It's a good practice to use a virtual environment to keep your project dependencies isolated.

#### **Option A: Using `venv`**

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### **Option B: Using `conda`**

```bash
# Create a new conda environment
conda create -n pytorch_lightning_env python=3.9

# Activate the environment
conda activate pytorch_lightning_env
```

### **2. Install Required Packages**

Install all the necessary Python packages using `pip`. Run the following command in your activated virtual environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets pytorch-lightning torchmetrics
```

**Note:** Replace `cu118` with your CUDA version if you're using GPU support, or use `cpu` if you're running on CPU.

### **3. Run the Script**

You can now run your script using Python. If you didn't rename the file, make sure to enclose the file name in quotes.

#### **If You Renamed the File:**

```bash
python fine_tuning_with_pytorch_lightning.py
```
### **4. Customize Script Parameters (Optional)**

Your script uses `argparse` to accept command-line arguments, allowing you to customize various aspects of the training process. You can view all the available arguments and their default values by running:

```bash
python fine_tuning_with_pytorch_lightning.py --help
```

**Output:**

```
usage: fine_tuning_with_pytorch_lightning.py [-h] [--model_name MODEL_NAME]
                                             [--dataset_name DATASET_NAME]
                                             [--max_epochs MAX_EPOCHS]
                                             [--batch_size BATCH_SIZE]
                                             [--learning_rate LEARNING_RATE]
                                             [--max_length MAX_LENGTH]
                                             [--stride STRIDE]
                                             [--patience PATIENCE]
                                             [--num_workers NUM_WORKERS]
                                             [--accelerator ACCELERATOR]
                                             [--devices DEVICES]
                                             [--output_dir OUTPUT_DIR]
                                             [--monitor_metric MONITOR_METRIC]
                                             [--monitor_mode MONITOR_MODE]

Fine-tune BERT model for text classification

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Name or path of the pre-trained model
  --dataset_name DATASET_NAME
                        Name of the Hugging Face dataset
  --max_epochs MAX_EPOCHS
                        Maximum number of training epochs
  --batch_size BATCH_SIZE
                        Batch size for training and validation
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer
  --max_length MAX_LENGTH
                        Maximum sequence length for tokenization
  --stride STRIDE       Stride size for tokenization
  --patience PATIENCE   Patience for early stopping
  --num_workers NUM_WORKERS
                        Number of worker threads for data loading
  --accelerator ACCELERATOR
                        Type of accelerator ('cpu', 'gpu')
  --devices DEVICES     Number of devices to use for training
  --output_dir OUTPUT_DIR
                        Directory to save the fine-tuned model
  --monitor_metric MONITOR_METRIC
                        Metric to monitor for checkpointing
  --monitor_mode MONITOR_MODE
                        Mode for monitoring metric ('min' or 'max')
```

#### **Running the Script with Custom Parameters**

For example, if you want to change the batch size to 16 and use a different dataset, you can run:

```bash
python fine_tuning_with_pytorch_lightning.py --batch_size 16 --dataset_name your-dataset-name
```

Replace `your-dataset-name` with the actual name of the dataset you wish to use.

### **5. Monitor the Training Process**

The script will start training the model, and you should see output in the console showing the training progress, including metrics like loss, precision, recall, accuracy, and F1 score for both training and validation.

### **6. After Training Completion**

Once training is complete:

- The best model checkpoint will be saved in the `checkpoints` directory.
- The fine-tuned model and tokenizer will be saved in the directory specified by `--output_dir` (default is `saved_model`).
- Validation results will be printed to the console.

### **Additional Tips**

- **GPU Support:** If you're training on a machine with a GPU, make sure `accelerator` is set to `'gpu'` and that PyTorch is installed with CUDA support.

- **CUDA Toolkit:** Ensure that the CUDA toolkit compatible with your GPU is installed on your system.

- **Batch Size:** Adjust the `--batch_size` parameter based on your GPU memory. If you encounter out-of-memory errors, try reducing the batch size.

- **Resume Training:** If you need to resume training from a checkpoint, you can modify the script to load from the existing checkpoint.

### **Example Command**

Here's an example of running the script with custom parameters:

```bash
python fine_tuning_with_pytorch_lightning.py \
    --model_name "bert-base-uncased" \
    --dataset_name "imdb" \
    --max_epochs 5 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --max_length 256 \
    --stride 64 \
    --patience 2 \
    --num_workers 4 \
    --accelerator "gpu" \
    --devices 1 \
    --output_dir "my_saved_model" \
    --monitor_metric "val_accuracy" \
    --monitor_mode "max"
```

**Explanation of the Parameters:**

- `--model_name`: Specifies the pre-trained model to fine-tune.
- `--dataset_name`: The Hugging Face dataset to use.
- `--max_epochs`: Number of epochs to train.
- `--batch_size`: Number of samples per batch.
- `--learning_rate`: Learning rate for the optimizer.
- `--max_length`: Maximum sequence length for inputs.
- `--stride`: How much overlap to have between tokens when tokenizing documents longer than `max_length`.
- `--patience`: Number of epochs with no improvement after which training will be stopped.
- `--num_workers`: Number of subprocesses for data loading.
- `--accelerator`: Type of device to use (`'cpu'` or `'gpu'`).
- `--devices`: Number of devices (GPUs) to use.
- `--output_dir`: Directory where the fine-tuned model and tokenizer will be saved.
- `--monitor_metric`: Metric to monitor for saving the best model.
- `--monitor_mode`: Whether to look for an increasing (`'max'`) or decreasing (`'min'`) metric.

### **Troubleshooting**

- **ModuleNotFoundError:** If you get errors about missing modules, double-check that you've installed all required packages in your virtual environment.

  ```bash
  pip install torch transformers datasets pytorch-lightning torchmetrics
  ```

- **CUDA Errors:** If you encounter CUDA-related errors, ensure that:

  - You have the correct version of CUDA installed.
  - PyTorch is installed with CUDA support compatible with your system.

- **Out of Memory Errors:** Reduce the batch size using the `--batch_size` argument.

- **Dataset Errors:** Ensure that the dataset name provided is available on the Hugging Face Hub or accessible locally.
