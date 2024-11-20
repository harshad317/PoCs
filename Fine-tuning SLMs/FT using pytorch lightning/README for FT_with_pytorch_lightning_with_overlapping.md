**How to Run Your Script: `FT_with_pytorch_lightning_with_overlapping.py`**

Running your script involves several steps, including setting up the environment, installing necessary dependencies, and executing the script with appropriate command-line arguments.

Below is a step-by-step guide to help you run your script successfully.

---

### **1. Prerequisites**

Before running the script, ensure that you have the following:

- **Python 3.6 or higher:** Your script requires Python 3.6+ due to dependency requirements.
- **Pip:** The Python package installer should be available to install necessary packages.
- **Virtual Environment (Optional but Recommended):** Using a virtual environment helps manage dependencies without affecting the global Python installation.

### **2. Required Python Packages**

Your script depends on several Python libraries:

- `torch` (PyTorch)
- `transformers`
- `datasets`
- `numpy`
- `scikit-learn`
- `argparse`

### **3. Setting Up the Environment**

#### **Option A: Using a Virtual Environment**

It's recommended to create a virtual environment to manage dependencies:

```bash
# Create a virtual environment named 'env'
python -m venv env

# Activate the virtual environment
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

#### **Option B: Using Conda Environment**

Alternatively, you can use Conda:

```bash
# Create a conda environment named 'env'
conda create -n env python=3.8

# Activate the environment
conda activate env
```

### **4. Installing Dependencies**

With the environment activated, install the required packages:

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose the appropriate command based on your CUDA version)
# For CPU-only:
pip install torch torchvision torchaudio

# For CUDA 11.7:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
pip install transformers datasets scikit-learn
```

**Note:** Replace the PyTorch installation command with the one suitable for your system (CPU or GPU with the correct CUDA version). You can find the right command on the [PyTorch Get Started page](https://pytorch.org/get-started/locally/).

### **5. Saving the Script**

Ensure that your script is saved as `FT_with_pytorch_lightning_with_overlapping.py` in your desired directory.

### **6. Understanding Command-Line Arguments**

Your script uses `argparse` to accept command-line arguments for various hyperparameters and configurations.

Here are the available arguments with their default values:

- `--lr`: Learning rate (default: `2e-5`)
- `--batch_size`: Batch size (default: `16`)
- `--num_epochs`: Number of epochs (default: `3`)
- `--weight_decay`: Weight decay (default: `0.01`)
- `--lr_scheduler`: Learning rate scheduler type (default: `'linear'`)
- `--grad_accum_steps`: Gradient accumulation steps (default: `1`)
- `--warmup_ratio`: Warmup ratio (default: `0.1`)
- `--model_name`: Pre-trained model name or path (default: `'michiyasunaga/BioLinkBERT-base'`)
- `--num_labels`: Number of labels for classification (default: `2`)
- `--metric_to_use`: Metric for selecting the best model (default: `'eval_f1'`)
- `--dataset_name`: Name of the dataset to load (default: `'DKTech/ICSR-data'`)
- `--output_dir`: Directory to save outputs (default: `'./OUTPUT/testing/'`)
- `--chunk_sizes`: List of chunk sizes to try (default: `[512]`)
- `--overlap_sizes`: List of overlap sizes to try (default: `[0, 32, 64, 128]`)

### **7. Running the Script**

Navigate to the directory containing your script:

```bash
cd path_to_your_script_directory
```

#### **Basic Command**

To run the script with default parameters:

```bash
python FT_with_pytorch_lightning_with_overlapping.py
```

#### **Specifying Arguments**

You can customize the hyperparameters by specifying them as command-line arguments. Here are some examples:

**Example 1:** Change the learning rate and batch size:

```bash
python FT_with_pytorch_lightning_with_overlapping.py --lr 3e-5 --batch_size 32
```

**Example 2:** Use a different pre-trained model and increase the number of epochs:

```bash
python FT_with_pytorch_lightning_with_overlapping.py --model_name bert-base-uncased --num_epochs 5
```

**Example 3:** Specify custom chunk sizes and overlap sizes:

```bash
python FT_with_pytorch_lightning_with_overlapping.py --chunk_sizes 512 1024 --overlap_sizes 0 64 128
```

**Example 4:** Change the output directory:

```bash
python FT_with_pytorch_lightning_with_overlapping.py --output_dir ./OUTPUT/experiment1/
```

**Example 5:** Full command with multiple custom parameters:

```bash
python FT_with_pytorch_lightning_with_overlapping.py \
    --lr 1e-5 \
    --batch_size 16 \
    --num_epochs 4 \
    --weight_decay 0.1 \
    --lr_scheduler cosine \
    --grad_accum_steps 2 \
    --warmup_ratio 0.2 \
    --model_name allenai/biomed_roberta_base \
    --num_labels 2 \
    --metric_to_use eval_f1 \
    --dataset_name DKTech/ICSR-data \
    --output_dir ./OUTPUT/biomed_roberta/ \
    --chunk_sizes 512 1024 \
    --overlap_sizes 0 64 128
```

### **8. Script Execution Flow**

When you run the script, it will:

1. **Parse Command-Line Arguments:** It reads the arguments you provided or uses default values.
2. **Set Up the Model and Tokenizer:** Initializes the model and tokenizer based on the `model_name`.
3. **Load the Dataset:** Downloads and prepares the dataset specified by `dataset_name`.
4. **Preprocess the Data:** Tokenizes and chunks the data according to `chunk_sizes` and `overlap_sizes`.
5. **Train and Evaluate the Model:** For each combination of chunk size and overlap size, it trains and evaluates the model.
6. **Save the Results:** After training, it saves the evaluation metrics to `results.json` in the specified `output_dir`.

### **9. Accessing the Results**

After the script finishes running, you can find the results in the specified output directory.

**Default Location:**

```bash
./OUTPUT/testing/results.json
```

This JSON file contains the evaluation metrics for each combination of chunk size and overlap size, as well as information about the best performing combination.

### **10. Example Output**

When the script runs, it will print out the training progress and evaluation results in the console. At the end, it will also print the best combination based on the specified metric (default is F1 score).

**Sample Console Output:**

```
Training with chunk_size=512 and overlap_size=0
... [Training and evaluation logs] ...
Results for chunk_512_overlap_0: {'eval_loss': 0.45, 'eval_accuracy': 0.86, 'eval_precision': 0.84, 'eval_recall': 0.88, 'eval_f1': 0.86}

Training with chunk_size=512 and overlap_size=32
... [Training and evaluation logs] ...
Results for chunk_512_overlap_32: {'eval_loss': 0.42, 'eval_accuracy': 0.88, 'eval_precision': 0.85, 'eval_recall': 0.90, 'eval_f1': 0.88}

...

Best combination: chunk_512_overlap_32 with F1 score: 0.8800

All results:
chunk_512_overlap_0: {...}
chunk_512_overlap_32: {...}
...

Results have been saved to ./OUTPUT/testing/results.json
```

### **11. Troubleshooting**

Here are some common issues you might encounter and how to address them:

#### **a. ModuleNotFoundError**

**Issue:**

```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**

Install the missing module using pip:

```bash
pip install torch
```

#### **b. CUDA Compatibility Issues**

If you are using a GPU and encounter CUDA errors, ensure that:

- You have the appropriate CUDA toolkit installed.
- PyTorch was installed with the correct CUDA version.

You can specify the CUDA version when installing PyTorch:

```bash
# For CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

#### **c. Dataset Not Found**

**Issue:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'DKTech/ICSR-data'
```

**Solution:**

Ensure that the dataset name is correct and available on the Hugging Face Datasets Hub. You can check the dataset's availability or specify a local path if you have the dataset locally.

#### **d. Out of Memory Errors**

If you run into out-of-memory (OOM) errors during training:

- Reduce the `batch_size`.
- Reduce the `chunk_sizes`.
- Ensure that other processes are not utilizing GPU memory.

#### **e. Other Errors**

If you encounter other errors, read the error messages carefully. Common issues can often be resolved by checking for typos in argument names or ensuring all dependencies are installed.

### **12. Additional Tips**

- **Check Available GPUs:** To see if PyTorch recognizes your GPUs:

  ```python
  import torch
  print(torch.cuda.is_available())
  print(torch.cuda.device_count())
  print(torch.cuda.current_device())
  print(torch.cuda.get_device_name(0))
  ```

- **Adjusting Hyperparameters:** Experiment with different hyperparameters to improve model performance.

- **Updating Libraries:** Ensure all libraries are up to date:

  ```bash
  pip install --upgrade transformers datasets
  ```

- **Logging and Monitoring:** You can add logging or use tools like [Weights & Biases](https://www.wandb.com/) for monitoring experiments.

### **13. Visualizing Results (Optional)**

You can use additional scripts to parse `results.json` and visualize the results. For example, you could plot F1 scores against overlap sizes.

---
