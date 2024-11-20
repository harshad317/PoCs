To run your `layerwise_ft.py` script for fine-tuning BioMedBERT with layer freezing and gradual unfreezing, you'll need to follow several steps to set up your environment, install the necessary dependencies, and execute the script with the appropriate command-line arguments. Below is a detailed guide to help you get started.

---

## **Step 1: Set Up Your Python Environment**

It's highly recommended to use a virtual environment to manage your Python packages and dependencies without affecting your system-wide Python installation.

### **Option 1: Using `venv` (Built-in Virtual Environment)**

1. **Create a Virtual Environment**

   ```bash
   python3 -m venv bert_env
   ```

2. **Activate the Virtual Environment**

   - **On Linux/MacOS:**

     ```bash
     source bert_env/bin/activate
     ```

   - **On Windows:**

     ```bash
     bert_env\Scripts\activate
     ```

### **Option 2: Using Conda (If Installed)**

1. **Create a Conda Environment**

   ```bash
   conda create --name bert_env python=3.9
   ```

2. **Activate the Conda Environment**

   ```bash
   conda activate bert_env
   ```

---

## **Step 2: Install Necessary Packages**

With your virtual environment activated, install the required Python packages. Ensure you have an internet connection as packages will be downloaded.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers
pip install datasets
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

**Note:**

- The `torch` installation command above assumes you have CUDA 11.7 installed. If you have a different version of CUDA or are using CPU-only, adjust the command accordingly. Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for specific instructions.

- It's important to install `torch` before `transformers` to ensure compatibility.

---

## **Step 3: Verify GPU Support (Optional but Recommended)**

Training large models like BERT can be resource-intensive. If you have a CUDA-compatible GPU, you can utilize it to accelerate training.

1. **Verify CUDA is Available**

   ```python
   import torch
   print(torch.cuda.is_available())
   ```

   If this prints `True`, your environment recognizes the GPU.

2. **Check CUDA Version**

   Ensure your CUDA version matches the version of PyTorch you installed.

---

## **Step 4: Save the Script**

Copy your script into a file named `layerwise_ft.py` in your working directory.

---

## **Step 5: Run the Script**

You can run the script with default arguments or specify your own.

### **Option 1: Run with Default Arguments**

Simply execute:

```bash
python layerwise_ft.py
```

### **Option 2: View Available Arguments**

To see all available command-line arguments and their defaults, run:

```bash
python layerwise_ft.py --help
```

**Example Output:**

```
usage: layerwise_ft.py [-h] [--learning_rate LEARNING_RATE]
                       [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                       [--weight_decay WEIGHT_DECAY]
                       [--lr_scheduler LR_SCHEDULER]
                       [--grad_accum_steps GRAD_ACCUM_STEPS]
                       [--warmup_ratio WARMUP_RATIO]
                       [--output_dir OUTPUT_DIR] [--max_length MAX_LENGTH]
                       [--model_name MODEL_NAME] [--labels LABELS]
                       [--metric METRIC] [--num_layers NUM_LAYERS]
                       [--early_stopping_patience EARLY_STOPPING_PATIENCE]
                       [--seed SEED]

Fine-tune BioMedBERT with layer freezing and gradual unfreezing.

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
                        Learning rate (default: 2e-5)
  --batch_size BATCH_SIZE
                        Batch size (default: 16)
  --num_epochs NUM_EPOCHS
                        Number of epochs (default: 3)
  --weight_decay WEIGHT_DECAY
                        Weight decay (default: 0.01)
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler (default: linear)
  --grad_accum_steps GRAD_ACCUM_STEPS
                        Gradient accumulation steps (default: 1)
  --warmup_ratio WARMUP_RATIO
                        Warmup ratio (default: 0.1)
  --output_dir OUTPUT_DIR
                        Output directory for model checkpoints (default:
                        ./OUTPUT/)
  --max_length MAX_LENGTH
                        Maximum sequence length (default: 1024)
  --model_name MODEL_NAME
                        Model name or path (default:
                        microsoft/BiomedNLP-PubMedBERT-base-uncased-
                        abstract)
  --labels LABELS       Number of labels for classification (default: 2)
  --metric METRIC       Metric to use for model evaluation (default: recall)
  --num_layers NUM_LAYERS
                        Number of layers in the model (default: 12)
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Early stopping patience (default: 3)
  --seed SEED           Random seed (default: 42)
```

### **Option 3: Run with Custom Arguments**

Customize the training by specifying parameters.

**Example Command:**

```bash
python layerwise_ft.py \
  --learning_rate 5e-5 \
  --batch_size 8 \
  --num_epochs 5 \
  --max_length 512 \
  --output_dir ./my_model_output \
  --metric f1 \
  --seed 123
```

---

## **Step 6: Understanding the Dataset**

Your script uses the dataset `DKTech/ICSR-data` from the Hugging Face Hub.

- This dataset will be automatically downloaded when you run the script for the first time.
- Ensure you have a stable internet connection and sufficient disk space.

If you encounter issues with data loading, verify the dataset exists or specify an alternative dataset.

---

## **Step 7: Monitoring the Training Process**

The script is designed to output training progress and results, including:

- Metrics after each epoch.
- A classification report on the test set.
- A confusion matrix plotted using Matplotlib and Seaborn.

**Note:**

- The confusion matrix will display in a new window if running locally with GUI support.
- If you're running on a server without display capabilities, you might need to adjust the plotting backend or save the plot to a file.

---

## **Step 8: Accessing Output and Logs**

The model checkpoints, logs, and outputs are saved in the directory specified by the `--output_dir` argument (default is `./OUTPUT/`).

- Inspect this directory to access:
  - Model checkpoints.
  - Training logs.
  - Configuration files.

---

## **Step 9: Troubleshooting Common Issues**

### **Issue 1: Out-Of-Memory Errors**

- **Solution:**
  - Reduce the `--batch_size`, e.g., set it to 8 or even 4.
  - Reduce `--max_length` if your data permits.
  - Disable mixed-precision training by editing the script and setting `fp16=False` in the `TrainingArguments`.

### **Issue 2: Slow Training**

- **Solution:**
  - Ensure you're using a GPU. Training on CPU can be significantly slower.
  - Verify that PyTorch is utilizing the GPU with:

    ```python
    import torch
    print(torch.cuda.is_available())
    ```

### **Issue 3: Dataset Not Found**

- **Solution:**
  - Verify the dataset name is correct in the `loading_data` function.
  - Check online if `DKTech/ICSR-data` is publicly available or requires authentication.

### **Issue 4: Display Issues with Plots**

- **Solution:**
  - If running on a headless server, save plots to files instead of displaying them:

    Modify the `plt.show()` line to save the figure:

    ```python
    plt.savefig('confusion_matrix.png')
    ```

---

## **Step 10: Customizing the Script (Optional)**

If you wish to modify the script for advanced use cases:

- **Change the Model:**

  Replace the `model_name` in the `parse_arguments` function or pass a different `--model_name` argument.

- **Use a Different Dataset:**

  Modify the `loading_data()` function to load a different dataset or your own data.

- **Adjust the Gradual Unfreezing Behavior:**

  Tweak the `GradualUnfreezingCallback` class to change how layers are unfrozen during training.

- **Modify Metrics:**

  Adjust the `compute_metrics` function to change how evaluation metrics are calculated.

---

## **Example: Full Command to Run the Script**

```bash
python layerwise_ft.py \
  --learning_rate 3e-5 \
  --batch_size 8 \
  --num_epochs 4 \
  --weight_decay 0.01 \
  --lr_scheduler linear \
  --grad_accum_steps 2 \
  --warmup_ratio 0.1 \
  --output_dir ./OUTPUT/ \
  --max_length 512 \
  --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --labels 2 \
  --metric f1 \
  --num_layers 12 \
  --early_stopping_patience 2 \
  --seed 42
```

---

## **Additional Tips**

- **Regularly Save Your Work:**

  Ensure you have backups of your script and any changes you make.

- **Documentation:**

  Refer to the official documentation for libraries used:

  - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
  - [Transformers Documentation](https://huggingface.co/docs/transformers/index)
  - [Datasets Documentation](https://huggingface.co/docs/datasets/index)
  - [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)

- **Experimentation:**

  Try different hyperparameter values to see how they affect model performance.

- **Reproducibility:**

  Set the `--seed` argument to ensure reproducible results across runs.

---
