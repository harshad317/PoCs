To run the `fine_tuning.py` script you've saved, please follow these steps:

---

## **Step 1: Ensure Required Packages are Installed**

Before running the script, make sure you have all the necessary Python packages installed. The script depends on several libraries including `transformers`, `datasets`, `scikit-learn`, `matplotlib`, `seaborn`, and `torch`.

Open a terminal or command prompt and run the following command:

```bash
pip install transformers datasets scikit-learn matplotlib seaborn torch
```

If you're using a **GPU** and want to utilize it for faster training, you should install the GPU-enabled version of PyTorch. Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system. For example:

```bash
# For CUDA 11.7
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

---

## **Step 2: Verify Python and Package Versions (Optional but Recommended)**

It's good practice to ensure that your Python environment is correctly set up. You can check your Python version by running:

```bash
python --version
```

Ensure you're using **Python 3.7** or higher, as this is required by some of the libraries.

---

## **Step 3: Verify Dataset Access**

The script uses the `DKTech/ICSR-data` dataset from the Hugging Face Datasets library. This dataset should be publicly accessible.

**Note:** If you encounter any issues accessing the dataset due to network restrictions or need to authenticate, you might have to set up Hugging Face CLI authentication:

```bash
huggingface-cli login
```

You'll be prompted to enter your Hugging Face API token, which you can obtain by creating an account at [Hugging Face](https://huggingface.co/join).

---

## **Step 4: Run the Script with Default Parameters**

Navigate to the directory where you saved `fine_tuning.py`:

```bash
cd /path/to/your/script/
```

Run the script using Python:

```bash
python fine_tuning.py
```

This will execute the script with the default hyperparameters specified in the script.

---

## **Step 5: Customize Hyperparameters (Optional)**

If you want to experiment with different hyperparameters, you can specify them using command-line arguments. Here's how you can do it:

```bash
python fine_tuning.py \
    --learning_rate 3e-5 \
    --batch_size 32 \
    --num_epochs 5 \
    --weight_decay 0.1 \
    --lr_scheduler linear \
    --grad_accum_steps 1 \
    --warmup_ratio 0.1 \
    --seed 42 \
    --max_length 512 \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --output_dir ./OUTPUT/ \
    --save_total_limit 3
```

### **Available Command-Line Arguments**

- `--learning_rate`: Learning rate for the optimizer (default: `2e-5`).
- `--batch_size`: Batch size per device during training and evaluation (default: `16`).
- `--num_epochs`: Total number of training epochs (default: `3`).
- `--weight_decay`: Weight decay for the optimizer (default: `0.01`).
- `--lr_scheduler`: Learning rate scheduler type (e.g., `linear`, `cosine`; default: `linear`).
- `--grad_accum_steps`: Number of gradient accumulation steps (default: `1`).
- `--warmup_ratio`: Warmup ratio for the learning rate scheduler (default: `0.1`).
- `--seed`: Random seed for reproducibility (default: `42`).
- `--max_length`: Maximum sequence length for tokenization (default: `512`).
- `--model_name`: Pre-trained model name or path (default: `'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'`).
- `--output_dir`: Directory to save model checkpoints and outputs (default: `'./OUTPUT/'`).
- `--save_total_limit`: Maximum number of checkpoints to keep (default: `3`).

---

## **Step 6: Monitor Training Progress**

As the script runs, it will output training and evaluation metrics after each epoch. You can monitor metrics like loss, accuracy, precision, recall, and F1 score directly in the console.

---

## **Step 7: View the Classification Report and Confusion Matrix**

After training completes, the script will evaluate the model on the test set and:

- **Print the classification report**: This includes precision, recall, F1-score, and support for each class.
- **Display the confusion matrix**: A visual representation of the performance of the classification model.

**Note:** The confusion matrix will be displayed using Matplotlib. Ensure that your environment supports GUI display if you're running the script on a local machine.

If you're running the script on a remote server or via SSH and cannot display plots, you can save the confusion matrix to a file by modifying the last part of the script:

```python
# Instead of plt.show(), save the figure
plt.savefig('confusion_matrix.png')
```

---

## **Troubleshooting Common Issues**

- **Module Not Found Errors**: If you encounter errors like `ModuleNotFoundError: No module named 'some_module'`, ensure that all required packages are installed in your Python environment.
- **CUDA/Device Errors**: If you have a GPU but encounter CUDA-related errors, ensure that the correct version of PyTorch with CUDA support is installed and that your GPU drivers are up to date.
- **Memory Errors**: If your system runs out of memory (OOM), try reducing the `--batch_size` or `--max_length` parameters.

---

## **Example Commands**

### **Running with Default Parameters**

```bash
python fine_tuning.py
```

### **Running with Custom Hyperparameters**

```bash
python fine_tuning.py \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --num_epochs 4 \
    --max_length 256 \
    --model_name bert-base-uncased \
    --output_dir ./bert_output/
```

This command fine-tunes the `bert-base-uncased` model with a learning rate of `5e-5`, batch size of `8`, for `4` epochs, with a maximum sequence length of `256`, and saves outputs to `./bert_output/`.

---

## **Additional Tips**

- **Using a Virtual Environment**: It's a good idea to run your Python scripts inside a virtual environment to manage dependencies.

  ```bash
  # Create a virtual environment
  python -m venv env

  # Activate the virtual environment (Windows)
  .\env\Scripts\activate

  # Activate the virtual environment (Unix or Linux)
  source env/bin/activate

  # Install required packages
  pip install transformers datasets scikit-learn matplotlib seaborn torch
  ```

- **GPU Utilization**: If you have access to a GPU, the script will automatically utilize it via PyTorch. Training on a GPU significantly speeds up the training process.

- **Modifying the Script**: Feel free to modify the script to suit your needs, such as changing the dataset, adding more evaluation metrics, or integrating with other tools.

---
