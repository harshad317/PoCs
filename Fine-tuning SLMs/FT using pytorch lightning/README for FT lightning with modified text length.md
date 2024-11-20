### **1. Ensure Python is Installed**

Make sure you have Python 3.7 or higher installed on your system. You can check your Python version by running:

```bash
python --version
```

---

### **2. Create a Virtual Environment (Optional but Recommended)**

It's a good practice to use a virtual environment to manage your Python dependencies. You can create one using `venv`:

```bash
python -m venv venv
```

Activate the virtual environment:

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

---

### **3. Install Required Packages**

The script depends on several Python packages. You can install them using `pip`. Here's how you can do it:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformers datasets pytorch-lightning torchmetrics
```

**Note:**

- The above installation of `torch` is for systems with CUDA 11.7. If you have a different CUDA version or are using CPU, adjust accordingly.
- For CPU-only installation of PyTorch:

  ```bash
  pip install torch torchvision torchaudio
  ```

---

### **4. Run the Script with Default Parameters**

To execute the script with default parameters, run:

```bash
python FT_lightning_with_modified_text_length.py.py
```

This command will start the fine-tuning process using the default settings specified in the script.

---

### **5. Run the Script with Custom Parameters**

The script uses an argument parser, allowing you to customize its behavior via command-line arguments. Here are the available arguments:

- `--model_name`: Pre-trained BERT model name (default: `'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'`)
- `--tokenizer_name`: Tokenizer name (default: `'dmis-lab/biobert-v1.1'`)
- `--dataset_name`: HuggingFace dataset name (default: `'DKTech/ICSR-data'`)
- `--output_dir`: Directory to save the fine-tuned model (default: `'saved_model_with_1024_tokens'`)
- `--max_seq_length`: Maximum sequence length (default: `1024`)
- `--batch_size`: Batch size for training (default: `8`)
- `--learning_rate`: Learning rate (default: `1e-6`)
- `--num_epochs`: Number of training epochs (default: `20`)
- `--patience`: Early stopping patience (default: `3`)
- `--num_workers`: Number of data loading workers (default: `2`)

**Example:**

To run the script with a batch size of 4 and a learning rate of 2e-5:

```bash
python FT_lightning_with_modified_text_length.py.py --batch_size 4 --learning_rate 2e-5
```

---

### **6. Detailed Steps for Running the Script**

#### **a. Adjust Parameters Based on Your Hardware**

Fine-tuning BERT models with long sequences can be memory-intensive. If you encounter out-of-memory errors, consider:

- Reducing `--batch_size`
- Reducing `--max_seq_length`

#### **b. Monitor Training Progress**

The script uses PyTorch Lightning, which provides progress bars and logging. Keep an eye on the terminal output to monitor training and validation metrics.

#### **c. Checkpointing and Early Stopping**

The script will automatically save the best model based on validation recall. It also implements early stopping to prevent overfitting.

---

### **7. After Training**

#### **a. Saved Model and Tokenizer**

The fine-tuned model and tokenizer are saved to the directory specified by `--output_dir`. By default, this is `saved_model_with_1024_tokens`.

#### **b. Evaluating the Model**

The script will evaluate the model on the validation set after training and print the results, including the best validation recall.

---

### **8. Full Example Command**

Here's a full command incorporating several custom parameters:

```bash
python FT_lightning_with_modified_text_length.py.py \
  --model_name 'bert-base-uncased' \
  --tokenizer_name 'bert-base-uncased' \
  --dataset_name 'DKTech/ICSR-data' \
  --output_dir 'my_finetuned_model' \
  --max_seq_length 512 \
  --batch_size 4 \
  --learning_rate 3e-5 \
  --num_epochs 10 \
  --patience 2 \
  --num_workers 4
```

---

### **9. Additional Considerations**

#### **a. Ensure Stable Internet Connection**

The script downloads pre-trained models and datasets from the HuggingFace hub. Make sure you have a stable internet connection during the initial run.

#### **b. GPU Usage**

Training large models is significantly faster on a GPU. If you have a GPU, PyTorch Lightning will automatically detect and utilize it. If not, training will proceed on the CPU, but it will be slower.

#### **c. Verify Dataset Availability**

If you encounter issues with the dataset `'DKTech/ICSR-data'`, ensure that it is available and you have access permissions. You can check on [HuggingFace Datasets](https://huggingface.co/datasets).

---

### **10. Example Outputs**

During training, you will see outputs similar to:

```plaintext
Epoch 0: 100%|████████| 100/100 [00:30<00:00,  3.25it/s, loss=0.693, v_num=0]
Validation: 0it [00:00, ?it/s]
Validation: 100%|████████| 25/25 [00:05<00:00,  4.67it/s]
Epoch 0, global step 100: val_recall reached 0.75 (best 0.75), saving model to "checkpoints/best-checkpoint.ckpt" as top 1
...
```

After training, the script will output the validation results and best validation recall.

---

### **11. Common Issues and Troubleshooting**

- **Out of Memory Errors:**

  - Reduce `--batch_size` or `--max_seq_length`.
  - Ensure no other heavy processes are running on your GPU.

- **Package Compatibility:**

  - If you face issues with package versions, consider specifying versions in your `pip install` commands.

    ```bash
    pip install transformers==4.31.0 datasets==2.14.0 pytorch-lightning==2.0.6
    ```

- **Dataset Not Found:**

  - Ensure that the dataset name is correct.
  - If you have a local dataset, you can modify the script to load data from your local files.

---
