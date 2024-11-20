Here's a comprehensive guide to help you run your code smoothly:

---

### **1. Install Python 3**

First, ensure that you have Python 3 installed on your machine. You can check your Python version by running:

```bash
python --version
```

or

```bash
python3 --version
```

If you don't have Python 3 installed, download and install it from the [official Python website](https://www.python.org/downloads/) or use a package manager appropriate for your operating system.

---

### **2. Set Up a Virtual Environment (Optional but Recommended)**

Using a virtual environment isolates your project and helps manage dependencies without affecting other projects or system-wide packages.

**Create a virtual environment:**

```bash
python3 -m venv env
```

**Activate the virtual environment:**

- On macOS/Linux:

  ```bash
  source env/bin/activate
  ```

- On Windows:

  ```bash
  env\Scripts\activate
  ```

---

### **3. Install Required Packages**

Your script depends on several Python packages. Install them using `pip`. Make sure your `pip` is up to date:

```bash
pip install --upgrade pip
```

**Install packages:**

```bash
pip install numpy pandas scikit-learn transformers datasets skorch
```

**Install PyTorch:**

PyTorch installation varies depending on your system and whether you have a CUDA-capable GPU.

- If **you have a GPU** and want to use CUDA (recommended for training models like BERT):

  Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system and CUDA version.

  For example, if you have CUDA 11.8:

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- If **you do not have a GPU** or don't want to use CUDA:

  ```bash
  pip install torch torchvision torchaudio
  ```

**Note:** Using a GPU significantly speeds up the training process for deep learning models.

---

### **4. Ensure Internet Access**

Your script downloads datasets and pre-trained models from Hugging Face's repositories. Make sure your internet connection is stable and that your network allows downloading from external sources.

---

### **5. Save Your Script**

Ensure your script (`ft_using_skorch.py`) is saved in your current working directory. Here’s a brief reminder of what your script does:

- Imports necessary libraries
- Loads the "DKTech/ICSR-data" dataset from Hugging Face
- Fine-tunes a BERT model for sequence classification using `skorch`
- Evaluates the model and prints classification metrics

---

### **6. Run the Script**

With all dependencies installed and your virtual environment activated, run your script:

```bash
python ft_using_skorch.py
```

---

### **7. Understand the Training Process**

**Training Time:**

- Training BERT models can be time-consuming, especially without a GPU. Be prepared for the script to run for an extended period.
- Monitor system resources to ensure your machine can handle the training process.

**Adjusting Batch Size:**

- The default `BATCH_SIZE` is set to 50. If you encounter memory issues, consider reducing the batch size:

  ```python
  BATCH_SIZE = 16  # Or a smaller number depending on your system's capabilities
  ```

---

### **8. Expected Output**

If everything runs correctly, the script will:

- Display progress bars and training metrics for each epoch, thanks to the `ProgressBar` callback.
- Print the classification report on the test set at the end of training, showing precision, recall, F1-score, and support for each class.

Example output:

```
              precision    recall  f1-score   support

           0       0.85      0.87      0.86       100
           1       0.80      0.78      0.79        50

    accuracy                           0.83       150
   macro avg       0.82      0.82      0.82       150
weighted avg       0.83      0.83      0.83       150
```

---

### **9. Troubleshooting Common Issues**

**ModuleNotFoundError:**

- Ensure all packages are installed in the correct environment. Activate your virtual environment before installing packages and running the script.

**CUDA Errors:**

- If you have a GPU but encounter CUDA errors, verify that your CUDA drivers are correctly installed and that PyTorch is set up to use CUDA.
- Check your CUDA version:

  ```bash
  nvcc --version
  ```

- Ensure compatibility between your CUDA version and the PyTorch binaries.

**Memory Errors:**

- Reduce the batch size if you run into `OutOfMemory` errors.
- Close other applications to free up system memory.

**Dataset Loading Errors:**

- If loading the dataset fails, check your internet connection.
- Update the `datasets` library to the latest version:

  ```bash
  pip install --upgrade datasets
  ```

**Version Conflicts:**

- Use the latest compatible versions of libraries.
- If you encounter issues, consider specifying versions in your `pip` install commands. For example:

  ```bash
  pip install torch==2.0.1 transformers==4.32.1
  ```

---

### **10. Additional Tips**

**Monitoring Training:**

- You can customize the callbacks to log more metrics or adjust verbosity.
- Use tools like `nvidia-smi` (on systems with NVIDIA GPUs) to monitor GPU usage.

**Experiment Tracking:**

- Consider using experiment tracking tools like TensorBoard or Weights & Biases to visualize training progress.

**Code Customization:**

- Feel free to adjust hyperparameters like `LR`, `MAX_EPOCHS`, and model architecture to better suit your dataset or improve performance.

---
