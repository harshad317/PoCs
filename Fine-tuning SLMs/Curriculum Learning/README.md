---

## **Running the Script**

1. **Open a Terminal or Command Prompt:**

   Navigate to the directory where you saved `curriculum_learning_ft.py`.

2. **Basic Execution with Default Parameters:**

   Simply run:

   ```bash
   python curriculum_learning_ft.py
   ```

   This will execute the script using the default hyperparameters specified in the script.

---

## **Specifying Command-Line Arguments**

The script accepts several command-line arguments that allow you to customize the training process. Here's a list of the available arguments and their default values:

- `--lr`: Learning rate (default: `2e-5`)
- `--batch_size`: Batch size (default: `16`)
- `--num_epochs`: Number of epochs (default: `3`)
- `--weight_decay`: Weight decay (default: `0.01`)
- `--lr_scheduler`: Learning rate scheduler (default: `'linear'`)
- `--grad_accum_steps`: Gradient accumulation steps (default: `1`)
- `--warmup_ratio`: Warmup ratio (default: `0.1`)
- `--model_name`: Model name or path (default: `'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'`)
- `--max_length`: Maximum sequence length (default: `1024`)

### **Example Usages:**

1. **Specify a Different Learning Rate and Batch Size:**

   ```bash
   python curriculum_learning_ft.py --lr 3e-5 --batch_size 32
   ```

2. **Change the Number of Epochs and Learning Rate Scheduler:**

   ```bash
   python curriculum_learning_ft.py --num_epochs 5 --lr_scheduler cosine
   ```

3. **Use a Different Pre-trained Model:**

   ```bash
   python curriculum_learning_ft.py --model_name bert-base-uncased
   ```

4. **Adjust Maximum Sequence Length:**

   ```bash
   python curriculum_learning_ft.py --max_length 512
   ```

5. **Combine Multiple Arguments:**

   ```bash
   python curriculum_learning_ft.py --lr 1e-5 --batch_size 8 --num_epochs 4 --model_name distilbert-base-uncased --max_length 256
   ```

### **Viewing Help and Available Arguments:**

You can display a help message that lists all available arguments and their descriptions by running:

```bash
python curriculum_learning_ft.py --help
```

This will output:

```
usage: curriculum_learning_ft.py [-h] [--lr LR] [--batch_size BATCH_SIZE]
                                 [--num_epochs NUM_EPOCHS]
                                 [--weight_decay WEIGHT_DECAY]
                                 [--lr_scheduler LR_SCHEDULER]
                                 [--grad_accum_steps GRAD_ACCUM_STEPS]
                                 [--warmup_ratio WARMUP_RATIO]
                                 [--model_name MODEL_NAME]
                                 [--max_length MAX_LENGTH]

Train a model for text classification

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate
  --batch_size BATCH_SIZE
                        Batch size
  --num_epochs NUM_EPOCHS
                        Number of epochs
  --weight_decay WEIGHT_DECAY
                        Weight decay
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler
  --grad_accum_steps GRAD_ACCUM_STEPS
                        Gradient accumulation steps
  --warmup_ratio WARMUP_RATIO
                        Warmup ratio
  --model_name MODEL_NAME
                        Model name or path
  --max_length MAX_LENGTH
                        Maximum sequence length
```

---

## **Detailed Steps**

1. **Ensure You Have Required Dependencies:**

   Before running the script, make sure you have all the necessary Python packages installed:

   ```bash
   pip install transformers datasets scikit-learn matplotlib seaborn torch
   ```

   If you're using a requirements file, it might look like this:

   ```bash
   pip install -r requirements.txt
   ```

   Where `requirements.txt` contains:

   ```
   transformers
   datasets
   scikit-learn
   matplotlib
   seaborn
   torch
   ```

2. **Run the Script with Desired Arguments:**

   Use the `python` command followed by the script name and any desired arguments.

   **Example with Custom Hyperparameters:**

   ```bash
   python curriculum_learning_ft.py --lr 5e-5 --batch_size 8 --num_epochs 5 --model_name bert-base-uncased --max_length 512
   ```

3. **Understand the Output:**

   The script will start executing and display various logs, including:

   - Training progress bars
   - Evaluation metrics after each epoch
   - Saving of models if a new best model is found
   - Final classification report and confusion matrix

4. **Visualizing the Confusion Matrix:**

   The script generates and displays a confusion matrix using `matplotlib` and `seaborn`.

   - **If You're Running Locally with a Display:**

     The confusion matrix will pop up in a new window after training is complete.

   - **If You're Running Remotely or On a Server Without Display:**

     You might encounter an error because there's no display environment.

     - **Solution:** Modify the plotting section in the script to save the plot instead of displaying it.

       ```python
       # Replace plt.show() with:
       plt.savefig('confusion_matrix.png')
       ```

     This will save the confusion matrix as an image file in the current directory.

---

## **Additional Notes**

- **Running on GPU:**

  The script will automatically utilize available GPUs via PyTorch for faster training.

  Ensure that your PyTorch installation is configured for GPU support.

- **Adjusting Model-Specific Code:**

  The script includes model-specific code related to BERT's position embeddings, particularly when adjusting `max_length`.

  **If You Use a Different Model Architecture**, such as RoBERTa or DistilBERT, you may need to adjust the code that modifies `model.bert.embeddings`.

  For example, for RoBERTa models, the embeddings might be accessed via `model.roberta.embeddings`.

- **Early Stopping and Callbacks:**

  The script uses the `EarlyStoppingCallback` to prevent overfitting by stopping training if the model's performance doesn't improve for 3 consecutive evaluation steps.

  If you want to adjust this behavior:

  ```python
  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
  ```

- **Logging and Output Directory:**

  The script saves outputs to `./OUTPUT/` and logs to `./logs/`. Ensure you have write permissions to these directories or adjust the paths accordingly in the `TrainingArguments`.

---

## **Example: Full Command**

Here's an example of running the script with several custom parameters:

```bash
python curriculum_learning_ft.py \
  --lr 3e-5 \
  --batch_size 16 \
  --num_epochs 4 \
  --weight_decay 0.01 \
  --lr_scheduler linear \
  --grad_accum_steps 2 \
  --warmup_ratio 0.1 \
  --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --max_length 512
```

---

## **Troubleshooting**

- **Import Errors:**

  If you receive errors such as `ModuleNotFoundError`, ensure all required packages are installed.

  ```bash
  pip install <missing-package>
  ```

- **CUDA Errors:**

  If you're running on a GPU and encounter CUDA errors, check your GPU setup and PyTorch installation.

- **Dataset Loading Issues:**

  If the script fails to load the dataset, ensure that you have internet access and that the `datasets` library is installed properly.

- **Memory Errors:**

  Adjust `batch_size`, `max_length`, or use gradient accumulation (`--grad_accum_steps`) to manage memory usage.

---

## **Modifying the Script**

If you need to adjust the script further:

- **Change Default Hyperparameters:**

  In the `argparse` section, modify the `default` values.

- **Add New Arguments:**

  For example, to add an argument for the output directory:

  ```python
  parser.add_argument('--output_dir', type=str, default='./OUTPUT/', help='Output directory')
  ```

  Then, update `TrainingArguments`:

  ```python
  training_args = TrainingArguments(
      output_dir=args.output_dir,
      ...
  )
  ```

---
