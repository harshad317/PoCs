## **Step 1: Ensure You Have the Necessary Dependencies Installed**

Before running the script, make sure you have all the required Python packages installed. You can install them using `pip`:

```bash
pip install transformers datasets scikit-learn torch matplotlib seaborn argparse
```

If you're using a virtual environment (which is recommended), activate it before installing the packages.

**Note:** If you plan to use a GPU for training, ensure you have the appropriate version of PyTorch installed that supports CUDA. You can find the correct installation command for your setup at [PyTorch's official website](https://pytorch.org/get-started/locally/).

## **Step 2: Verify the Script Name and Location**

Ensure that your script is saved as `layerwise_FT.py` in your working directory.

## **Step 3: Open a Terminal or Command Prompt**

Navigate to the directory where your `layerwise_FT.py` script is located:

```bash
cd /path/to/your/script
```

## **Step 4: Running the Script with Default Parameters**

To run the script with the default settings, simply execute:

```bash
python layerwise_FT.py
```

This will start the training process using the default hyperparameters specified in the script.

## **Step 5: Customizing Hyperparameters via Command-Line Arguments**

If you want to adjust the hyperparameters or change the pre-trained model, you can pass arguments when running the script.

### **Available Arguments:**

- `--model_name`: Name or path of the pre-trained model (default: `'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'`)
- `--lr`: Learning rate (default: `2e-5`)
- `--batch_size`: Batch size (default: `16`)
- `--num_epochs`: Number of epochs (default: `3`)
- `--weight_decay`: Weight decay (default: `0.01`)
- `--lr_scheduler`: Learning rate scheduler type (default: `'linear'`)
- `--grad_accum_steps`: Gradient accumulation steps (default: `1`)
- `--warmup_ratio`: Warmup ratio (default: `0.1`)

### **Example Commands:**

**Using a Different Model:**

```bash
python layerwise_FT.py --model_name bert-base-uncased
```

**Adjusting Hyperparameters:**

```bash
python layerwise_FT.py --lr 3e-5 --batch_size 8 --num_epochs 5
```

**Combining Arguments:**

```bash
python layerwise_FT.py --model_name bert-base-uncased --lr 3e-5 --batch_size 8 --num_epochs 5 --weight_decay 0.1 --lr_scheduler cosine --grad_accum_steps 2 --warmup_ratio 0.2
```

### **Viewing Help:**

To see all available arguments and their descriptions, run:

```bash
python layerwise_FT.py --help
```

**Output:**

```
usage: layerwise_FT.py [-h] [--model_name MODEL_NAME] [--lr LR]
                       [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                       [--weight_decay WEIGHT_DECAY]
                       [--lr_scheduler LR_SCHEDULER]
                       [--grad_accum_steps GRAD_ACCUM_STEPS]
                       [--warmup_ratio WARMUP_RATIO]

Train a transformer model for sequence classification.

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Pre-trained model name or path (e.g.,
                        "bert-base-uncased")
  --lr LR               Learning rate (e.g., 2e-5)
  --batch_size BATCH_SIZE
                        Batch size (e.g., 16)
  --num_epochs NUM_EPOCHS
                        Number of epochs (e.g., 3)
  --weight_decay WEIGHT_DECAY
                        Weight decay (e.g., 0.01)
  --lr_scheduler LR_SCHEDULER
                        Learning rate scheduler (e.g., linear)
  --grad_accum_steps GRAD_ACCUM_STEPS
                        Gradient accumulation steps (e.g., 1)
  --warmup_ratio WARMUP_RATIO
                        Warmup ratio (e.g., 0.1)
```

## **Step 6: Monitoring the Training Process**

Once you run the script, it will start training the model. You should see progress logs in the terminal, similar to:

```
***** Running training *****
  Num examples = 1000
  Num Epochs = 3
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 1
  Total optimization steps = 188
...
```
