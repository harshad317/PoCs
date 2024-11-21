## Introduction

`moa_final.py` is a Python script designed to extract patient information, drugs, and adverse reactions from medical report texts and classify these reports as either Individual Case Safety Reports (ICSR) or DISCARD. The script leverages the Mistral 7B model via the Ollama API for natural language understanding tasks, including entity extraction and classification.

## Requirements

- **Python**: Version 3.7 or higher
- **Python Packages**:
  - `pandas`
  - `datasets` (HuggingFace)
  - `openai`
  - `pydantic`
  - `tqdm`
- **Ollama**: Installed and running with the Mistral 7B model
- **Internet Connection**: For downloading the dataset and models

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/harshad317/PoCs/tree/main
cd ./MOA
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install pandas datasets openai pydantic tqdm
```

### 4. Install and Configure Ollama

#### Install Ollama

Download and install Ollama from the [official website](https://ollama.ai/). Follow the platform-specific installation instructions.

#### Download the Mistral 7B Model

```bash
ollama pull mistral:7b-instruct-v0.3-q8_0
```

#### Start the Ollama Server

```bash
ollama serve
```

By default, the Ollama server runs on `http://localhost:11434/v1`.

### 5. Configure the Script

Open `moa_final.py` in a text editor and ensure the `OpenAI` client is configured correctly:

```python
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

- **base_url**: The URL where the Ollama server is running.
- **api_key**: Set to `"ollama"` (required by the `openai` library but not used by Ollama).

## Usage

Run the script:

```bash
python moa_final.py
```

The script will:

1. Load the "DKTech/ICSR-data" dataset.
2. Process each report in the test set:
   - Extract entities (patient details, drug, adverse reaction) using the Mistral model.
   - Classify the report based on the extracted entities.
   - Classify the report based on the full text.
   - Make a final prediction based on both classifications.
3. Save the results to `mistral.csv`.

## Dataset

The script uses the "[DKTech/ICSR-data](https://huggingface.co/datasets/DKTech/ICSR-data)" dataset from HuggingFace, which contains medical reports labeled as 'ICSR' (Individual Case Safety Report) or 'DISCARD'.

## Detailed Workflow

1. **Data Loading and Preparation**

   - Loads the dataset using the HuggingFace `datasets` library.
   - Maps and relabels the original labels (`0` and `1`) to `'DISCARD'` and `'ICSR'` for clarity.

2. **Entity Extraction**

   - For each report, sends a prompt to the Mistral model to extract entities:
     - **System Prompt**:  
       ```
       Extract patient, drug, and adverse reaction information from the following text.
       ```
     - **User Input**:  
       ```
       Text: [Report Text]
       ```
   - Expects the model to return a JSON-formatted string containing the extracted entities:
     ```json
     {
       "patient": "...",
       "drug": "...",
       "adverse_reaction": "..."
     }
     ```

3. **Classification Based on Entities**

   - Classifies the report based on the extracted entities:
     - **System Prompt**:  
       ```
       Classify the report as ICSR or DISCARD based on the extracted entities.
       ```
     - **User Input**:  
       ```
       Entities: [Extracted Entities in JSON]
       ```

4. **Classification Based on Full Text**

   - Classifies the report using the full text:
     - **System Prompt**:  
       ```
       Classify the given text as ICSR or DISCARD.
       ```
     - **User Input**:  
       ```
       Text: [Report Text]
       ```

5. **Final Prediction**

   - Makes a final classification based on the two model predictions:
     - **System Prompt**:  
       ```
       Make a final classification based on two model predictions.
       ```
     - **User Input**:  
       ```
       Model 1: [Entity-Based Classification], Model 2: [Text-Based Classification]
       ```

6. **Result Logging**

   - Logs and saves the following to `mistral.csv`:
     - Original label
     - Extracted entities
     - Entity-based classification
     - Abstract-based classification
     - Final prediction

## Notes and Tips

- **Error Handling**: The script includes robust error handling for API calls. If an API call fails after the maximum number of retries, it logs an error and proceeds to the next report.

- **Model Output Format**: Ensure the Mistral model outputs responses in the expected format, especially for JSON outputs during entity extraction.

- **Server Availability**: Confirm the Ollama server is running and accessible before executing the script.

- **Performance**: Processing the entire test set may take time depending on your hardware. For quicker testing, consider processing a subset of the data.

## Troubleshooting

- **Connection Errors**: If you encounter connection issues, verify:

  - The Ollama server is running (`ollama serve`).
  - The `base_url` in `moa_final.py` points to the correct address.
  - There are no firewall or network issues blocking the connection.

- **API Key Errors**: If you receive API key-related errors:

  - Ensure `api_key` is set to `"ollama"` in `moa_final.py`.
  - The `openai` library requires an API key parameter, but Ollama does not enforce API key validation.

- **Incorrect Model Responses**:

  - If the model doesn't return responses in the expected format, you may need to adjust the prompts or fine-tune the model.
  - Verify that the prompts are correctly formatted and that the model understands the instructions.

- **Module Import Errors**:

  - Ensure all required Python packages are installed in your environment.
  - Activate the virtual environment if you're using one.

## License

[Specify your project's license here, e.g., MIT License]

## Contact

For questions or feedback, please contact:

- **Name**: Harshad Patil
- **Email**: hhpatil001@gmail.com

---
