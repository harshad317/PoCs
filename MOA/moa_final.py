import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union
import time
from tqdm import tqdm
import csv
import json

class ICSR(BaseModel):
    patient: Optional[Union[str, List[str], dict]] = Field(None)
    drug: Optional[Union[str, List[str], dict]] = Field(None)
    adverse_reaction: Optional[Union[str, List[str], dict]] = Field(None)

class Classify(BaseModel):
    classification: Literal['ICSR', 'DISCARD'] = Field(None)

def relabel_and_format(example):
    if example['label'] == 0:
        example['label'] = 'DISCARD'
    elif example['label'] == 1:
        example['label'] = 'ICSR'
    return example

df = load_dataset("DKTech/ICSR-data")
df = df.map(relabel_and_format)

client = OpenAI(base_url="https://11434-cs-776365198600-default.cs-asia-southeast1-yelo.cloudshell.dev/v1", api_key="ollama")

def safe_api_call(messages, model="mistral:7b-instruct-v0.3-q8_0", max_retries=3):
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {str(e)}. Retrying...")
            time.sleep(1)
    return None

results = []

for index, example in tqdm(df['test'].to_pandas().iterrows()):
    print("="*10, index + 1, "="*10)
    try:
        # Extract entities
        entities_response = safe_api_call([
            {"role": "system", "content": "Extract patient, drug, and adverse reaction information from the following text."},
            {"role": "user", "content": f"Text: {example['text']}"}
        ])
        
        if entities_response:
            entities = json.loads(entities_response)
        else:
            raise Exception("Failed to extract entities")

        # Classify based on entities
        entity_classification = safe_api_call([
            {"role": "system", "content": "Classify the report as ICSR or DISCARD based on the extracted entities."},
            {"role": "user", "content": f"Entities: {json.dumps(entities)}"}
        ])

        # Classify based on abstract
        abstract_classification = safe_api_call([
            {"role": "system", "content": "Classify the given text as ICSR or DISCARD."},
            {"role": "user", "content": f"Text: {example['text']}"}
        ])

        # Final prediction
        final_prediction = safe_api_call([
            {"role": "system", "content": "Make a final classification based on two model predictions."},
            {"role": "user", "content": f"Model 1: {entity_classification}, Model 2: {abstract_classification}"}
        ])

        results.append({
            'original_label': example['label'],
            'patient_details': str(entities.get('patient', '')),
            'drug': str(entities.get('drug', '')),
            'adverse_reaction': str(entities.get('adverse_reaction', '')),
            'entity_based_classification': entity_classification,
            'abstract_based_classification': abstract_classification,
            'final_prediction': final_prediction
        })

        print("Classification based on the Entities: ", entity_classification)
        print("Classification based on the abstract: ", abstract_classification)
        print("Final prediction: ", final_prediction)

    except Exception as e:
        print(f"Error processing example {index + 1}: {str(e)}")
        results.append({
            'original_label': example['label'],
            'patient_details': 'Error',
            'drug': 'Error',
            'adverse_reaction': 'Error',
            'entity_based_classification': 'Error',
            'abstract_based_classification': 'Error',
            'final_prediction': 'Error'
        })

with open('mistral.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['original_label', 'patient_details', 'drug', 'adverse_reaction', 'entity_based_classification', 'abstract_based_classification', 'final_prediction']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("Results saved to mistral.csv")
