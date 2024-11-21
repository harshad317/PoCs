import argparse
import pandas as pd
from typing import Optional
from pydantic import BaseModel
from typing import List
from langchain_community.llms import Ollama
from langchain_core.pydantic_v1 import Field
import time
from datasets import load_dataset

def relabel_and_format(example):
    if example['label'] == 0:
        example['label'] = 'DISCARD'
    elif example['label'] == 1:
        example['label'] = 'ICSR'
    return example

def main(args):
    df = load_dataset("DKTech/ICSR-data")
    df = df.map(relabel_and_format)

    test = df['test'].to_pandas()
    if args.limit:
        test = test[:args.limit]

    system_msg = """You are a Pharmacovigilance Professional. Your task is to classify the given text as ICSR (Individual Case Safety Report) or not. MAKE SURE YOU DO NOT MAKE ANY ASSUMPTIONS AND YOU WILL AWARDED WITH $1000 FOR IT.

    To determine whether an abstract qualifies as an ICSR, it should include several key elements:

    1. Identifiable Patient(Humans only): The report must contain enough information of a patient to identify the patient who experienced the adverse event. This does not necessarily mean personal identifiers but could include age, gender, initials, etc., ensuring that it's clear there was a specific individual affected.

    2. **Suspected Product(s)**: The report should clearly specify which drug(s), biologic(s), vaccine(s), or medical device(s) are suspected to have caused or contributed to the adverse event.

    3. **Adverse Event/Reaction Description**: A detailed description of what happened to the patient after using the suspected product is essential. This includes symptoms, diagnosis (if available), severity of reactions/events, outcomes (recovery status/death/disability/etc.), and any relevant laboratory data or test results.

    To classify an article as an ICSR then make sure that all the criterias are met, otherwise it is DISCARD.

    Final classification should be ICSR when all criterias are met and DISCARD if ANY ONE of the criterias are are met.

    ONE WORD ANSWER IS EXPECTED. ICSR or DISCARD and No reasoning needed . You will rewarded with $1000 for following the steps one by one.

    Think step by step"""

    llm = Ollama(model=args.model, temperature=0, top_p=15, top_k=0.15, num_thread=args.cpu, system=system_msg)

    results = []
    start_index = 0
    if args.resume:
        try:
            results_df = pd.read_csv(args.output_file)
            start_index = len(results_df)
            results = results_df.to_dict('records')
        except FileNotFoundError:
            pass

    start_time = time.time()
    total_time = 0.0
    for i, example in enumerate(test.iterrows(), start=start_index):
        idx, row = example
        print("="*15, f"{idx+1}", "="*15)
        output = llm(row['text'])
        print(f"Classification: {output}")
        print("\n")
        results.append({'text': row['text'], 'label': output.split()[0]})

        if (i + 1) % 100 == 0:
            results_df = pd.DataFrame(results)
            results_df.to_csv(args.output_file, index=False)
            end_time = time.time()
            print("\n")
            print(f"Time after {i} rows: {(end_time - start_time) / 3600:.2f} hours")
            total_time += (end_time - start_time) / 3600
            hours = int(total_time)
            minutes = int((total_time - hours) * 60)
            print(f"Total time taken: {hours} hours {minutes} minutes")
            print("\n")
            start_time = time.time()

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify ICSR data')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of examples to process')
    parser.add_argument('--model', type=str, default='mistral:v0.3', help='Specify the model to use')
    parser.add_argument('--cpu', type= int, default=6, help= 'No of CPUs to use')
    parser.add_argument('--output_file', type=str, default='results.csv', help='Specify the output file name')
    parser.add_argument('--resume', action='store_true', help='Resume from the last processed index')
    args = parser.parse_args()
    main(args)
