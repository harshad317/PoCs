import argparse
import ollama
import concurrent.futures
from datasets import load_dataset
import pandas as pd
import time

def relabel_and_format(example):
    if example['label'] == 0:
        example['label'] = 'DISCARD'
    elif example['label'] == 1:
        example['label'] = 'ICSR'
    return example

def process_prompt(text, model):
    result = ollama.generate(model=model, prompt=text)
    response = result.get('response', '')
    if 'ICSR' in response:
        return {'response': 'ICSR'}
    else:
        return {'response': 'DISCARD'}

def main(args):
    df = load_dataset("DKTech/ICSR-data")
    df = df.map(relabel_and_format)
    test = df['test'].to_pandas()
    total_samples = len(test)
    test = test[:args.num_samples]
    
    print(f"Processing {len(test)} articles...")

    prompt = """You are a Pharmacovigilance Professional. Your task is to classify the given text as ICSR (Individual Case Safety Report) or not. MAKE SURE YOU DO NOT MAKE ANY ASSUMPTIONS AND YOU WILL AWARDED WITH $1000 FOR IT.

    To determine whether an abstract qualifies as an ICSR, it should include several key elements:

    1. Identifiable Patient(Humans only): The report must contain enough information of a patient to identify the patient who experienced the adverse event. This does not necessarily mean personal identifiers but could include age, gender, initials, etc., ensuring that it's clear there was a specific individual affected.

    2. Suspected Product(s): The report should clearly specify which drug(s), biologic(s), vaccine(s), or medical device(s) are suspected to have caused or contributed to the adverse event.

    3. Adverse Event/Reaction Description: A detailed description of what happened to the patient after using the suspected product is essential. This includes symptoms, diagnosis (if available), severity of reactions/events, outcomes (recovery status/death/disability/etc.), and any relevant laboratory data or test results.

    To classify an article as an ICSR then make sure that all the criterias are met, otherwise it is DISCARD.

    Final classification should be ICSR when all criterias are met and DISCARD if ANY ONE of the criterias are are met.

    ONE WORD ANSWER IS EXPECTED. ICSR or DISCARD and No reasoning needed . You will rewarded with $1000 for following the steps one by one.

    Think step by step"""

    start_time = time.time()
    articles_processed = 0
    icsr_count = 0
    discard_count = 0

    batch_size = 100
    for i in range(0, len(test), batch_size):
        batch = test.iloc[i:i+batch_size].copy()
        batch['text'] = prompt + " " + batch["text"]
        #batch['prediction'] = 'DISCARD'

        batch_texts = batch['text'].tolist()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(lambda text: process_prompt(text, args.model), text) for text in batch_texts]
            batch_results = [future.result() for future in futures]
            
        batch['prediction'] = [result['response'] for result in batch_results]
        
        # Count ICSR and DISCARD
        batch_icsr_count = (batch['prediction'] == 'ICSR').sum()
        batch_discard_count = (batch['prediction'] == 'DISCARD').sum()
        
        icsr_count += batch_icsr_count
        discard_count += batch_discard_count
        articles_processed += len(batch)

        # Write batch results to CSV
        batch[['text', 'label', 'prediction']].to_csv(args.output_file, index=False, mode='a', header=(i==0))

        print(f"{articles_processed} articles are done in {(time.time() - start_time) / 60:.2f} mins")
        print(f"Number of articles classified as ICSR: {icsr_count}")
        print(f"Number of articles classified as DISCARD: {discard_count}")
        print("="*50)
        print("\n")

        # Clear the batch from memory
        del batch
        del batch_results

    end_time = time.time()
    total_time = end_time - start_time
    print("="*50)
    print("\n")
    print(f"{articles_processed} articles finished in {total_time / 60:.2f} mins")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify text as ICSR or DISCARD")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model to use")
    parser.add_argument("--output_file", type=str, default="predictions.csv", help="Output file for predictions")
    args = parser.parse_args()
    main(args)
