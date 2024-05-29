import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Reformatting the original dataset into gpt-3.5 prompt template

def process_dataset(dataset_name, split, num_rows, json_filename, jsonl_filename):
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    # Extract the first num_rows rows
    first_rows = [dataset[i] for i in range(num_rows)]

    # Save the extracted rows to a JSON file
    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(first_rows, json_file, ensure_ascii=False, indent=4)

    # Load the formatted data from the JSON file
    with open(json_filename, 'r') as f:
        formatted_data = json.load(f)

    # Prepare the data in JSONL format
    with open(jsonl_filename, 'w', encoding='utf-8') as outfile:
        for entry in formatted_data:
            json.dump(entry, outfile)
            outfile.write('\n')

# Process training data
process_dataset("deepset/germandpr", "train", 50, "original_train_data_50.json", "original_train_data_50.jsonl")

# Process test data
process_dataset("deepset/germandpr", "test", 50, "original_test_data_50.json", "original_test_data_50.jsonl")


#  Reformatting the transformed data into gpt-3.5 prompt template 

def prepare_data(input_file, output_file):
    with open(input_file, 'r') as f:
        formatted_data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in formatted_data:
            json.dump(entry, outfile)
            outfile.write('\n')

# Prepare training data
prepare_data('formatted_train_data.json', 'train_data.jsonl')

# Prepare testing data
prepare_data('formatted_test_data.json', 'test_data.jsonl')

