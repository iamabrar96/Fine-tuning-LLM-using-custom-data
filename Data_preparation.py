from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import json

# **********************************************************************************************************************
# ************************** step 1 creating a dataset ********************************************

dataset = load_dataset("deepset/germandpr", split="train")

def encode_texts(texts, tokenizer, model):
    """Encode a list of texts into vectors."""
    with torch.no_grad():
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        model_output = model(**encoded_input)
        # Use mean pooling for sentence-level embeddings
        embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()


positive_contexts = [i['text'] for i in dataset['positive_ctxs']]

all_contexts = list(np.array(positive_contexts[0:100]).flatten()) #+ list(np.array(hard_negative_contexts[0:100]).flatten().tolist())

positive_contexts = positive_contexts[0:100]

# Load a pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
easy_negative_contexts_1 = []

t = 0

for i in range(len(positive_contexts)):
    if t == 15:
        break
    positive_context_vector = encode_texts(positive_contexts[i], tokenizer, model)
    all_contexts_vectors = encode_texts(all_contexts, tokenizer, model)

    dimension = all_contexts_vectors.shape[1]  # Dimension of the vectors
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
    index.add(all_contexts_vectors)  # Add vectors to the index

    # Perform the search
    k = len(all_contexts_vectors)  # Number of neighbors to find to ensure we check all
    distances, indices = index.search(positive_context_vector, k)

    # distances and indices are sorted from smallest to largest by default,
    # so to find the most dissimilar (largest distances), we take the last 3 entries
    top_k_dissimilar_indices = indices[0][-3:]

    top_k_dissimilar_distances = distances[0][-3:]

    largest_dissimilar_index = np.argmax(top_k_dissimilar_distances)
    
    largest_dissimilar_context_index = top_k_dissimilar_indices[largest_dissimilar_index]

    easy_negative_contexts = [all_contexts[i] for i in top_k_dissimilar_indices]

    top_k_dissimilar_indexes_int = int(top_k_dissimilar_indices[0])

    top_k_dissimilar_indexes_int = int(largest_dissimilar_context_index)
    
    easy_negative_dictionary = dataset[top_k_dissimilar_indexes_int]['positive_ctxs']

    easy_negative_contexts_1.append(easy_negative_dictionary)

    t+=1

# Create a new list of dictionaries dynamically for the first three elements
new_dataset_list = []
for i in range(15):
    dataset_element = dataset[i].copy()  # Make a copy to avoid modifying the original dataset
    dataset_element["easy_negative_ctxt"] = easy_negative_contexts_1[i]
    new_dataset_list.append(dataset_element)

# Store the new_dataset_list in a JSON file
output_file_path = 'new_dataset_list_test_50.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_dataset_list, f, ensure_ascii=False, indent=4)

print(f"New dataset list saved to {output_file_path}")



# Load the transformed train dataset
with open('new_dataset_list.json', 'r') as f:
    transformed_dataset = json.load(f)

# Prepare the training data
training_data = []

for entry in transformed_dataset:
    question = entry['question']
    answers = entry['answers'][0]
    positive_context = entry['positive_ctxs']['text'][0]
    hard_negatives = entry['hard_negative_ctxs']['text']
    easy_negative = entry['easy_negative_ctxt']['text'][0]

    # Create prompt for positive context
    training_data.append({
        "prompt": f"Question: {question}\nContext: {positive_context}\nAnswer:",
        "completion": f" {answers}"
    })

    # Create prompts for hard negative contexts
    for hard_negative in hard_negatives:
        training_data.append({
            "prompt": f"Question: {question}\nContext: {hard_negative}\nAnswer:",
            "completion": " No answer."
        })

    # Create prompt for easy negative context
    training_data.append({
        "prompt": f"Question: {question}\nContext: {easy_negative}\nAnswer:",
        "completion": " No answer."
    })

# Save the formatted data
with open('formatted_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=4)

print(f"Formatted training data saved to formatted_train_data.json")


