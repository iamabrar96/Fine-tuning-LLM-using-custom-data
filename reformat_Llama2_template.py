import json

# Load the modified train dataset
with open('new_dataset_list.json', 'r') as file:
    data = json.load(file)

# Function to format each data entry for Llama2 prompt template
def format_for_llama2(entry):
    question = entry["question"]
    answer = entry["answers"][0]  # Assuming a single answer

    # Extracting positive context
    positive_title = entry["positive_ctxs"]["title"][0] if entry["positive_ctxs"]["title"] else ""
    positive_text = entry["positive_ctxs"]["text"][0] if entry["positive_ctxs"]["text"] else ""

    # Extracting hard negative contexts
    hard_negative_titles = entry["hard_negative_ctxs"]["title"]
    hard_negative_texts = entry["hard_negative_ctxs"]["text"]

    hard_negative_contexts = ""
    for i in range(len(hard_negative_titles)):
        hard_negative_title = hard_negative_titles[i] if i < len(hard_negative_titles) else ""
        hard_negative_text = hard_negative_texts[i] if i < len(hard_negative_texts) else ""
        hard_negative_contexts += f"\nTitle: {hard_negative_title}\nText: {hard_negative_text}"

    # Extracting easy negative context
    easy_negative_title = entry["easy_negative_ctxt"]["title"][0] if entry["easy_negative_ctxt"]["title"] else ""
    easy_negative_text = entry["easy_negative_ctxt"]["text"][0] if entry["easy_negative_ctxt"]["text"] else ""

    # Constructing the prompt
    prompt = (
        f"<s> [INST] <<SYS>>\nSystem prompt\n"
        f"Positive Contexts:\nTitle: {positive_title}\nText: {positive_text}\n\n"
        f"Hard Negative Contexts:{hard_negative_contexts}\n\n"
        f"Easy Negative Contexts:\nTitle: {easy_negative_title}\nText: {easy_negative_text}\n"
        f"<</SYS>>\n\nUser prompt: {question} [/INST] Model answer: {answer} </s>"
    )

    return {"prompt": prompt}

# Format all entries
formatted_data = [format_for_llama2(entry) for entry in data]

# Output to JSON
with open('original_formatted_test_data_llama2.json', 'w') as f:
    json.dump(formatted_data, f, indent=4)



# Load the modified test dataset
with open('new_dataset_list_test.json', 'r') as file:
    data = json.load(file)

# Function to format each data entry for Llama2 prompt template
def format_for_llama2(entry):
    question = entry["question"]
    answer = entry["answers"][0]  # Assuming a single answer

    # Extracting positive context
    positive_title = entry["positive_ctxs"]["title"][0] if entry["positive_ctxs"]["title"] else ""
    positive_text = entry["positive_ctxs"]["text"][0] if entry["positive_ctxs"]["text"] else ""

    # Extracting hard negative contexts
    hard_negative_titles = entry["hard_negative_ctxs"]["title"]
    hard_negative_texts = entry["hard_negative_ctxs"]["text"]

    hard_negative_contexts = ""
    for i in range(len(hard_negative_titles)):
        hard_negative_title = hard_negative_titles[i] if i < len(hard_negative_titles) else ""
        hard_negative_text = hard_negative_texts[i] if i < len(hard_negative_texts) else ""
        hard_negative_contexts += f"\nTitle: {hard_negative_title}\nText: {hard_negative_text}"

    # Extracting easy negative context
    easy_negative_title = entry["easy_negative_ctxt"]["title"][0] if entry["easy_negative_ctxt"]["title"] else ""
    easy_negative_text = entry["easy_negative_ctxt"]["text"][0] if entry["easy_negative_ctxt"]["text"] else ""

    # Constructing the prompt
    prompt = (
        f"<s> [INST] <<SYS>>\nSystem prompt\n"
        f"Positive Contexts:\nTitle: {positive_title}\nText: {positive_text}\n\n"
        f"Hard Negative Contexts:{hard_negative_contexts}\n\n"
        f"Easy Negative Contexts:\nTitle: {easy_negative_title}\nText: {easy_negative_text}\n"
        f"<</SYS>>\n\nUser prompt: {question} [/INST] Model answer: {answer} </s>"
    )

    return {"prompt": prompt}

# Format all entries
formatted_data = [format_for_llama2(entry) for entry in data]

# Output to JSON
with open('original_formatted_test_data_llama2.json', 'w') as f:
    json.dump(formatted_data, f, indent=4)
