# Fine-Tuning Language Models Using Custom Data

## Overview

This project involves fine-tuning language models using an improved version of the "deepset/germandpr" dataset. The goal is to enhance the model's ability to distinguish between relevant and irrelevant contexts by adding easy negative examples to the training data.

## Original Dataset

The original dataset, "deepset/germandpr", contains:
- **9275 training examples**
- **1025 testing examples**

Each example is a question/answer pair, consisting of:
- **One question**
- **One answer**
- **One positive context**
- **Three negative contexts**

You can find the original dataset [here](https://huggingface.co/datasets/deepset/germandpr).

## Modifications

### Adding Easy Negative Examples

To improve the dataset, an easy negative example was added to each row. The purpose is to train the model to better distinguish between relevant and irrelevant contexts by exposing it to plausible but incorrect information.

#### Method

We used the L2 distance metric from Faiss to find the most dissimilar index (vector) relative to the positive context in each row. This dissimilar index was chosen as the easy negative example.

## Reformatting for Model Fine-Tuning

The modified dataset was reformatted into the respective prompt templates for the Llama2 and GPT-3.5 models, preparing it for the fine-tuning process.

### Fine-Tuning Techniques

#### QLoRA for Llama2

For fine-tuning the Llama2 model, the Quantized Low Rank Adaptation (QLoRA) technique was used. This technique reduces the model's parameters, making it feasible to train on platforms with resource constraints, such as Google Colab.

## Summary

- **Original Dataset**: [deepset/germandpr](https://huggingface.co/datasets/deepset/germandpr)
- **Improvements**: Added easy negative examples using L2 distance for better context distinction.
- **Model Reformatting**: Adapted the dataset for Llama2 and GPT-3.5.
- **Fine-Tuning**: Employed QLoRA for Llama2 to manage resource constraints during training.

This project aims to enhance the performance of language models by providing them with a more challenging training set, improving their ability to discern relevant information from irrelevant contexts.
