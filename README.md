# Custom NER Model with SpaCy

This repository contains a Python script that demonstrates how to train a custom Named Entity Recognition (NER) model using the SpaCy library. The model is designed to recognize quantities and products from user queries related to pricing.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training Data](#training-data)
- [Model Training](#model-training)
- [Testing the Model](#testing-the-model)


## Requirements

- Python 3.6 or higher
- SpaCy 3.x

You can install SpaCy using pip:

```bash
pip install spacy
```
# Installation
Clone this trepository to your local machine:

```bash
git clone https://github.com/SamraAzizi/custom-ner-model.git
cd custom-ner-model
```

# Usage

1. Run the Scripts: Execute the script to train  the NER model and test it with sample queries.

```bash
python train_ner.py
```

2. View the output: The script will print the entities recogniezed in the rest quieres.

# Training Data
The training data consists of a list of tuples, where each tuple contains a text string and a dictionary of annotations. The annotations specify the entities to be recognized, including their start and end character indices and their labels (e.g., QUANTITY, PRODUCT).

Example training data:
```bash
train_data = [
    ("What is the price of 5 apples?", {"entities": [(21, 22, "QUANTITY"), (23, 29, "PRODUCT")]}),
    ...
]

```

# Model Training
The model is trained using the following steps:

1. Load a blank SpaCy model.
2. Add a NER pipeline if it is not already present.
3. Add entity labels from the training data.
4. Disable other pipeline components to focus on training the NER model.
5. Shuffle the training data and train the model for a specified number of epochs (30 in this case).
6. Save the trained model to disk.

# Tesing the Model 
After training, the model is loaded from disk and tested with a set of predefined queries. The recognized entities are printed to the console.

Example test queries:

```bash
test_texts = [
    "How much for 3 oranges?",
    "I want 15 eggs for the conference.",
    "Can you give me the price for 6 bread?"
]

```