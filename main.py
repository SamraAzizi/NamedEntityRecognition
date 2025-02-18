import spacy
import random
from spacy.util import minibatch
from spacy.training.example import Example

# Training Data
train_data = [
    ("What is the price of 5 apples?", {"entities": [(21, 22, "QUANTITY"), (23, 29, "PRODUCT")]}),
    ("How much does 2 kg of rice cost?", {"entities": [(13, 17, "QUANTITY"), (21, 25, "PRODUCT")]}),
    ("Can I get the price for 3 oranges?", {"entities": [(25, 26, "QUANTITY"), (27, 34, "PRODUCT")]}),
    ("Tell me the cost of 1 liter of milk.", {"entities": [(20, 27, "QUANTITY"), (31, 35, "PRODUCT")]}),
    ("What is the rate of 500 grams of sugar?", {"entities": [(21, 30, "QUANTITY"), (34, 39, "PRODUCT")]}),
    ("I need the price for 12 eggs.", {"entities": [(22, 24, "QUANTITY"), (25, 29, "PRODUCT")]}),
    ("How much for 250 ml of oil?", {"entities": [(13, 19, "QUANTITY"), (23, 26, "PRODUCT")]}),
    ("Can you tell me the price of 6 mangoes?", {"entities": [(30, 31, "QUANTITY"), (32, 39, "PRODUCT")]}),
    ("Find the cost of 3 loaves of bread.", {"entities": [(17, 18, "QUANTITY"), (27, 32, "PRODUCT")]}),
]

# Load Spacy model
nlp = spacy.blank('en')  # Use blank model to train from scratch

# Add NER pipeline if not already present
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe('ner')

# Add labels from training data
for _, annotations in train_data:
    for ent in annotations['entities']:
        ner.add_label(ent[2])

# Disable other pipes for training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    nlp.initialize()  # Initialize the training

    epochs = 30  # Reduce epochs for efficiency
    for epoch in range(epochs):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=2)

        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)

            nlp.update(examples, drop=0.3, losses=losses)

        print(f'Epoch {epoch+1}, Loss: {losses}')

# Save the trained model
nlp.to_disk('custom_ner_model')

# Load the trained model
trained_nlp = spacy.load('custom_ner_model')

# Test the trained model
test_texts = [
    "How much for 3 oranges?",
    "I want 15 chairs for the conference.",
    "Can you give me the price for 6 desks?"
]

for text in test_texts:
    doc = trained_nlp(text)
    print(f'Text: {text}')
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
    print()
