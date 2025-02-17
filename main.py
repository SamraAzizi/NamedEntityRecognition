import spacy
import random
from spacy.util import minibatch
from spacy.training.example import Example



train_data = [
    ("What is the price of 5 apples?", {"entities": [(21, 22, "QUANTITY"), (23, 29, "PRODUCT")]}),
    ("How much does 2 kg of rice cost?", {"entities": [(13, 14, "QUANTITY"), (19, 23, "PRODUCT")]}),
    ("Can I get the price for 3 oranges?", {"entities": [(25, 26, "QUANTITY"), (27, 34, "PRODUCT")]}),
    ("Tell me the cost of 1 liter of milk.", {"entities": [(20, 21, "QUANTITY"), (30, 34, "PRODUCT")]}),
    ("What is the rate of 500 grams of sugar?", {"entities": [(21, 30, "QUANTITY"), (34, 39, "PRODUCT")]}),
    ("I need the price for 12 eggs.", {"entities": [(22, 24, "QUANTITY"), (25, 29, "PRODUCT")]}),
    ("How much for 250 ml of oil?", {"entities": [(13, 19, "QUANTITY"), (23, 26, "PRODUCT")]}),
    ("Can you tell me the price of 6 mangoes?", {"entities": [(30, 31, "QUANTITY"), (32, 39, "PRODUCT")]}),
    ("Find the cost of 3 loaves of bread.", {"entities": [(17, 18, "QUANTITY"), (27, 32, "PRODUCT")]}),
    ("What is the price of 10 chocolates?", {"entities": [(21, 23, "QUANTITY"), (24, 34, "PRODUCT")]}),
    ("How much does 1 dozen bananas cost?", {"entities": [(13, 19, "QUANTITY"), (20, 27, "PRODUCT")]}),
    ("I want to buy 2 packets of biscuits.", {"entities": [(15, 16, "QUANTITY"), (27, 35, "PRODUCT")]}),
    ("What is the price for 500 ml of juice?", {"entities": [(23, 29, "QUANTITY"), (33, 38, "PRODUCT")]}),
    ("Tell me how much 4 water bottles cost?", {"entities": [(18, 19, "QUANTITY"), (20, 33, "PRODUCT")]}),
    ("Can I get the rate of 3 kilograms of flour?", {"entities": [(23, 34, "QUANTITY"), (38, 43, "PRODUCT")]}),
    ("What is the cost of 7 cans of soda?", {"entities": [(20, 21, "QUANTITY"), (30, 34, "PRODUCT")]}),
    ("How much do 2 packets of tea cost?", {"entities": [(13, 14, "QUANTITY"), (25, 28, "PRODUCT")]}),
    ("Find out the price of 4 pineapples.", {"entities": [(23, 24, "QUANTITY"), (25, 35, "PRODUCT")]}),
    ("I need the price for 1 kg of cheese.", {"entities": [(22, 25, "QUANTITY"), (29, 35, "PRODUCT")]}),
    ("What is the rate for 3 liters of yogurt?", {"entities": [(22, 30, "QUANTITY"), (34, 40, "PRODUCT")]}),
]


nlp = spacy.load('en_core_web_md')

if 'ner' not nlp.pipe_names:
    nlp.add_pipe('ner')
else:
    nlp.get_pipe('ner')


for _,annotations in train_data:
    for ent in annotations['entities']:
        if ent not in ner.labels:
            ner.add_label()