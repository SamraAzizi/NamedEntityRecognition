import spacy


text = [
    'John loves to go to shopping!',
    'Jasra is palying a game',
    'musa is not preparing for exam',
    'ahmed is the guy behind the car'
    'elon musk is my favorite person',
    'the athlete was about to win the game'
]
nlp = spacy.load('en_core_web_md')