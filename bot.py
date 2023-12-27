import re
from os import path
from collections import Counter
from warnings import warn
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request

# Spell Checker
class SpellChecker:
    def __init__(self, local_path, language='en'):
        try:
            self.WORDS = Counter(self.words(open(path.join(
                local_path, language, "words.txt"), encoding='utf-8').read()))
        except FileNotFoundError:
            warn("words.txt for language `{}` not found in `{}`".format(language, local_path),
                 ResourceWarning)
            self.WORDS = Counter()
        self.total_word_count = sum(self.WORDS.values())
        if self.total_word_count == 0:
            self.total_word_count = 1

    @staticmethod
    def words(text):
        return re.findall(r'\w+', text.lower())

    def correction(self, text, min_word_length=4):
        return " ".join(i if len(i) < min_word_length or self.WORDS[i]
                        else max(self.candidates(i), key=self.probability)
                        for i in text.split())

    def probability(self, word):
        return self.WORDS[word] / self.total_word_count

    def candidates(self, word):
        return (self.known([word]) or self.known(self.edits1(word)) or
                self.known(self.edits2(word)) or [word])

    def known(self, words):
        return {w for w in words if w in self.WORDS}

    @staticmethod
    def edits1(word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

# Load the spell checker
spell_checker = SpellChecker(local_path='path/to/your/spellchecker/files', language='en')

# Flask App
app = Flask(__name__)

# Load the chatbot model and data
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
tasks = json.loads(open('tasks.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.form['user_message']

    # Spell check the user's message
    corrected_msg = spell_checker.correction(user_message)

    # If the corrected message is different from the original, use the corrected message for prediction
    if corrected_msg != user_message:
        user_message = corrected_msg

    # Predict the class based on the corrected message
    ints = predict_class(user_message, model)

    # Get the response from the chatbot
    bot_response = get_response(ints, tasks)
    return bot_response

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = [spell_checker.correction(word) for word in clean_up_sentence(sentence)]
    
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(ints, tasks_json):
    tag = ints[0]['intent']
    list_of_tasks = tasks_json['tasks']
    for i in list_of_tasks:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

if __name__ == '__main__':
    app.run(debug=True)
