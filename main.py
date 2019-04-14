import argparse
import string
import nltk
from nltk import tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from textblob import TextBlob
from polyglot.text import Text, Word
import json

parser = argparse.ArgumentParser()
parser.add_argument('--files', type=str, nargs='+')
parser.add_argument('--period', type=int)

args = parser.parse_args()
files = args.files
period = args.period


def clean_text(data):
    data = data.translate(
        str.maketrans('', '', string.digits))
    data = data.translate(
        str.maketrans('\n', ' '))
    data = data.lower()

    return data


def text_to_sentences(data):
    sentences = tokenize.sent_tokenize(data)

    for i in range(len(sentences)):
        sentences[i] = sentences[i].translate(
            str.maketrans('', '', string.punctuation))
        sentences[i] = sentences[i].split()

    return sentences


def average_words_per_sentence(sentences):
    sentence_count = len(sentences)

    words = 0
    for sentence in sentences:
        words += len(sentence)

    return words / sentence_count


def average_word_length(sentences):
    words = 0
    for sentence in sentences:
        words += len(sentence)

    word_lengths = 0
    for sentence in sentences:
        for word in sentence:
            word_lengths += len(word)

    return word_lengths / words


def pos_distribution(data):
    pos = {}

    for tag in ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']:
        pos[tag] = 0

    blob = TextBlob(data)
    pos_tags = blob.tags
    total_tags = len(pos_tags)

    pos_tags = [e[1] for e in pos_tags]
    for tag in pos_tags:
        pos[tag] += 1

    # pos = sorted(pos.items(), key=lambda x: -x[1])
    for tag in pos.keys():
        pos[tag] /= total_tags

    return pos


def keywords(data):
    tfidf = TfidfVectorizer()
    tfs = tfidf.fit_transform(data.split())
    labels = tfidf.get_feature_names()
    tfs = tfs.toarray()
    scores = np.average(tfs, axis=0)

    keywords = {}

    for i in range(len(labels)):
        keywords[labels[i]] = scores[i]

    keywords = sorted(keywords.items(), key=lambda x: x[1])
    keywords = [e[0] for e in keywords][:10]

    return keywords


def char_frequency(data):
    data = data.translate(
        str.maketrans('', '', string.punctuation))

    total_letters = 0
    dict = {}

    for n in string.ascii_letters.lower():
        dict[n] = 0

    for n in data:
        keys = dict.keys()
        if n in string.ascii_letters:
            total_letters += 1
            if n in keys:
                dict[n] += 1
            else:
                dict[n] = 1

    # dict = sorted(dict.items(), key=lambda x: -x[1])
    for n in string.ascii_letters[:26]:
        dict[n] /= total_letters

    return dict


def morphemes(data):
    data = data.translate(
        str.maketrans('', '', string.punctuation))

    data = Text(data)
    data.language = 'en'
    morphemes = list(data.morphemes)
    morphemes = {i: morphemes.count(i) for i in set(morphemes)}
    morphemes = sorted(morphemes.items(), key=lambda x: -x[1])[:30]

    return morphemes


def sentiment(data):
    data = TextBlob(data)

    return data.sentiment.polarity


def vocab_score(data):
    data = data.translate(
        str.maketrans('', '', string.punctuation))

    words = data.split()
    total_count = len(words)
    unique_count = len(set(words))

    score = np.log(unique_count)/np.log(total_count)
    return score


for filename in files:
    print('(*)', filename)

    file = open(filename)
    data = file.read()

    data = clean_text(data)

    pos_distribution = pos_distribution(data)

    keywords = keywords(data)

    char_frequency = char_frequency(data)

    sentiment = sentiment(data)

    # print('[*] Morphemes:',
    #      morphemes(data))

    vocab_score = vocab_score(data)

    sentences = text_to_sentences(data)

    average_words_per_sentence = average_words_per_sentence(sentences)

    average_word_length = average_word_length(sentences)

    result = {
        'name': filename[:-4],
        'period': period,
        'pos': pos_distribution,
        'keywords': keywords,
        'chars': char_frequency,
        'sentiment': sentiment,
        'vocab': vocab_score,
        'wps': average_words_per_sentence,
        'wl': average_word_length
    }

    with open(filename[:-4] + '.json', 'w+') as fp:
        json.dump(result, fp)
