import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Doc
from spacy.matcher import Matcher
from spacy.util import filter_spans
import numpy as np
import pandas as pd
import json
import operator
from string import punctuation

stopwords=list(STOP_WORDS)
punctuation=punctuation + '\n'

nlp = spacy.load('en_core_web_lg')

strings = []
vectors = []
for key, vector in nlp.vocab.vectors.items():
    try:
        strings.append(nlp.vocab.strings[key])
        vectors.append(vector)
    except:
        pass
vectors = np.vstack(vectors)

def related_adjectives(noun, noun_chunks):
    patterns=[
    [{'POS': 'ADJ', 'OP':'+'}, {'LOWER': noun.text, 'POS': noun.pos_}]
    ]
    matcher = Matcher(nlp.vocab)
    matcher.add("related-adjectives", patterns)
    adjectives = []
    for chunk in noun_chunks:
        matches = matcher(chunk)
        spans = []
        for match_id, start, end in matches:
            span = chunk[start:end]
            spans.append(span)
        matches = filter_spans(spans)
        for match in matches:
            idx = next(idx for idx,token in enumerate(match) if token.text==noun.text)
            if match[:idx].text not in adjectives:
                adjectives.append(match[:idx].text)
    return adjectives

def most_similar(word, pos, count = 200):

    main = word.vector

    diff = vectors - main
    diff = diff**2
    diff = np.sqrt(diff.sum(axis = 1), dtype = np.float64)

    df = pd.DataFrame(strings, columns = ['keyword'])
    df['diff'] = diff
    df = df.sort_values('diff', ascending = True).head(count)
    df['keyword'] = df['keyword'].str.lower()
    df = df.drop_duplicates(subset = 'keyword', keep = 'first')
    similar_list = []
    for keyword in df['keyword'].tolist():
        token = nlp(keyword)[0]
        if token.lemma_ not in similar_list and token.lemma_ not in stopwords and token.pos_==pos and token.lemma_ not in word.text.lower():
            similar_list.append(token.lemma_)

    return similar_list

def get_definition(word):
    data = json.load(open("dict.json"))
    if word.text.lower() in data:
        meaning = data[word.text.lower()]
    elif word.lemma_ in data:
        meaning = data[word.lemma_]
    else:
        return ''
    if isinstance(meaning, list):
        meaning = meaning[0]
    meaning = meaning.split('.')[0]
    meaning = meaning.split(';')[0]
    return meaning

def get_frequencies(doc):
    word_frequencies={}
    for token in doc:
        if token.text.lower() not in word_frequencies.keys():
            word_frequencies[token.text.lower()] = 1
        else:
            word_frequencies[token.text.lower()] += 1

    max_word_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_word_frequency
    #return word_frequencies
    return sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

def scores(sentence_tokens, word_frequencies):
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    return sentence_scores

def get_noun_chunks(doc, sorted_doc):
    noun_chunks = []
    for token in sorted_doc:
        chunk = doc[token.left_edge.i:token.right_edge.i+1]
        if len(chunk)>1 and not chunk[-1].is_punct:
            noun_chunks.append(chunk)
    return noun_chunks

"""
def get_noun_chunks(sentences):

    patterns=[

    #[{'DEP': {'IN': ['amod', 'advmod']}, 'OP':'*'},{'POS': 'DET', 'OP': '*'}, {'DEP':'attr', 'OP': '*'}, {'POS': 'NOUN'}],

    [{'DEP': 'predet', 'OP': '*'}, {'POS': {'IN':['DET','NUM']}, 'OP':'*'}, {'DEP': {'IN': ['quantmod','nummod']}, 'OP':'*'}, {'LOWER': {'IN': ['-', '–']}, 'OP': '*'},
    {'DEP': {'IN': ['amod', 'advmod', 'npadvmod']}, 'OP':'*'},{'LOWER': {'IN': ['-', '–']}, 'OP': '*'}, {'DEP': 'conj', 'OP': '*'}, {'DEP': {'IN': ['amod', 'advmod']}, 'OP': '*'},
    {'LOWER': {'IN': ['-', '–']}, 'OP': '*'}, {'DEP': 'conj', 'OP': '*'}, {'POS': 'ADJ', 'OP':'*'}, {'LOWER': {'IN': ['-', '–']}, 'OP': '*'}, {'DEP': 'compound', 'OP': '*'},
    {'POS': 'NOUN'}],

    [{'DEP': 'predet', 'OP': '*'}, {'POS': {'IN':['DET','NUM']}, 'OP':'*'}, {'DEP': {'IN': ['quantmod','nummod', 'np']}, 'OP':'*'}, {'LOWER': {'IN': ['-', '–']}, 'OP': '*'},
    {'ENT_TYPE':'QUANTITY', 'OP':'*'},{'LOWER': {'IN': ['-', '–']}, 'OP': '*'}, {'DEP': {'IN': ['amod', 'advmod']}, 'OP':'*'},{'LOWER': {'IN': ['-', '–']}, 'OP': '*'},
    {'DEP': 'conj', 'OP': '*'}, {'DEP': {'IN': ['amod', 'advmod']}, 'OP': '*'}, {'LOWER': {'IN': ['-', '–']}, 'OP': '*'}, {'DEP': 'conj', 'OP': '*'}, {'POS': 'ADJ', 'OP':'*'},
    {'LOWER': {'IN': ['-', '–']}, 'OP': '*'}, {'DEP': 'compound', 'OP': '*'}, {'POS': 'NOUN'}],

    [{'DEP': 'predet', 'OP': '*'}, {'POS': {'IN':['DET','NUM']}, 'OP':'*'}, {'DEP': {'IN': ['quantmod','nummod']}, 'OP':'*'}, {'LOWER': {'IN': ['-', '–']}, 'OP': '*'},
    {'ENT_TYPE':'QUANTITY', 'OP':'*'},{'LOWER': {'IN': ['-', '–']}, 'OP': '*'}, {'DEP': {'IN': ['amod', 'advmod']}, 'OP':'*'},{'LOWER': {'IN': ['-', '–']}, 'OP': '*'},
    {'DEP': 'conj', 'OP': '*'}, {'DEP': {'IN': ['amod', 'advmod']}, 'OP': '+'}, {'LOWER': {'IN': ['-', '–']}, 'OP': '*'}, {'DEP': 'conj', 'OP': '*'}, {'DEP': 'compound', 'OP': '*'},
    {'POS': 'NOUN'}],

    [{'DEP': 'predet', 'OP': '*'}, {'POS': {'IN':['DET','NUM']}, 'OP':'*'}, {'DEP': {'IN': ['quantmod','nummod']}, 'OP':'*'}, {'LOWER': {'IN': ['-', '–']}, 'OP': '*'},
    {'ENT_TYPE':'QUANTITY', 'OP':'*'},{'LOWER': {'IN': ['-', '–']}, 'OP': '*'}, {'DEP': {'IN': ['amod', 'advmod']}, 'OP':'*'}, {'LOWER': {'IN': ['-', '–']}, 'OP': '*'},
    {'DEP': 'conj', 'OP': '*'}, {'DEP': {'IN': ['amod', 'advmod']}, 'OP': '+'}, {'DEP': 'conj', 'OP': '*'}, {'LOWER': {'IN': ['-', '–']}, 'OP': '*'}, {'DEP': 'compound', 'OP': '*'}, {'POS': 'NOUN'}]

    ]
    # Add pattern to matcher
    matcher = Matcher(nlp.vocab)
    matcher.add("noun-phrases", patterns)
    expressions = []
    # call the matcher to find matches
    expressions = []
    for sentence in sentences:
        matches = matcher(sentence)
        spans = []
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]  # Get string representation
            spans.append(sentence[start:end])  # The matched span
        matches = filter_spans(spans)
        for match in matches:
            if match not in expressions and len(match)>1:
                expressions.append(match)

    return expressions
"""

def get_prep_sentences(sentences):
    patterns=[

    [{'OP': '*'}, {'POS': {'IN': ['VERB', 'AUX']}}, {'OP': '*'}, {'DEP': 'prep'}, {'OP': '*'}, {'POS': 'NOUN'}, {'OP': '*'},
    {'POS': {'IN': ['VERB', 'AUX']}}, {'OP': '*'}, {'DEP': {'IN': ['dobj', 'iobj', 'pobj', 'obj']}}, {'OP': '*'}]

    ]

    matcher = Matcher(nlp.vocab)
    matcher.add("prep-sentences", patterns)
    expressions = []
    # call the matcher to find matches
    expressions = []
    for sentence in sentences:
        matches = matcher(sentence)
        spans = []
        for match_id, start, end in matches:
            spans.append(sentence[start:end])  # The matched span
        matches = filter_spans(spans)
        for match in matches:
            if match not in expressions and len(match)>1:
                expressions.append(match)

    return expressions

def get_verb_sentences(sentences):

    #### WH expressions ####
    matcher = Matcher(nlp.vocab)

    patterns = [
    [{'OP': '*'}, {'POS': 'ADJ'}, {"OP": '*'}, {'POS': 'VERB', 'OP':'+'}, {'OP': '*'}, {'POS': 'ADP'}, {'OP':'*'}, {'DEP': {'IN': ['dobj', 'iobj', 'pobj', 'obj']}}, {'OP': '*'}]
    ]

    matcher.add("verb_sentences", patterns)
    expressions = []
    for sentence in sentences:
        matches = matcher(sentence)
        spans = []
        for match_id, start, end in matches:
            spans.append(sentence[start:end])  # The matched span
        matches = filter_spans(spans)
        for match in matches:
            if match not in expressions and len(match)>1:
                expressions.append(match)
    return expressions

def get_det_sentences(sentences):

    matcher = Matcher(nlp.vocab)

    patterns = [
    [{'OP': '*'}, {'POS': {'IN': ['DET', 'NUM']}}, {'OP': '*'}, {'POS': 'NOUN'}, {'POS': {'IN': ['VERB', 'AUX']}}, {"OP": '*'}, {'DEP': {'IN': ['dobj', 'iobj', 'pobj', 'obj']}}, {'OP': '*'}]
    ]

    matcher.add("subj_sentences", patterns)
    expressions = []
    for sentence in sentences:
        matches = matcher(sentence)
        spans = []
        for match_id, start, end in matches:
            spans.append(sentence[start:end])  # The matched span
        matches = filter_spans(spans)
        for match in matches:
            if match not in expressions and len(match)>1:
                expressions.append(match)
    return expressions
