import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from spacy.pipeline import DependencyParser
from spacy.tokens import Token
from spacy.attrs import *
from style_utils import *
from styles import *

stopwords=list(STOP_WORDS)
punctuation=punctuation + '\n'

set_frequency = lambda token: frequencies[token.text]
Token.set_extension('freq', default=0)
set_definition = lambda token: get_definition(token)
Token.set_extension('definition', default='')

def prepare_text(text):
    nlp = spacy.load('en_core_web_lg')

    doc  = nlp(text)

    word_frequencies = get_frequencies(doc)
    frequencies = dict(word_frequencies)

    sub_doc = []
    for token in doc:
        if (len(token.text)>2) and (token.text.lower() not in stopwords) and (token.text not in punctuation)\
        and token.pos_ in ['NOUN', 'ADJ'] and token.dep_!='compound' and token.ent_type_ not in ['PERSON', 'PRODUCT']:
            token._.freq = frequencies[token.text.lower()]
            token._.definition = get_definition(token)
            if token.lemma_ not in [token.lemma_ for token in sub_doc]:
                sub_doc.append(token)
        else:
            token._.freq = 0
            token._.definition = ''

    sentence_tokens = [sent for sent in doc.sents if len(sent)>1]
    sentence_scores = scores(sentence_tokens, dict(word_frequencies))
    sentence_tokens = sorted(sentence_tokens, key=lambda x: sentence_scores[x], reverse=True)

    sorted_doc = sorted(sub_doc, key=lambda x: x._.freq, reverse=True)

    noun_chunks = get_noun_chunks(doc, sorted_doc)

    # Get places (organizations or geographic) ordered by frequency
    places = {}
    for ent in doc.ents:
        if (ent.label_ in ['GPE', 'ORGANIZATION']) and (ent.text[0].isupper()):
            clean_text = ent.text.replace('’s','')
            if clean_text not in places.keys():
                places[clean_text] = 1
            else:
                places[clean_text] += 1
    places = OrderedDict(sorted(places.items(), key=operator.itemgetter(1), reverse=True))
    places = list(places.keys())

    return (sorted_doc, doc, sentence_tokens, noun_chunks, places)

def apply_style(sorted_doc, doc, sentence_tokens, noun_chunks, places, style):

    #(sorted_doc, doc, sentence_tokens, noun_chunks, places) = prepare_text(text)

    stylized_text = ''

    if style=='negativities':

        #try:
        stylized_text = negativities(sorted_doc, sentence_tokens, noun_chunks, places)
        #except:
        #    pass

    elif style=='hesitation':

        #try:
        stylized_text = hesitation(sorted_doc, doc, sentence_tokens, noun_chunks, places)
        #except:
        #   pass

    return stylized_text
