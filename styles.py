from style_utils import *
from collections import OrderedDict
from pyinflect import getInflection
from string import punctuation
from random import randrange
from spacy.lang.en.stop_words import STOP_WORDS

stopwords=list(STOP_WORDS)

"""
Negativities
"""
def negativities(doc, sentence_tokens, noun_chunks, places):
    neg = ["It was neither ", ", nor ", ", but "]
    neg_text = ''
    keywords = {}
    i = 0
    places = [nlp(place)[0] for place in places if len(nlp(place))==1]
    for token in places:
        if i<3 and token.has_vector:
            similar_list = most_similar(token, pos=token.pos_)[2:4]
            similar_list = [word.capitalize() for word in similar_list if not word.isupper()]
            if token.text not in keywords.keys() and len(similar_list)>1:
                keywords[token.text] = similar_list + [token.text]
                i += 1
        else:
            break
    for token in doc:
        if i<5:
            similar_list = most_similar(token, pos=token.pos_)[2:4]
            if token.text.lower() not in keywords.keys() and len(similar_list)>1 and len(token._.definition)>0:
                keywords[token.text.lower()] = similar_list + [token._.definition[0].lower() + token._.definition[1:]]
                i += 1
        elif i<10:
            similar_list = most_similar(token, pos=token.pos_)[2:4]
            chunks = sorted([chunk for chunk in noun_chunks if token.text in chunk.text], key=len, reverse=True)
            if token.text.lower() not in keywords.keys() and len(similar_list)>1 and len(chunks)>0:
                keywords[token.text.lower()] = similar_list + [chunks[0].text[0].lower() + chunks[0].text[1:]]
                noun_chunks.remove(chunks[0])
                i += 1
        else:
            break
    for word in keywords.keys():
        neg_text += neg[0] + keywords[word][0] + neg[1] + keywords[word][1] + neg[2] + keywords[word][2] + '. '

    return neg_text

"""
Hesitation
"""
def hesitation(sorted_doc, doc, sentence_tokens, noun_chunks, places):

    nouns = []
    for token in sorted_doc:
        if token.has_vector and token.pos_=='NOUN' and token.dep_!='compound' and token.ent_type_ not in ['PERSON', 'PRODUCT'] and token not in nouns:
            nouns.append(token)

    verb_sentence = get_verb_sentences(sentence_tokens)[0]
    sentence_tokens.remove(verb_sentence)
    for chunk in noun_chunks:
        if chunk.text in verb_sentence.text:
            noun_chunks.remove(chunk)
    prep_sentence = get_prep_sentences(sentence_tokens)[0]
    chunk = next(chunk for chunk in noun_chunks if chunk.text in prep_sentence.text)
    sentence_tokens.remove(prep_sentence)
    noun_chunks.remove(chunk)

    expressions = ['I don\'t really know ', 'I\'m not sure of ', 'I\'m trying to remember ']
    r = randrange(0,3)
    hes_text = expressions[r]

    if len(places)>0:
        word = places[0]
        places.remove(word)
        similar_list = places[-3:]
        hes_text += 'where it happened...'
        for i in range(len(similar_list)-1):
            hes_text += similar_list[i] + ', '
        if similar_list:
            hes_text += similar_list[-1] + '? '
        hes_text += word + ', perhaps? There were...but what were there, though? '
    else:
        word = nouns[0]
        nouns.remove(word)
        similar_list = most_similar(word, pos=nouns.pos_)[2:5]
        hes_text += 'what happened...'
        for i in range(len(similar_list)-1):
            hes_text += similar_list[i] + ', '
        if similar_list:
            hes_text += similar_list[-1] + '? '
        hes_text += word.text + ', perhaps? There were...but what were there, though? '


    word = next(noun for noun in nouns if related_adjectives(noun, noun_chunks) and noun._.definition and len([chunk for chunk in noun_chunks if noun.text in chunk.text])>1)
    nouns.remove(word)
    similar_list = most_similar(word, pos='NOUN')[2:5]
    similar_list[0] = similar_list[0].capitalize()
    for i in range(len(similar_list)-1):
        hes_text += similar_list[i] + ', '
    hes_text += similar_list[-1] + '? ' + word.text.capitalize() + '? Yes, '

    hes_text += word._.definition[0].lower() + word._.definition[1:]

    chunks = sorted([chunk for chunk in noun_chunks if word.text in chunk.text], key=len, reverse=True)
    adjective = related_adjectives(word, chunks)[0]
    expression = next(c for c in chunks if adjective in c.text)
    noun_chunks.remove(expression)
    for sentence in sentence_tokens:
        if expression.text in sentence.text:
            sentence_tokens.remove(sentence)
    hes_text += ', and ' + adjective

    expressions = ['. I think that\'s how it was. ', '. I\'m pretty sure that\'s how it was. ', '. Yes, it was probably about that. ']
    r = randrange(0,3)
    hes_text += expressions[r]

    hes_text += expression.text.capitalize() + '. '

    expression = verb_sentence
    i = expression[0].i
    iadj = next(token.i for token in expression if token.pos_=='ADJ')
    iverb = next(token.i for token in expression if token.pos_=='VERB' and token.i>iadj)
    iadp = next(token.i for token in expression if token.pos_=='ADP' and token.i>iverb)
    iobj = next(token.i for token in expression if token.dep_ in ['dobj', 'iobj', 'pobj', 'obj'] and token.i>iadp)
    obj = doc[iobj]

    adjective = doc[iadj]
    sub_word = ''
    if not doc[iadj-1].is_punct:
        sub_word += ' '
    sub_word += adjective.text + ', ' + adjective.text + ', ' + adjective.text + ' '
    hes_text += doc[i:iadj].text + sub_word
    hes_text += doc[iadj+1:iverb].text
    if doc[iverb-1].text!=',':
        hes_text += '...'

    verb = doc[iverb]
    inflection = verb.tag_
    similar_list = most_similar(verb, pos='VERB')

    verbs = []
    i = 0
    for word in similar_list:
        similar_verb = getInflection(nlp(word)[0].lemma_, inflection)
        if i<2 and similar_verb:
            similar_verb = similar_verb[0]
            if similar_verb!=verb.text and similar_verb not in verbs:
                verbs.append(similar_verb)
                i += 1
    hes_text += ' ' + verbs[0] + '...no, no: ' + verb.text
    if doc[iverb+1].text not in punctuation:
        hes_text += ' '
    hes_text += doc[iverb+1:iadp].text

    expressions = [', I don\'t really know...', ', I guess...', ', maybe...']
    r = randrange(0,3)
    hes_text += expressions[r]

    word = doc[iobj]
    similar_list = most_similar(word, pos=word.pos_)[2:5]
    hes_text += doc[iadp:iobj].text + ' ' + similar_list[0] + '? '
    for w in similar_list[1:]:
        hes_text += doc[iadp:iobj].text + ' ' + w + '? '
    hes_text += 'rather...more precisely...' + doc[iadp:obj.right_edge.i].text
    if not doc[obj.right_edge.i].is_punct:
        hes_text += obj.right_edge.text
    hes_text += '. '

    sentence = prep_sentence
    i = sentence[0].i
    iverb = next(token.i for token in sentence if token.pos_=='VERB')
    iprep = next(token.i for token in sentence if token.dep_=='prep' and token.i>iverb)
    inoun = next(token.i for token in sentence if token.pos_=='NOUN' and token.i>iprep)
    iverb2 = next(token.i for token in sentence if (token.pos_ in ['VERB', 'AUX']) and token.i>inoun)
    iobj = next(token.i for token in sentence if (token.dep_ in ['dobj', 'iobj', 'pobj', 'obj']) and token.i>iverb2)
    obj = doc[iobj]
    adjectives = related_adjectives(doc[inoun], noun_chunks)[:2]
    if len(adjectives)<2:
        similar_list = most_similar(doc[inoun], pos='ADJ')[:2-len(adjectives)]
        for word in similar_list:
            adjectives.append(word)
    similar_list = []
    for adjective in adjectives:
        adjective = nlp(adjective)[0]
        similar_list.append(most_similar(adjective, pos='ADJ')[0])

    hes_text += doc[i:iverb+1].text + ', yes, that\'s right, ' + doc[iverb:iprep+1].text.lower() + ', no doubt, ' + doc[iprep+1:inoun+1].text + ' '
    if len(adjectives)>1 and len(similar_list)>1:
        hes_text += '(' + adjectives[0] + ' or ' + similar_list[0] + '?) ' + '(' + adjectives[1] + ' or ' + similar_list[1] + '?) '
    hes_text += doc[inoun+1:iverb2].text + ' probably ' + doc[iverb2:obj.right_edge.i].text + '. '

    #Last paraghraph
    sentence = get_det_sentences(sentence_tokens)[0]
    chunk = next(chunk for chunk in noun_chunks if chunk.text in sentence.text)

    hes_text += 'I rather think that '
    idet = next(token.i for token in sentence if token.pos_ in ['DET', 'NUM'])
    inoun = next(token.i for token in sentence if token.pos_=='NOUN' and token.i>idet)
    iverb = next(token.i for token in sentence if token.pos_ in ['VERB', 'AUX'] and token.i>inoun)
    iobj = next(token.i for token in sentence if token.dep_ in ['dobj', 'pobj', 'iobj', 'obj'] and token.i>iverb)
    obj = doc[iobj]
    hes_text += doc[idet:iverb].text[0].lower() + doc[idet:iverb].text[1:]
    if doc[iverb-1].text!=',':
        hes_text += '...'
    else:
        hes_text += ' '
    hes_text += 'I don\'t know...'
    hes_text += doc[iverb:obj.right_edge.i].text + '?'

    return(hes_text)
