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
    neg = ["It was neither", ", nor", ", but "]
    neg_text = ''
    keywords = {}
    i = 0
    places = [nlp(place)[0] for place in places if len(nlp(place))==1]
    for token in places:
        if i<1 and token.has_vector:
            similar_list = most_similar(token, pos=token.pos_)[1:3]
            similar_list = [" " + word.capitalize() for word in similar_list]
            if token.text not in keywords.keys() and len(similar_list)>1:
                keywords[token.text] = similar_list + [token.text]
                i += 1
        else:
            break
    for token in doc:
        if i<4:
            similar_list = most_similar(token, pos=token.pos_)[3:5]
            similar_list = [(" " + word) for word in similar_list]
            if token.text not in keywords.keys() and len(similar_list)>1 and len(token._.definition)>0:
                keywords[token.text] = similar_list + [token._.definition[0].lower() + token._.definition[1:]]
                i += 1
        if i<10:
            chunks = sorted([chunk for chunk in noun_chunks if token.text in chunk.text], key=len, reverse=True)
            if chunks:
                valid_chunk = True
                chunk = chunks[0]
                noun_chunks.remove(chunk)
                similar_chunks = ["", ""]
                for tk in chunk:
                    similar_list = most_similar(tk, pos=tk.pos_)[2:4] if not tk.is_stop else [tk.text, tk.text]
                    if tk.is_title:
                        similar_list = [word.capitalize() for word in similar_list]
                    if len(similar_list)>1:
                        similar_chunks[0] += ' ' + similar_list[0] if similar_list[0][0] not in ["’", "''"] else similar_list[0]
                        similar_chunks[1] += ' ' + similar_list[1] if similar_list[1][0] not in ["’", "''"] else similar_list[1]
                    else:
                        valid_chunk = False
                if valid_chunk and token.text not in keywords.keys():
                    keywords[token.text] = [similar_chunks[0], similar_chunks[1], chunk.text]
                    i+= 1
        else:
            break
    for word in keywords.keys():
        neg_text += neg[0] + keywords[word][0] + neg[1] + keywords[word][1] + neg[2] + keywords[word][2] + '. '

    return neg_text

"""
Hesitation
"""
def hesitation(sorted_doc, doc, sentence_tokens, noun_chunks, places):

    keywords = []
    for token in sorted_doc:
        if token.has_vector and token.pos_=='NOUN' and token.dep_!='compound' and token.ent_type_ not in ['PERSON', 'PRODUCT'] and token not in keywords:
            keywords.append(token)

    expressions = ['I don\'t really know ', 'I\'m not sure of ', 'I\'m trying to remember ', 'Well, I heard something about ']
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
        word = keywords[0]
        keywords.remove(word)
        similar_list = most_similar(word, pos=keywords.pos_)[2:5]
        hes_text += 'what happened...'
        for i in range(len(similar_list)-1):
            hes_text += similar_list[i] + ', '
        if similar_list:
            hes_text += similar_list[-1] + '? '
        hes_text += word.text + ', perhaps? There were...but what were there, though? '


    word = next(w for w in keywords if related_adjectives(w, noun_chunks) and w._.definition and len([chunk for chunk in noun_chunks if w.text in chunk.text])>1)
    keywords.remove(word)
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

    expressions = ['. I think that\'s how it was. ', '. I\'m pretty sure that\'s how it was. ', '. Yes, it was probably about that. ', '. Well, something like that. ']
    r = randrange(0,3)
    hes_text += expressions[r]

    hes_text += expression.text.capitalize() + '. '

    verb_sentence = get_verb_sentences(sentence_tokens)[0]
    sentence_tokens.remove(verb_sentence)
    for chunk in noun_chunks:
        if chunk.text in verb_sentence.text:
            noun_chunks.remove(chunk)
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

    expressions = [", I don't really know...", ", I'm still on the fence...", "I guess..."]
    r = randrange(0,3)
    hes_text += expressions[r]

    word = doc[iobj]
    similar_list = most_similar(word, pos=word.pos_)[2:5]
    hes_text += doc[iadp:iobj].text + ' ' + similar_list[0] + '?'
    for w in similar_list[1:]:
        hes_text += ' ' + w + '? '
    hes_text += 'rather...more precisely...' + doc[iadp:obj.right_edge.i].text
    if not doc[obj.right_edge.i].is_punct:
        hes_text += ' ' + obj.right_edge.text
    hes_text += '. '

    prep_sentence = get_prep_sentences(sentence_tokens)[0]
    chunk = next(chunk for chunk in noun_chunks if chunk.text in prep_sentence.text)
    sentence_tokens.remove(prep_sentence)
    noun_chunks.remove(chunk)
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
        hes_text += '(' + adjectives[0] + ' or ' + similar_list[0] + '?) ' + '(' + adjectives[1] + ' or ' + similar_list[1] + '?) Anyway...'
    hes_text += doc[iverb:iverb2].text + '...probably...' + doc[iverb2:obj.right_edge.i+1].text + '. '

    #Last paraghraph
    sentence = get_det_sentences(sentence_tokens)[0]
    hes_text += 'I rather think that...'
    iroot = sentence.root.i
    hes_text += doc[sentence[0].i:iroot].text.strip(" ()\"")
    if doc[iroot-1].text!=',':
        hes_text += '...'
    else:
        hes_text += ' '
    hes_text += 'I don\'t know...'
    iobj = next(token.i for token in sentence if token.dep_ in ['dobj', 'iobj', 'pobj', 'obj'] and token.i>iroot)
    hes_text += doc[iroot:iobj+1].text + '?'

    return(hes_text)
