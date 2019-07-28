from keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
import pandas as pd
from pprint import pprint
import seaborn as sb
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec, Phrases
import pickle
from keras.models import model_from_json

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
def vectorize_data(data, vocab: dict) -> list:
    keys = list(vocab.keys())
    filter_unknown = lambda word: vocab.get(word, None) is not None
    encode = lambda review: list(map(keys.index, filter(filter_unknown, review)))
    vectorized = list(map(encode, data))
    return vectorized

def preprocessing(input_review):
    
    stop_words = stopwords.words('english')
    data = input_review
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    data_words = list(sent_to_words(data))
    bigram = gensim.models.Phrases(data_words, min_count=5) 
    trigram = gensim.models.Phrases(bigram[data_words])  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_words_nostops = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in data_words]
    data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    embedding_vector_size = 256
    bigrams_model =  Word2Vec.load("Bigram_Word2VecModel2.model")
    X_data = bigram_mod[data_lemmatized]
    input_length = 300
    X_pad = pad_sequences(sequences=vectorize_data(X_data, vocab=bigrams_model.wv.vocab),maxlen=input_length,padding='post')
    return X_pad