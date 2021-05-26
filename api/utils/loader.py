from operator import itemgetter
import os.path
import nltk
import numpy as np
import os
import gensim
from gensim import corpora, utils
from multiprocessing import cpu_count
from gensim.models import TfidfModel
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from gensim.similarities import MatrixSimilarity, WmdSimilarity, SoftCosineSimilarity
from gensim.similarities.index import AnnoyIndexer

import logging

logger = logging.getLogger('loader')

def text_processing(documents):
    '''Processing the information received from the database,
    remove common words and tokenize'''
    texts = [utils.simple_preprocess(document['text'])
             for document in documents]
    return texts

def load_dictionary():
    '''Loads the saved dictionary created while training the models'''
    data_path = os.path.abspath(os.path.dirname(__file__))
    dictionary = corpora.Dictionary.load_from_text(os.path.join(data_path, "../data/dict.sav"))
    logger.info("Dictionary loaded.")
    return dictionary

def load_corpus():
    '''Loads the saved corpus created while training the models'''
    data_path = os.path.abspath(os.path.dirname(__file__))
    corpus = corpora.MmCorpus(os.path.join(data_path, "../data/corpus.mm"))
    logger.info("Corpus loaded.")
    return corpus

def load_tfidf_model():
    '''Loads the saved tfidf model'''
    data_path = os.path.abspath(os.path.dirname(__file__))
    tfidf = TfidfModel.load(os.path.join(data_path, "../data/tfidf.sav"))
    logger.info("Tfidf model loaded.")
    return tfidf

def load_docsim(algo):
    '''Loads the softcosine similarity index/matrix for the model passed as the argument'''
    data_path = os.path.abspath(os.path.dirname(__file__))
    if algo =='ft':
        docsim = SoftCosineSimilarity.load(os.path.join(data_path, "../data/fasttext/ft_docsim_index.sav"))
        logger.info("fasttext docsim index loaded.")
    if algo =='w2v':
        docsim = SoftCosineSimilarity.load(os.path.join(data_path, "../data/word2vec/w2v_docsim_index.sav"))
        logger.info("word2vec docsim index loaded.")
    if algo =='lsi':
        docsim = MatrixSimilarity.load(os.path.join(data_path, "../data/lsi/lsi_sim_matrix.sav"))
        logger.info("lsi docsim index loaded.")
    return docsim
