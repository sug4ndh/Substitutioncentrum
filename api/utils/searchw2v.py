import os.path
import os
import logging
from gensim.models import Word2Vec
from .loader import load_tfidf_model, load_dictionary, load_docsim

logger = logging.getLogger('w2v')

def load_word2vec_model():
    '''Loads the saved word2vec model'''
    data_path = os.path.abspath(os.path.dirname(__file__))
    w2v_model = Word2Vec.load(os.path.join(data_path, "../data/word2vec/word2vec.sav"))
    logger.info("Word2Vec")
    return w2v_model

def w2v_softcossim(query, algo):
    # Compute Soft Cosine Measure between the query and the documents.
    tfidf = load_tfidf_model()
    dictionary = load_dictionary()
    docsim_index = load_docsim(algo)
    query = tfidf[dictionary.doc2bow(query.lower().split())]
    sims = docsim_index[query]
    return sims