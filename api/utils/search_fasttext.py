import os.path
import os
import logging
from gensim.models.fasttext import FastText
from .loader import load_tfidf_model, load_dictionary, load_docsim

logger = logging.getLogger('ft')

def load_fasttext_model():
    '''Loads the saved fasttext model'''
    data_path = os.path.abspath(os.path.dirname(__file__))
    ft_model = FastText.load(os.path.join(data_path, "../data/fasttext/fasttext.sav"))
    logger.info("Fasttext model loaded.")
    return ft_model

def ft_softcossim(query, algo):
    # Compute Soft Cosine Measure between the query and the documents.
    tfidf = load_tfidf_model()
    dictionary = load_dictionary()
    docsim_index = load_docsim(algo)
    query = tfidf[dictionary.doc2bow(query.lower().split())]
    sims = docsim_index[query]
    return sims