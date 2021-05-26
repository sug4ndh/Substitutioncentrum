import json
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

logger = logging.getLogger('app')

def text_processing(documents):
    # remove common words and tokenize
    texts = [utils.simple_preprocess(document['text'])
             for document in documents]
    return texts

def load_data_index():
    data_path = os.path.abspath(os.path.dirname(__file__))
    data_index = np.load(os.path.join(data_path, "data/zipped_data.sav.npy"))
    logger.info("Data index loaded.")
    return data_index

def load_dictionary():
    data_path = os.path.abspath(os.path.dirname(__file__))
    dictionary = corpora.Dictionary.load_from_text(os.path.join(data_path, "data/dict.sav"))
    logger.info("Dictionary loaded.")
    return dictionary

def load_corpus():
    data_path = os.path.abspath(os.path.dirname(__file__))
    corpus = corpora.MmCorpus(os.path.join(data_path, "data/corpus.mm"))
    logger.info("Corpus loaded.")
    return corpus

def load_fasttext_model():
    data_path = os.path.abspath(os.path.dirname(__file__))
    ft_model = FastText.load(os.path.join(data_path, "data/fasttext/model-fasttext.sav"))
    logger.info("Fasttext model loaded.")
    return ft_model

def load_tfidf_model():
    data_path = os.path.abspath(os.path.dirname(__file__))
    tfidf = TfidfModel.load(os.path.join(data_path, "data/tfidf.sav"))
    logger.info("Tfidf model loaded.")
    return tfidf

def load_docsim(algo):
    data_path = os.path.abspath(os.path.dirname(__file__))
    if algo =='ft':
        docsim = SoftCosineSimilarity.load(os.path.join(data_path, "data/fasttext/ft_docsim_index.sav"))
        logger.info("fasttext docsim model loaded.")
    return docsim

def ft_softcossim(query, algo='ft'):
    # Compute Soft Cosine Measure between the query and the documents.
    tfidf = load_tfidf_model()
    dictionary = load_dictionary()
    docsim_index = load_docsim(algo)
    query = tfidf[dictionary.doc2bow(query.lower().split())]
    similarities = docsim_index[query]
    return similarities

def rev_results(x):
    return sorted(x, key=itemgetter(0), reverse=True)

def ft_gen_search_results(query):
    algo = "ft"
    cos_sim = ft_softcossim(query, algo)
    unsorted_sim = []
    search_res = []
    data_index = load_data_index()
    if query == None or query == "":
        logger.info("Empty search query submitted.")
        return ([])
    elif len(np.atleast_1d(cos_sim)) == 1:
        logger.info("No results found for the search query.")
        return ([])
    elif len(cos_sim) > 0:
        for i in range(len(cos_sim)):
            unsorted_sim.append((cos_sim[i], i))
        sorted_sim = rev_results(unsorted_sim)
        for i in range(len(sorted_sim)):
            if sorted_sim[i][0] > .2:
                search_res.append({data_index[sorted_sim[i][1]][0]:str(sorted_sim[i][0])})
        logger.info("Search query results found: "+str(len(search_res)))
        return search_res
    else:
        logger.info("No results found.")
        return ([])