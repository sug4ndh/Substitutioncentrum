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
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from gensim.similarities import MatrixSimilarity, WmdSimilarity, SoftCosineSimilarity
from gensim.similarities.index import AnnoyIndexer
import logging

logger = logging.getLogger('app')

def read_data(training_data_path):
    lst_chem = []
    documents = []
    data_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(data_path, str(training_data_path)), mode='r', errors='ignore') as json_file:
        for docu in json_file:
            lst_chem.append(json.loads(docu))
    for ads in lst_chem[0]:
        documents.append({"text": 'Title : ' + ads['Title'] + ', ' \
                         + 'Summary : ' + ads['Summary'] + ', ' \
                         + 'Description : ' + ads['Description'] + ', ' \
                         + 'ReplacingAlternatives : ' +  str(ads['ReplacingAlternatives']) + ', ' \
                         + 'Categories : ' + str(ads['Categories']),\
                         "title": ads['Title']

                          })
    return documents

def read_initial_data():
    # importing chemsec data
    data_path = os.path.abspath(os.path.dirname(__file__))
    lst_chem = []
    documents = []
    with open(os.path.join(data_path, "data/chemsec.json"), mode='r', errors='ignore') as json_file:
        for docu in json_file:
            lst_chem.append(json.loads(docu))
    # importing subsport data
    lst_subsport = []
    with open(os.path.join(data_path, "data/subsport_data_final.json"), mode='r', errors='ignore') as json_file:
        for docu in json_file:
            lst_subsport.append(json.loads(docu))
    for ads in lst_chem[0]:
        documents.append({"text": 'Title : ' + ads['Title'] + ', ' \
                         + 'Summary : ' + ads['Summary'] + ', ' \
                         + 'Description : ' + ads['Description'] + ', ' \
                         + 'ReplacingAlternatives : ' +  str(ads['ReplacingAlternatives']) + ', ' \
                         + 'Categories : ' + str(ads['Categories']),\
                         "title": ads['Title']

                          })
    for ads in lst_subsport[0]['data']:
        documents.append({"text": 'Title : ' + str(ads['title']) + ', ' \
                          + 'Summary : ' + str(ads['abstract']) + ', '  \
                          + 'Description : ' + str(ads['details']) + ', ' \
                          + 'ReplacingAlternatives : ' + str(ads['alternative_substance']) + ', ' \
                          + 'Substituted_substance : ' + str(ads['substituted_substance']) + ',' \
                          + 'Categories : ' + str(ads['use']),
                          "title": str(ads['title'])}
                         )
    return documents

def text_processing(documents):
    # remove common words and tokenize
    texts = [utils.simple_preprocess(document['text'])
             for document in documents]
    return texts

def create_and_save_dic_and_corpus(texts):
    texts = [[token for token in text]
             for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    data_path = os.path.abspath(os.path.dirname(__file__))
    dictionary.save_as_text(os.path.join(data_path, "data/dict.sav"))
    corpora.MmCorpus.serialize(os.path.join(data_path, "data/corpus.mm"), corpus)
    return dictionary, corpus

# Create tfidf model and train word2vec model
def train_tfidf_model(corpus):
    tfidf = TfidfModel(corpus)
    data_path = os.path.abspath(os.path.dirname(__file__))
    tfidf.save(os.path.join(data_path, "data/tfidf.sav"))
    return tfidf

def train_word2vec_model(texts):
    w2v_model = Word2Vec(texts, workers=cpu_count(), min_count=1, size=300, seed=12345)
    data_path = os.path.abspath(os.path.dirname(__file__))
    w2v_model.save(os.path.join(data_path, "data/word2vec.sav"))
    return w2v_model

def create_word_emb_sim_index(w2v_model=None, tfidf=None, corpus=None):
    documents = read_initial_data()
    texts = text_processing(documents)
    _, corpus = create_and_save_dic_and_corpus(texts)
    w2v_model = train_word2vec_model(texts)
    similarity_index = WordEmbeddingSimilarityIndex(w2v_model.wv)
    return similarity_index

def create_sparse_sim_matrix(similarity_index=None, dictionary=None, tfidf=None, corpus=None):
    documents = read_initial_data()
    texts = text_processing(documents)
    dictionary, corpus = create_and_save_dic_and_corpus(texts)
    tfidf = train_tfidf_model(corpus)
    similarity_index = create_sim_index()
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)
    return similarity_matrix

def create_sim_index(w2v_model=None, dictionary=None, tfidf=None, corpus=None):
    documents = read_initial_data()
    texts = text_processing(documents)
    _, corpus = create_and_save_dic_and_corpus(texts)
    similarity_matrix = create_sparse_sim_matrix()
    docsim_index = SoftCosineSimilarity(corpus, similarity_matrix)
    data_path = os.path.abspath(os.path.dirname(__file__))
    docsim_index.save(os.path.join(data_path, "data/docsim_index.sav"))
    return docsim_index

def load_dictionary():
    data_path = os.path.abspath(os.path.dirname(__file__))
    dictionary = corpora.Dictionary.load_from_text(os.path.join(data_path, "data/dict.sav"))
    return dictionary

def load_corpus():
    data_path = os.path.abspath(os.path.dirname(__file__))
    corpus = corpora.MmCorpus(os.path.join(data_path, "data/corpus.mm"))
    return corpus

def load_word2vec_model():
    data_path = os.path.abspath(os.path.dirname(__file__))
    w2v_model = Word2Vec.load(os.path.join(data_path, "data/word2vec.sav"))
    return w2v_model

def load_tfidf_model():
    data_path = os.path.abspath(os.path.dirname(__file__))
    tfidf = TfidfModel.load(os.path.join(data_path, "data/tfidf.sav"))
    return tfidf

def softcossim(query):
    # Compute Soft Cosine Measure between the query and the documents.
    data_path = os.path.abspath(os.path.dirname(__file__))
    tfidf_path = os.path.join(data_path, "data/tfidf.sav")
    tfidf = TfidfModel.load(tfidf_path)
    dictionary = corpora.Dictionary.load_from_text(os.path.join(data_path, "data/dict.sav"))
    docsim_index = SoftCosineSimilarity.load(os.path.join(data_path, "data/docsim_index.sav"))
    query = tfidf[dictionary.doc2bow(query.lower().split())]
    similarities = docsim_index[query]
    return similarities

def rev_results(x):
    return sorted(x, key=itemgetter(0), reverse=True)

def gen_search_results(query):
    documents = read_initial_data()
    cos_sim = softcossim(query)
    #print(cos_sim, type(cos_sim))
    #print(cos_sim)
    unsorted_sim = []
    search_res = []
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
                search_res.append({documents[sorted_sim[i][1]]["title"]:str(sorted_sim[i][0])})
        logger.info("Search query results found: "+str(len(search_res)))
        return search_res
    else:
        logger.info("No results found.")
        return ([])