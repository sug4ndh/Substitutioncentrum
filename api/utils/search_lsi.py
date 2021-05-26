from operator import itemgetter
import numpy as np
import os
from gensim import similarities
from gensim.models import LsiModel
from .loader import load_docsim, load_dictionary

def load_lsi_model():
    data_path = os.path.abspath(os.path.dirname(__file__))
    lsi_model = LsiModel.load(os.path.join(data_path, "../data/lsi/lsimodel.sav"))
    return lsi_model

def lsi_sim(query, algo):
    dictionary = load_dictionary()
    vec_bow = dictionary.doc2bow(query.lower().split())
    lsi_model = load_lsi_model()
    vec_lsi = lsi_model[vec_bow]
    lsi_index = load_docsim(algo)
    sims = lsi_index[vec_lsi]
    return sims

