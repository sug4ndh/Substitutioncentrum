from operator import itemgetter
import numpy as np
import os
from .search_fasttext import ft_softcossim
from .searchw2v import w2v_softcossim
from .search_lsi import lsi_sim
import logging

logger = logging.getLogger('testsearch')

def rev_results(x):
    return sorted(x, key=itemgetter(0), reverse=True)

def load_data_index():
    data_path = os.path.abspath(os.path.dirname(__file__))
    data_index = np.load(os.path.join(data_path, "../data/zipped_data.sav.npy"))
    logger.info(data_path)
    logger.info("Data index loaded.")
    return data_index

def gen_search_results(query, algo='ft'):
    unsorted_sim = []
    search_res = []
    data_index = load_data_index()
    if algo =='ft':
        cos_sim = ft_softcossim(query, algo)
    if algo =='w2v':
        cos_sim = w2v_softcossim(query, algo)
    if algo == 'lsi':
        cos_sim = lsi_sim(query, algo)

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
            if sorted_sim[i][0] > 0:
                #search_res.append({data_index[sorted_sim[i][1]][0]:str(sorted_sim[i][0])})
                search_res.append(data_index[sorted_sim[i][1]][0])
        logger.info("Search query results found: "+str(len(search_res)))
        return search_res
    else:
        logger.info("No results found.")
        return ([])