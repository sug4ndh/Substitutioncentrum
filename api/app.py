from flask import Flask, request
from .utils.common import gen_search_results
import logging

app = Flask('app')
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

@app.route('/')
def welcome():
    '''Root page - displays various endpoints'''
    return 'ML model search serving API\nEndpoints:\n' \
           '\n 1. searchft - for using fasttext based search\n' \
           '\n 2. searchw2v - for using word2vec based search\n' \
           '\n 3. searchlsi - for using bow with lsi based search', 200

@app.route("/searchft", methods=["GET"])
def predict_ft():
    '''Calls fasttext based model to perform similarity search'''
    query = request.args.get('query', None)
    app.logger.debug("query ############ " + query)
    algo = 'ft'
    res = gen_search_results(query, algo)
    return {"result": res}, 200

@app.route("/searchw2v", methods=["GET"])
def predict_w2v():
    '''Calls word2vec based model to perform similarity search'''
    query = request.args.get('query', None)
    app.logger.debug("query ############ " + query)
    algo = 'w2v'
    res = gen_search_results(query, algo)
    return {"result": res}, 200

@app.route("/searchlsi", methods=["GET"])
def predict_lsi():
    '''Calls bow+lsi based model to perform similarity search'''
    query = request.args.get('query', None)
    app.logger.debug("query ############ " + query)
    algo = 'lsi'
    res = gen_search_results(query, algo)
    return {"result": res}, 200

@app.route("/health")
def health():
    return "OK", 200
