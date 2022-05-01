import numpy as np 
from src.config import TOTAL_NUMBER_OF_DOCUMENTS, STARTING_DOCUMENT_INDEX
from src.data_processing import process_query
from collections import Counter 


def compute_tf_idf_scores(inverted_index, query, idf, doc_ids, Lengths):
    scores = np.zeros(len(doc_ids))
    query = query.split(' ')
    tf_query = Counter(query)
    for t in query:
        if t in inverted_index:
            w_tq = tf_query[t] * idf[t]
            postings_list = inverted_index[t]
            for (d, tf_td) in postings_list:
                scores[d] += tf_td * w_tq 
    for i in range(len(doc_ids)):
        scores[i] /= Lengths[i]
    
    return scores 

def compute_tf_scores(inverted_index, query, doc_ids, Lengths):
    scores = np.zeros(len(doc_ids))
    query = query.split(' ')
    tf_query = Counter(query)
    for t in query:
        if t in inverted_index:
            w_tq = tf_query[t] * 1. # No IDF multiplication 
            postings_list = inverted_index[t]
            for (d, tf_td) in postings_list:
                scores[d] += tf_td * w_tq 
    for i in range(len(doc_ids)):
        scores[i] /= Lengths[i]
    
    return scores 

def compute_boolean_scores(inverted_index, query, doc_ids): 
    scores = np.zeros(len(doc_ids))
    query = query.split(' ')
    query_norm = 1 / len(query)
    for t in query:
        if t in inverted_index:
            postings_list = inverted_index[t]
            for (d, _) in postings_list:
                scores[d] += d * query_norm 

    # Tolerance of 0.75, If a document has more than 75% of the non stop-words terms in the query, it gets a score of 1 
    scores = scores[ scores > 0.75]
    scores = [int(s) for s in scores]
    return scores


def get_top_k_documents(doc_ids, scores, k):
    idx = (-scores).argsort()[:k]
    top_k_docs = [(doc_ids[i], scores[i]) for i in idx]
    return top_k_docs

def generate_log_string(top_k_docs, query_id):
    log_string = ''
    for rank, (doc_id, score)  in enumerate(top_k_docs):
        log_string += f'{query_id} 0 {doc_id} {rank+1} {score} tf-idf\n'
    return log_string

def retrieve_documents(method, query, inverted_index, idf, doc_ids, Lengths):
    query, query_id = process_query(query)
    if method == 'tf-idf':
        scores = compute_tf_idf_scores(inverted_index, query, idf, doc_ids, Lengths)
        top_k_docs = get_top_k_documents(doc_ids, scores, 50)        

    elif method =='tf':
        scores = compute_tf_scores(inverted_index, query, doc_ids, Lengths)
        top_k_docs = get_top_k_documents(doc_ids, scores, 50)
    
    elif method =='boolean':
        scores = compute_boolean_scores(inverted_index, query, doc_ids)
        top_k_docs = get_top_k_documents(doc_ids, scores, 50)

    log_string = generate_log_string(top_k_docs, query_id)
    
    return log_string 

        