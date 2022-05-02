import numpy as np 
from src.config import TOTAL_NUMBER_OF_DOCUMENTS, STARTING_DOCUMENT_INDEX
import src.config as config
from src.data_processing import process_query, remove_stopwords, read_stopwords_file, clean_entry
from collections import Counter 


def compute_tf_idf_scores(inverted_index, query, idf, doc_ids, norms, Lengths):
    scores = np.zeros(len(doc_ids))
    tf_query = Counter(query)
    for t in tf_query.keys():
        if t in inverted_index:
            w_tq = ( tf_query[t] ) * (idf[t] ** 2 )
            postings_list = inverted_index[t]
            for (d, tf_td) in postings_list:
                # Using the median of norms worked better than the respective norm 
                scores[d] += (tf_td / config.norm)  * w_tq 
    for i in range(len(doc_ids)):
        scores[i] /= Lengths[i]
    
    return scores 

def compute_relevance_feedback_scores(inverted_index, query, relevant_docs, idf, doc_ids, norms, Lengths):
    scores = np.zeros(len(doc_ids))
    tf_query = Counter(query)
    
    for t in tf_query.keys():
        if t in inverted_index:
            w_tq = ( tf_query[t] ) * (idf[t] ** 2 )
            postings_list = inverted_index[t]
            for (d, tf_td) in postings_list:
                # Using the median of norms worked better than the respective norm 
                scores[d] += (tf_td / config.norm)  * w_tq 
    for document in relevant_docs:
        tf_doc = Counter(document)
        for t in tf_doc.keys():
            if t in inverted_index:
                w_tq = ( tf_doc[t] ) * (idf[t] ** 2 )
                postings_list = inverted_index[t]
                for (d, tf_td) in postings_list:
                    # Using the median of norms worked better than the respective norm 
                    scores[d] += (tf_td / config.norm)  * w_tq 
                    
    for i in range(len(doc_ids)):
        scores[i] /= Lengths[i]
    
    return scores 

def compute_discounted_relevance_feedback_scores(inverted_index, query, relevant_docs, idf, doc_ids, norms, Lengths):
    scores = np.zeros(len(doc_ids))
    tf_query = Counter(query)
    for t in tf_query.keys():
        if t in inverted_index:
            w_tq = ( tf_query[t] ) * (idf[t] ** 2 )
            postings_list = inverted_index[t]
            for (d, tf_td) in postings_list:
                # Using the median of norms worked better than the respective norm 
                scores[d] += config.alpha*(tf_td / config.norm)  * w_tq 
    for enum, document in enumerate(relevant_docs):
        tf_doc = Counter(document)
        for t in tf_doc.keys():
            if t in inverted_index:
                w_tq = ( tf_doc[t] ) * (idf[t] ** 2 )
                postings_list = inverted_index[t]
                for (d, tf_td) in postings_list:
                    # Discount factor scaled by rank of document in question 
                    scores[d] += ((1 - config.alpha)**(enum + 1))*(tf_td / config.norm)  * w_tq 
                    
    for i in range(len(doc_ids)):
        scores[i] /= Lengths[i]
    
    return scores 

def compute_tf_scores(inverted_index, query, doc_ids, Lengths):
    scores = np.zeros(len(doc_ids))
    tf_query = Counter(query)
    for t in tf_query.keys():
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
    top_k_docs = [(i, doc_ids[i], scores[i]) for i in idx]
    return top_k_docs

def generate_log_string(top_k_docs, query_id, method):
    log_string = ''
    for rank, (relative_index, doc_id, score)  in enumerate(top_k_docs):
        log_string += f'{query_id} 0 {doc_id} {rank+1} {score} {method}\n'
    return log_string

def retrieve_documents(method, corpus, query, inverted_index, idf, doc_ids, norms, Lengths):  
    if method == 'tf-idf':
        print("here")
        scores = compute_tf_idf_scores(inverted_index, query, idf, doc_ids, norms, Lengths)
        top_k_docs = get_top_k_documents(doc_ids, scores, 50)        

    elif method =='tf':
        scores = compute_tf_scores(inverted_index, query, doc_ids, Lengths)
        top_k_docs = get_top_k_documents(doc_ids, scores, 50)
    
    elif method =='boolean':
        scores = compute_boolean_scores(inverted_index, query, doc_ids)
        top_k_docs = get_top_k_documents(doc_ids, scores, 50)

    elif method == 'relevance-feedback':
        scores = compute_tf_idf_scores(inverted_index, query, idf, doc_ids, norms, Lengths)
        relevant_docs = get_top_k_documents(doc_ids, scores, 5)
        stopwords_list = read_stopwords_file('stopwords.txt')
        relevant_docs_processed = []
        for (relative_index, document_id, score) in relevant_docs:
            extra_document = corpus[relative_index]
            # Clean document 
            extra_document = extra_document.split(" ")

            # Remove stopwords from document 
            extra_document = remove_stopwords(extra_document, stopwords_list)
            extra_document = [token.lower() for token in extra_document]
            
            extra_document = ' '.join(extra_document)
        
            # Clean document 
            extra_document = clean_entry(extra_document)

            # Tokenize document 
            tokens = extra_document.split(" ")

            relevant_docs_processed.append(tokens)

        scores = compute_relevance_feedback_scores(inverted_index, query, relevant_docs_processed, idf, doc_ids, norms, Lengths)
        top_k_docs = get_top_k_documents(doc_ids, scores, 50)        
    
    elif method == 'discounted-relevance-feedback':
        scores = compute_tf_idf_scores(inverted_index, query, idf, doc_ids, norms, Lengths)
        relevant_docs = get_top_k_documents(doc_ids, scores, 5)
        stopwords_list = read_stopwords_file('stopwords.txt')
        relevant_docs_processed = []
        for (relative_index, document_id, score) in relevant_docs:
            extra_document = corpus[relative_index]
            # Clean document 
            extra_document = extra_document.split(" ")

            # Remove stopwords from document 
            extra_document = remove_stopwords(extra_document, stopwords_list)
            extra_document = [token.lower() for token in extra_document]
            
            extra_document = ' '.join(extra_document)
        
            # Clean document 
            extra_document = clean_entry(extra_document)

            # Tokenize document 
            tokens = extra_document.split(" ")

            relevant_docs_processed.append(tokens)

        scores = compute_discounted_relevance_feedback_scores(inverted_index, query, relevant_docs_processed, idf, doc_ids, norms, Lengths)
        top_k_docs = get_top_k_documents(doc_ids, scores, 50)    

    return top_k_docs 

        