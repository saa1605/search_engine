import numpy as np 
from src.config import TOTAL_NUMBER_OF_DOCUMENTS, STARTING_DOCUMENT_INDEX
from collections import Counter 

def compute_tf_idf_scores(inverted_index, query, idf, doc_ids, Lengths):
    scores = np.zeros(len(doc_ids))
    query = query.split(' ')
    tf_query = Counter(query)
    for t in query:
        w_tq = tf_query[t] * idf[t]
        postings_list = inverted_index[t]
        for (d, tf_td) in postings_list:
            scores[int(d)-STARTING_DOCUMENT_INDEX] += tf_td * w_tq 
    for i in range(len(doc_ids)):
        scores[i] /= Lengths[str(STARTING_DOCUMENT_INDEX+i)]
    
    return scores 
        