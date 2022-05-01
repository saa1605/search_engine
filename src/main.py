import re 
from collections import Counter
import sys 
sys.path.append('..')
import json 
import numpy as np 
from src.data_processing import collect_document_ids, split_into_documents, clean_entry, remove_stopwords, read_stopwords_file, process_query
from src.inverted_index import create_inverted_index, calculate_idf, save_inverted_index
from src.config import TOTAL_NUMBER_OF_DOCUMENTS, STARTING_DOCUMENT_INDEX, CORPUS_PATH, QUERY_PATH
from src.document_ranking import compute_tf_idf_scores, retrieve_documents




def main():
    corpus = split_into_documents(CORPUS_PATH)
    doc_ids = collect_document_ids(corpus)

    with open(QUERY_PATH) as f:
        query_string = f.read()
    queries = re.split('</top>\n', queries)[:-1]

    inverted_index, Lengths = create_inverted_index(corpus, doc_ids)

    # Boolean Retrieval 

    # TF Retrieval 

    # TF-IDF Retrieval
    idf = calculate_idf(inverted_index, TOTAL_NUMBER_OF_DOCUMENTS)
    log_string = ''
    for query in queries:
        log_string += retrieve_documents('tf-idf', query, inverted_index, idf, doc_ids, Lengths)
    with open('tf_idf_log_file.txt', 'w') as f:
        f.write(log_string)

    # Relevance Feedback 
    