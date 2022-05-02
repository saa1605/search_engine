import re 
from collections import Counter
import sys 
sys.path.append('..')
import json 
import numpy as np 
from src.data_processing import collect_document_ids, split_into_documents, clean_entry, remove_stopwords, read_stopwords_file, process_query
from src.inverted_index import create_inverted_index, calculate_idf, save_inverted_index, get_document_norm
from src.config import TOTAL_NUMBER_OF_DOCUMENTS, STARTING_DOCUMENT_INDEX, CORPUS_PATH, QUERY_PATH
from src.document_ranking import compute_tf_idf_scores, retrieve_documents, generate_log_string


def main():
    corpus = split_into_documents(CORPUS_PATH)
    doc_ids = collect_document_ids(corpus)

    with open(QUERY_PATH) as f:
        query_string = f.read()
    queries = re.split('</top>\n', query_string)[:-1]

    print("Corpus parsed, creating inverted index")

    inverted_index, Lengths = create_inverted_index(corpus, doc_ids)

    print("Inverted index created, getting document normalization constants...")
    
    norms = get_document_norm(corpus)

    print("Normalization Constants Calculated, calculating Document Frequencies...")

    idf = calculate_idf(inverted_index, TOTAL_NUMBER_OF_DOCUMENTS)

    method = 'discounted-relevance-feedback'

    print(f"Retrieving documents using {method}")

    log_string = ''
    for query in queries:
        query, query_id = process_query(query)
        top_k_docs = retrieve_documents(method, corpus, query, inverted_index, idf, doc_ids, norms, Lengths)
        log_string += generate_log_string(top_k_docs, query_id)

    print(f"Document Retrieved, Writing to log file")

    with open('../logs/log.txt', 'w') as f:
        f.write(log_string)

if __name__ == '__main__':
    main()