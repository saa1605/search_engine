import re
from collections import Counter 
import numpy as np 

def collect_document_ids(corpus):
    '''collect a single list of document_ids given the database file'''
    document_ids = []
    for document in corpus: 

        pattern = "\.U\n(.*?)\n"
        doc_id_find = re.search(pattern, document)
        if doc_id_find:
            doc_id = doc_id_find.group(1)
            document_ids.append(doc_id)
    return document_ids


def split_into_documents(database_file):
    '''Splits the singular corpus into an array of documents'''
    database_file = open(database_file)
    data = database_file.read()
    pattern = re.compile("\.I\s\d*\n")
    corpus = re.split(pattern, data)[1:]  # The first item is ''
    return corpus



def clean_entry(entry):
    '''Clean a single document by removing headers, punctuation, tabs, spaces, newlines, and digits'''
    # Remove . Headers
    entry = re.sub('\.[A-Z]', '', entry)

    # Remove punctuation
    entry = re.sub(r'[^\w\s]', ' ', entry)

    # Remove \n, \r and \t
    entry = re.sub(r'[\n\t\r]', ' ', entry)

    # Remove digits
    entry = re.sub(r'[\d]', '', entry)

    # Remove double spaces
    entry = re.sub(' +', ' ', entry)

    # Remove isolated small case letters
    entry = re.sub(r"\b\d+\b *|\b[a-z]\b *","",entry)

    # Remove leading and trailing spaces
    entry = entry.strip()

    return entry

def process_query(query):
    patterns = ['<num>.*\n', '<desc>.*\n', '<.*>']
    query_id_find = re.search('<num>\sNumber:\s(.+?)\n', query)
    if query_id_find:
        query_id = query_id_find.group(1)
    for pattern in patterns: 
        query = re.sub(pattern, ' ', query)

    # Split query to remove stopwords first
    query = query.split(' ')
    query = [q.lower() for q in query]
    stop_words_list = read_stopwords_file('stop_words_english.txt')
    query = remove_stopwords(query, stop_words_list)

    # join the query again for the cleaning process
    query = ' '.join(query)
    query = clean_entry(query)

    # Split query into tokens 
    query = query.split(' ')
    
    
    return query, query_id

def read_stopwords_file(filename):
    '''Read stopwords from an external file'''
    with open('stopwords.txt') as f:
        stopwords_string = f.read()
        stop_words_list = stopwords_string.split('\n')
    return stop_words_list

def remove_stopwords(tokens, stop_words_list):
    '''Remove all stopwords from a list of tokens using a stop_words_list'''
    return [tok for tok in tokens if tok not in stop_words_list]