import re
from collections import Counter 
import numpy as np 

def collect_document_ids(database_file):
    '''collect a single list of document_ids given the database file'''
    database_file = open(database_file)
    data = database_file.read()
    pattern = re.compile("\.I\s\d*\n")
    ids = re.findall(pattern, data)
    for i in range(len(ids)):
        ids[i] = ids[i].split(' ')[-1][:-1] # This split and slicing extracts the numerical document_id
    return ids


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
    entry = re.sub(r'\s[a-z]\s', ' ', entry)

    # Remove leading and trailing spaces
    entry = entry.strip()

    return entry

def read_stopwords_file(filename):
    '''Read stopwords from an external file'''
    with open('stopwords.txt') as f:
        stopwords_string = f.read()
        stop_words_list = stopwords_string.split('\n')
    return stop_words_list

def remove_stopwords(tokens, stop_words_list):
    '''Remove all stopwords from a list of tokens using a stop_words_list'''
    return [tok for tok in tokens if tok not in stop_words_list]