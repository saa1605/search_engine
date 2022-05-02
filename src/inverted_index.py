from collections import Counter
from src.data_processing import clean_entry, read_stopwords_file, remove_stopwords
import numpy as np
import json 



def create_inverted_index(corpus, doc_ids):
    '''Created inverted index for all unique terms with (document_ids, term_frequncies) in posting list'''
    # Create empty inverted index 
    inverted_index = {}
    document_lengths = np.zeros(len(corpus))

    # Read stopwords from a stopword file 
    stopwords_list = read_stopwords_file('stop_words_english.txt')

    # Iterate over the entire courpus of list of documents 
    for i, document in enumerate(corpus):
        doc_id = i


        document = document.split(" ")

        # Remove stopwords from document 
        document = remove_stopwords(document, stopwords_list)
        document = [token.lower() for token in document]
        
        document = ' '.join(document)
    
        # Clean document 
        document = clean_entry(document)

        # Tokenize document 
        tokens = document.split(" ")

        # Calculate and store document lengths 
        document_lengths[doc_id] = len(tokens)

        # Create a dictionary which stores term frequencies 
        term_frequencies = Counter(tokens)

        # Euclidian normalize the term frequencies 
        # denom = np.sum(np.array([(count)**2 for count in term_frequencies.values()]))
        # denom = np.sqrt(denom)            
        # Iterate over all unique terms 
        for term in term_frequencies.keys():
            # term_frequencies[term] /= deno
            # If the term already exists in inverted index, append the current (doc_id, term_frequency) to the postings list                
            if term in inverted_index:
                inverted_index[term].append((doc_id, term_frequencies[term]))
            # If the term does not exist in inverted index, create a entry with the term as key nd add (doc_id, term_frequency) to the posting list
            else:
                inverted_index[term] = [(doc_id, term_frequencies[term])]
    return inverted_index, document_lengths

def get_document_norm(corpus):
    norms = np.zeros(len(corpus))
    for i, document in enumerate(corpus): 
        term_frequencies = Counter(document)
        norm = np.sum(np.array([(count)**2 for count in term_frequencies.values()]))
        norm = np.sqrt(norm) 
        norms[i] = norm 
    return norms  

def calculate_idf(inverted_index, num_documents):
    '''Calculate the inverse document frequency for all unique terms in the vocabulary'''
    # Create empty idf dictionary 
    idf = {}

    # Iterate over all terms in inverted index 
    for item in inverted_index.keys():
        # IDF_t = log( N / number_of_documents_t_appears_in)
        idf[item] = np.log(num_documents/(len(inverted_index[item])))
    
    return idf 

def save_inverted_index(inverted_index):
    '''Save the inverted index to a json file'''
    with open('inverted_index.json', 'w') as j:
        json.dump(inverted_index, j)