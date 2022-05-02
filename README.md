All python code to be run can be found in the src folder

The folder consists of 5 Python Files

1. config.py: Consists of configuration information that is global to all scripts
2. data_processing.py: Contains parsing functions for both query and corpus
3. inverted_index.py: Contains functions that create the inverted index from a parsed corpus
4. document_ranking.py: Contains scoring and ranking functions for Boolean, TF, TF_IDF, Relevance Feedback and Custom Algorithm (Discounted Weighted Relevance Feedback)

Running the main.py script will parse data, create an inverted index, get a ranking using the custom algorithm and write the results to a log file in logs/ folder

The log file that currently exists in the log folder contains the results of the Custom Algorithm and can be used with trec eval script as it is.

The file evaluations.txt in the main directory contains the results after running the trec eval script

To run other algorithms, change the method variable on line 33 of main.py to any of the following

1. tf
2. tf-idf
3. relevance-feedback
4. discounted-relevance-feedback
