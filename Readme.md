# Maximum-entropy NER tagger

## Overview:
The goal of this project was to implement and train a named-entity recognizer (NER). Most of the feature builder functionality was implemented using spaCy, an industrial-strength, open-source NLP library written in Python/Cython. For classification a maximum entropy (MaxEnt) classifier is used.

## Implementation Details
The dataset for this task is the [2003 CoNLL (Conference on Natural Language Learning)](https://www.aclweb.org/anthology/W03-0419.pdf) corpus, which is primarily composed of Reuters news data. The data files are pre-preprocessed and already contain one token per line, its part-of-speech (POS) tag, a BIO (short for beginning, inside, outside) chunk tag, and the corresponding NER tag.

For this project different combinations of meta and context features were used. Features were extracted using Python Spacy library and NLTK was used to train the classifier.

## Running the Code:
All code is implemented in `Final_submission.py`. Just change `mode` on line 7 to train, dev or test. The program will first build the sentences, create features for the sentences and train the classifier if mode is set to train or tag words and calculate accuracy if mode is set to dev or test. The program will also create two files, one for the output and other for features.

## Features and Results
By default, the model always makes use of the POS and BIO tags that are provided with the CoNLL data sets. In addition, I experimented with the following groups of token features:

- Length
- Token
- is_digit
- is_country
- Shape
- Lemma
- Prefix
- Suffix
- Ortho 
- Lower
- Context Features
- Cluster-ID

### Results with Meta Features on Development set.
1. Token, Title, Is_digit, Lower_Case = F1: 72.6999
2. Token, Title, Is_digit, Lower_Case, Lemma, norm, orth_ = F1: 73.01
3. Token, Title, Is_digit, Lower_Case, Lemma, norm, orth_, pos, bio = F1: 73.29
4. Token, Title, Is_digit, Lower_Case, Lemma, norm, orth_, pos, bio, shape_ , prefix, suffix= F1: 75.57

### Context Features
Context Features For varying window sizes (number of tokens before the current token and number of token after the current token). Meta features of all the context tokens were also used for experiments:
1. Token + All Meta Features + Context (window size = 2) = 82.57
2. Token + All Meta Features + Context (window size = 3) = 79.41
3. Token + All Meta Features + Context (window size = 1) = 81.13

### Gazzetted list
Next I included the Gazzetted list of Countries and added the feature "is_country" which is "True" if it is a country and "False" if it is not. 
1. Token + All Meta Features + Context (window size = 2) + is_country = 84.75

### Word Embeddings
Next I used [google Word2Vec embeddings]("https://code.google.com/archive/p/word2vec/") which are 300 dimension vectors for each word. I used genism library to process these wors2vec tokens. I trained the K means clustering algorithm to predict the clusters ID for each token, total 1000 clusters were used and the cluster id was added as the feature. If the word doesn't belong to Word2Vec embeddings it gets a default -1 cluster id.

1. Token + All Meta Features + cluster = 82.69
2. Token + All Meta Features + context + cluster = 86.37

## Conclusion
Highest F1: of 86.37 was obtained using Word2Vec cluster, meta features and context features.

Model used for predition of Test dataset tags is trained on combined dev and train set.