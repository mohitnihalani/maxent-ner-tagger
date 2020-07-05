import spacy
from spacy.tokens import Doc
import nltk
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from score import score
import pickle
import os


FILE_DIRECTORY = "./CONLL_NAME_CORPUS_FOR_STUDENTS"
dev_file_name = os.path.join(FILE_DIRECTORY,"CONLL_dev.pos-chunk")
train_file_name = os.path.join(FILE_DIRECTORY,"CONLL_train.pos-chunk-name")
test_file_name = os.path.join(FILE_DIRECTORY,"CONLL_test.pos-chunk")

train_feature_file = "./train_features"
test_feature_file = "./test_features"
dev_feature_file = "./dev_features"

dev_generated_tags = "./dev_gen_tags"
test_generated_file = "./test_gen_tags"

cluster_file = "./cluster_file"
countries_file = "./Countries"
model_file_name = "./maxent_model"


"""
Change Mode; ['train','dev','test']
"""
mode = 'train'

modes = {
    'train': {
        'data': train_file_name,
        'features': train_feature_file
    },
    'dev': {
        'data': dev_file_name,
        'features': dev_feature_file,
        'output': dev_generated_tags,
        'key': os.path.join(FILE_DIRECTORY,"CONLL_dev.name"),
    },
    'test': {
        'data': test_file_name,
        'features': test_feature_file,
        'output': test_generated_file,
        'key': ""
    },
    'combine': {
        'features': "./combine_feature_file",
        'output': "./combine_output"
    }
}

nlp = spacy.load('en_core_web_sm')


unique_ner_tags = {'O': 0, 'I-ORG': 1, 'I-MISC': 2, 'I-PER': 3, 'I-LOC': 4, 'B-LOC': 5, 'B-MISC': 6, 'B-ORG': 7}
ner_tags_index = {0: 'O', 1: 'I-ORG', 2: 'I-MISC', 3: 'I-PER', 4: 'I-LOC', 5: 'B-LOC', 6: 'B-MISC', 7: 'B-ORG',-1: ''}
feature_list = ['title','shape','orth','prefix','suffix','norm','is_digit']
word_cluster_map = {}
window_size = 2
countries = set()

"""
Build Features:
1. Length
2. Token
3. is_digit
4. is_country
5. Shape
6. Lemma
7. Prefix
8. Suffix
9. Ortho 
10. Lower
11. Context Features
12. Cluster-ID
"""


def build_features(sentence,feature_set):
    sentence_features = []
    doc = Doc(nlp.vocab,words=sentence)
    nlp.tagger(doc)
    nlp.parser(doc)
    for token in doc:
        feature_set[token.i]['length'] = len(token.text)
        feature_set[token.i]['title'] = token.is_title
        feature_set[token.i]['is_digit'] = token.is_digit
        feature_set[token.i]['norm'] = token.norm_
        feature_set[token.i]['country'] = (feature_set[token.i]['token'] in countries)
        feature_set[token.i]['shape'] = token.shape_
        feature_set[token.i]['orth'] = token.orth_
        feature_set[token.i]['lemma'] = token.lemma_
        feature_set[token.i]['prefix'] = token.prefix_
        feature_set[token.i]['suffix'] = token.suffix_
        feature_set[token.i]['lower'] = token.lower_

        if(feature_set[token.i]['token'] in word_cluster_map):
          feature_set[token.i]['cluster'] = word_cluster_map[feature_set[token.i]['token']]
        else: 
          feature_set[token.i]['cluster'] = -1
    for token in doc:
        for i in range(1,window_size+1):
            for x in feature_list:
                if(token.i - i < 0):
                  if(x == 'cluster' or x == 'brown'): feature_set[token.i]['prev-'+str(i)+x] = -1
                  else: feature_set[token.i]['prev-'+str(i)+x] = '--START--'
                else:
                  feature_set[token.i]['prev-'+str(i)+x] = feature_set[token.i-i][x]

                if(token.i + i > len(feature_set)-1):
                    if(x == 'cluster' or x == 'brown'): feature_set[token.i]['next-'+str(i)+x] = -1
                    else: feature_set[token.i]['next-'+str(i)+x] = '--END--'
                else:
                    feature_set[token.i]['next-'+str(i)+x] = feature_set[token.i+i][x]
    return feature_set

"""
Write Features to File
"""
def write_feature_file(data,file_name):
    print("Writing features to file")
    coloumn_names = data[0].keys()
    data_frame = pd.DataFrame(data,columns=coloumn_names)
    data_frame.to_csv(file_name,index = False)

def build_sentences(file_name):
  train_data = []
  i = 0
  with open(file_name,'r') as file:
    sentence = []
    temp_sentence = []
    for line in file:
      if not line.split():
        if(len(sentence) > 0):
          feat = build_features(sentence=temp_sentence,feature_set=sentence) 
          train_data.extend(feat)
          sentence = []
          temp_sentence = []
          new_line = {}
          for keys in train_data[0].keys():
            if(keys != "token"):
              new_line[keys] = -1
            else:
              new_line[keys] = "NEW-LINE"
          train_data.append(new_line)
      else:
        i += 1
        token, pos, bio, ner = line.strip("\n").split("\t")
        ner = unique_ner_tags[ner]
        tok = {'token': token,'pos':pos,'bio': bio,'ner':ner}
        sentence.append(tok)
        temp_sentence.append(token)

      if(i % 10000 == 0): print("{0}\t word processed".format(i))
  file.close()
  return train_data

def build_dev_train_data_features(file_name):
  print("Buildign Features for DEV/TEST Dataset")
  test_dev_data = []
  i = 0
  with open(file_name,'r') as file:
    sentence = []
    temp_sentence = []
    for line in file:
      if not line.split():
        if(len(sentence) > 0):
          feat = build_features(sentence=temp_sentence,feature_set=sentence)
          
          test_dev_data.extend(feat)
          sentence = []
          temp_sentence = []
          new_line = {}
          for keys in test_dev_data[0].keys():
            if(keys != "token"):
              new_line[keys] = -1
            else:
              new_line[keys] = "NEW-LINE"
          test_dev_data.append(new_line)
      else:
        i += 1
        token, pos, bio = line.strip("\n").split("\t")
        tok = {'token': token,'pos':pos,'bio': bio}
        sentence.append(tok)
        temp_sentence.append(token)

        if(i % 10000 == 0): print("{0}\t word processed".format(i))
  file.close()
  return test_dev_data


"""
Train MaxEnt Tagger
"""
def train_tagger():
    data_frame = pd.read_csv(modes[mode]['features'],keep_default_na=False)

    features = list(data_frame)
    features.remove('ner')

    X_TRAIN = data_frame[features].to_dict("records")
    Y_TRAIN = data_frame['ner'].values

    len(X_TRAIN)
    
    training_data = tuple(zip(X_TRAIN, Y_TRAIN))  
    print("Training Data Samples:  ",len(training_data))
    print("Training Started")
    classifier = nltk.classify.MaxentClassifier.train(training_data, 'IIS', trace=10, max_iter=2)

    print("Saving Classifier")
    save_classifier = open(model_file_name, "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


"""
Tag Words
"""
def tag_words(file_name):
  sentences = build_dev_train_data_features(file_name)

  write_feature_file(sentences,modes[mode]['features'])
  classifier_saved = open(model_file_name, "rb")
  classifier = pickle.load(classifier_saved)
  classifier_saved.close()

  data_frame = pd.read_csv(modes[mode]['features'],keep_default_na=False)
  features = list(data_frame)
  X_TEST = data_frame[features].to_dict("records")
  words = data_frame['token'].values

  OUTPUT_TAGS = []
  for index,feature in enumerate(X_TEST):
    tag = classifier.classify(feature)
    OUTPUT_TAGS.append((words[index],ner_tags_index[tag]))

  with open(modes[mode]['output'],"w") as output:
    for word,tag in OUTPUT_TAGS:
      if(word=="NEW-LINE"): output.write("\n")
      else: output.write("{0}\t{1}\n".format(word, tag))
  output.close()


"""
Load External Cluster and Countries File
"""
def load_cluster_countries__file():
  with open(cluster_file,'r') as file:
    for line in file:
      word,cluster = line.strip("\n").split("\t")
      word_cluster_map[word] = cluster
  file.close()

  with open(countries_file,'r') as file:
    for line in file:
      line = line.rstrip("\n").split()[0]
      countries.add(line)
  file.close()



def main():
    load_cluster_countries__file()
    if(mode == "train"):
        train_data = build_sentences(modes[mode]['data'])
        write_feature_file(train_data,modes[mode]['features'])
        train_tagger()
    if(mode == 'dev' or mode == 'test'):
        tag_words(modes[mode]['data'])
        score(modes[mode]['key'],modes[mode]['output'])

main()