from gensim.models import KeyedVectors
from sklearn.cluster import MiniBatchKMeans
import pickle


filename = './GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
print(mode.wv.shape)
X = model[model.vocab]
kmeans = MiniBatchKMeans(n_clusters=1000,random_state=0,verbose=1).fit(X)
save_classifier = open("./k_means" "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

with open('cluster_file',"w") as output:
    for word,tag in zip(list(model.vocab),kmeans.labels_):
      output.write("{0}\t{1}\n".format(word, tag))
output.close()