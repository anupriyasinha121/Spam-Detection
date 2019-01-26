import numpy as np
import pandas as pd

df = pd.read_csv("spam.csv", encoding='latin-1')


# print(df)



# Getting stopwords

stopwords =[]

def get_the_sw():
    with open('stopwords.txt') as stops:
        for i in stops:
            stopwords.append(i.strip())
            
get_the_sw()
#print(len(stopwords))
#print(stopwords)

df.v2[0:5]


# Get the corpus

mails = []

def get_the_corpus():
    for i in df.v2:
        mail = i.strip().split()
        mail = [word for word in mail if word not in stopwords]
        mails.append(mail)
get_the_corpus()
mails


# # Removing single frequency words

from collections import defaultdict
frequency = defaultdict(int)

def removal_single_freq():
    for mail in mails:
        for word in mail:
            frequency[word]+=1
            
removal_single_freq()
mails = [[word for word in mail if frequency[word]>1] for mail in mails]
mails


import gensim

dictionary = gensim.corpora.Dictionary(mails)   #create dictionary of words in mails

bag_of_words = [dictionary.doc2bow(mail) for mail in mails]    # creating bag of words: tuple of (word's dintionary index with its frequency in particular document)

# print(bag_of_words)

tfidf = gensim.models.TfidfModel(bag_of_words)
# print(tfidf)
records = tfidf[bag_of_words]
# print(records)
# print(type(records))

n_unique_tokens = len(dictionary)
dense_bow = gensim.matutils.corpus2dense(bag_of_words, num_terms = n_unique_tokens).transpose()   # corpus2dense : Convert corpus into a dense numpy 2D array, with documents as columns.
datset = dense_bow


# In[34]:


target = []

for i in df.v1:
    if i=='ham':
        target.append(1)
    elif i == 'spam':
        target.append(0)
        
# len(target)
print(target)



print(len(datset), len(target))


import pickle
filename = 'datset_vec.pkl'
file_vec = open(filename, 'wb')
pickle.dump(datset , file_vec)
file_vec.close()



file_vec = open(filename, 'rb')
dataset = pickle.load(file_vec)
file_vec.close()

datset


from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True)


from sklearn import svm
from sklearn import naive_bayes as nb
from sklearn.metrics import f1_score
accuracies = []
scores = []


# # SVM classifier

for train, test in kf.split(dataset):
    print("1")
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []
    for i in train:
        train_set.append(dataset[i])
        train_labels.append(target[i])
        
    for i in test:
        test_set.append(dataset[i])
        test_labels.append(target[i])
        
    classifier = svm.SVC()
    
# Naive Bayes classifier

    
#     classifier = nb.GaussianNB()
#    classifier = nb.BernoulliNB()
    predicted = classifier.fit(train_set, train_labels).predict(test_set)
    
    score = f1_score(test_labels, predicted, average='weighted')
    scores.append(score)
    incorrect = (test_labels != predicted).sum()
    accuracy = (len(test_set) - incorrect)/len(test_set)*100
    accuracies.append(accuracy)
    
print("Maximum accuracy attained", max(accuracies))    
print("f1score", scores[np.argmax(accuracies)])
print("\n")
