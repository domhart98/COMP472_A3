import math
import os
import string
from decimal import Decimal
from pathlib import Path

import pandas as pd

#Take name of input file, generate path, read data from file into training set.
#Return each column in data set as a dataframe.
def read_input_file(fname):
    dirpath = Path(__file__).parent.absolute()
    print(dirpath)
    path = os.path.join(dirpath, fname)
    print(path)

    training_set = pd.read_csv(path, delimiter="\t")
    # Split dataset into instances and classes
    X = training_set.iloc[:,1]
    y = training_set.iloc[:,2]
    id = training_set.iloc[:,0]
    return id, X, y

#Take a text, and count the frequency of each word. Return dictionary where
#each word is a key, and the value is the frequency with which that word appears.
def tokenize_text(text):

   lower_case_text = text.lower()

   no_punctuation_text = lower_case_text.translate(str.maketrans('', '', string.punctuation))

   words = no_punctuation_text.split()

   wordfreq = {}
   for word in words:
       if word not in wordfreq:
           wordfreq[word] = 0
       wordfreq[word] += 1

   return wordfreq


class MultinomialNB():

    def __init__(self, true_vocabulary, false_vocabulary, tweet_count, false_count, true_count, model):
        self.true_vocabulary = true_vocabulary
        self.false_vocabulary = false_vocabulary
        self.tweet_count = tweet_count
        self.false_count = false_count
        self.true_count = true_count
        self.model = model
        return

#Take collection of tweets and collection of q1_labels. Tokenize each tweet and add
#word frequencies to class vocabulary. If the tweet is factual, add word frequencies to true_vocab.
#Otherwise, add to false_vocab.
    def fit(self, X, y):
        self.true_vocabulary = {}
        self.false_vocabulary = {}

        training_data = pd.DataFrame({'text':X, 'q1_label':y})

        for row in training_data.iterrows():
            self.tweet_count += 1
            wordfreq = tokenize_text(row[1][0])

            for word, freq in wordfreq.items():
                if row[1][1] == 'yes':
                    if word in self.true_vocabulary.keys():
                        self.true_vocabulary[word] += freq
                    else:
                        self.true_vocabulary[word] = freq
                else:
                    if word in self.false_vocabulary.keys():
                        self.false_vocabulary[word] += freq
                    else:
                        self.false_vocabulary[word] = freq

            if row[1][1] == 'yes':
                self.true_count += 1
            else:
                self.false_count += 1

        return

#Take a word frequency dictionary. Calculate score for each potential class, using either true_vocab
#or false_vocab. Return the higher score with the corresponding prediction.
    def score(self, wordfreq):


        true_score = math.log10(self.true_count/self.tweet_count)
        false_score = math.log10(self.false_count/self.tweet_count)

        for word, freq in wordfreq.items():
            if self.model == 'NB_BOW':
                if self.true_vocabulary.get(word) != None:
                    true_score = true_score+(math.log10((self.true_vocabulary.get(word)+0.1/(sum(self.true_vocabulary.values()))+len(self.true_vocabulary)))*freq)
                if self.false_vocabulary.get(word) != None:
                    false_score = false_score+(math.log10((self.false_vocabulary.get(word)+0.1/(sum(self.false_vocabulary.values()))+len(self.false_vocabulary)))*freq)
            if self.model == 'NB_FBOW':
                if (self.true_vocabulary.get(word) != None) and (self.true_vocabulary.get(word) > 1):
                    true_score = true_score+(math.log10((self.true_vocabulary[word]+0.1/(sum(self.true_vocabulary.values()))+len(self.true_vocabulary)))*freq)
                if (self.false_vocabulary.get(word) != None) and (self.false_vocabulary.get(word) > 1):
                    false_score = false_score+(math.log10((self.false_vocabulary[word]+0.1/(sum(self.false_vocabulary.values()))+len(self.false_vocabulary)))*freq)


        if true_score >= false_score:
            pred = 'yes'
            true_score = "{:.2E}".format(Decimal(true_score))
            return true_score, pred
        else:
            pred = 'no'
            false_score = "{:.2E}".format(Decimal(false_score))
            return false_score, pred

#Take group of tweets and predict most likely class for each tweet.
    def predict(self,X):
        scores = []
        q1_labels = []

        for text in X:
            score, q1_label = self.score(tokenize_text(text))
            scores.append(score)
            q1_labels.append(q1_label)

        results = pd.DataFrame({'q1_label': q1_labels,
                                'Score': scores})

        return results


#user menu to select model
model_num = 0
while model_num == 0:
    model_num= int(input("Enter number to select learning model: \n1.NB_BOW\n2.NB_FBOW"))
    if model_num == 1 :
        classifier = MultinomialNB(0,0,0,0,0,'NB_BOW')
        out_file = "NB_BOW"
    elif model_num == 2:
        classifier = MultinomialNB(0,0,0,0,0,'NB_FBOW')
        out_file = "NB_FBOW"
    else:
        model_num = int(input("Enter valid number to select learning model: \n1.NB_BOW\n2.NB_FBOW"))

#read training data
z, X, y = read_input_file('covid_training.tsv')

classifier.fit(X, y)

#read test data
z, X, y = read_input_file('covid_test_public.tsv')
id = pd.DataFrame({'ID':z})

#create results dataframe
results = pd.concat([id, classifier.predict(X), y], axis=1)

#calculate performance metrics
correct = []
yes_TP = 0
yes_FP = 0
yes_FN = 0
no_TP = 0
no_FP = 0
no_FN = 0
correct_count = 0
for row in results.iterrows():
    if row[1][1] == row[1][3]:
        correct.append('correct')
        correct_count += 1
        if row[1][1] == 'yes':
            yes_TP += 1
        else:
            no_TP += 1
    else:
        correct.append('wrong')
        if row[1][1] == 'yes':
            yes_FP += 1
        else:
            no_FP += 1
        if row[1][3] == 'yes':
            yes_FN += 1
        else:
            no_FN += 1

yes_precision = yes_TP/(yes_TP+yes_FP)
yes_recall = yes_TP/(yes_TP+yes_FN)
yes_f1 = (2*yes_precision*yes_recall)/(yes_precision+yes_recall)
if (no_TP+no_FP) == 0:
    no_precision = 0
else:
    no_precision = no_TP / (no_TP + no_FP)
if (no_TP+no_FN) == 0:
    no_recall = 0
else:
    no_recall = no_TP/(no_TP+no_FN)
if (no_precision+no_recall) == 0:
    no_f1 = 0
else:
    no_f1 = (2*no_precision*no_recall)/(no_precision+no_recall)

#append correct/wrong dataframe to the results
correct = pd.DataFrame(correct, columns=['correct'])

#Finalize the results dataframe and output to text file
final_results = pd.concat([results, correct], axis=1)
final_results.to_csv('C:\\Users\\domha\\PycharmProjects\\comp472\\assignment3\\'+classifier.model+'_trace.txt', sep = '_', header=False, index=False)

accuracy = correct_count/len(final_results.index)

#Output performance metrics to text file
with open('C:\\Users\\domha\\PycharmProjects\\comp472\\assignment3\\'+classifier.model+'_eval.txt', 'w') as f:
    f.write("%f \n%f %f\n%f %f\n%f %f" % (accuracy, yes_precision, no_precision, yes_recall, no_recall, yes_f1, no_f1))


















