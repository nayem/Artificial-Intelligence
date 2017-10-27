# Assignment 4 Part 1
# Spam Classifier
# bxiong-knayem-takter-a4
#
# --------------Naive Bayes Training ----------
# Model file binary: <MODEL_FILE>_bayes_binary.txt
# Model file continuous: <MODEL_FILE>_bayes_continuous.txt
# Execution time: 9.65166401863s
#
#  --------------Naive Bayes Testing ----------
# Confusion matrix::
# SPAM->positive, NOT_SPAM->negative
# Binary : {'TP': 44.67502, 'TN': 1.722788, 'FP': 0.27408, 'FN': 53.328113}
# Binary Accuracy: 98.003133%
# Continuous: {'TP': 44.67502, 'TN': 1.722788, 'FP': 0.27408, 'FN': 53.328113}
# Continuous Accuracy: 98.003133%
# Execution time: 9.64557504654s
#
# --------------Decision Tree Training ----------
# Model file binary: <MODEL_FILE>_dt_model_file_binary.txt
# Model file continuous: <MODEL_FILE>_dt_model_file_cont.txt
# Execution time: 971.850014925s
#
#  --------------Decision Tree Testing ----------
# Confusion matrix::
# SPAM->positive, NOT_SPAM->negative
# Binary : {'FP': 0.39154267815191857, 'TN': 52.975724353954575, 'FN': 0.6264682850430697, 'TP': 46.00626468285043}
# Binary Accuracy: 98.9819890368%
# Continuous: {'FP': 0.15661707126076743, 'TN': 52.50587314017228, 'FN': 1.096319498825372, 'TP': 46.24119028974158}
# Continuous Accuracy: 98.7470634299%
# Execution time: 5.17199993134
#
# ---------- Train Data Set (Naive Bayes+Decision Tree) -------------
# In this assignment we classified spam and non spam emails using Naive bayes and Decision Tree Methods.
# Train Data: In this section we read all the files from SPAM and NON SPAM folder inside Train Folder.
# We parsed each word from each document and calculate their frequency in that documents and stored them in a dictionary for each document
# To improve the performance of our train function we used regular expression and ignored the words which will not add any value in the implementation.
# Like we ignored single character words(e.g. I, a), whole email address(e.g. name@gmail.com), word with more than one consecutive hyphen (e.g. go--home), ending dot(.) and other symbols(e.g. <,>,")
# And consider word with hyphens in words(e.g. a-good-one), apostrophe (e.g. adam's).  
#
# ============ Train Naive Bayes =============
# Initially, we calculate the frequecy of the words in each training files.
# Then we calculate the tf-idf value in two ways, one for binary and another for continuous.
# We consider tf-idf value, since this will give less value for frequent words like- 'the', 'is', 'of', 'at'.
#    Binary: tf(d,w)=1 if word w is in the document d, otherwise 0
#    Continuous: tf(d,w)=0.5+0.5*(frequecy of word w in document d)/(maximum frequecy of a word in d)
#    idf(D,w)=log(Total Number of documents in D)/(Number of document in which w is present)
# tf-idf value is a value of a document that shows how a word represent a document.
# We consider this as an weight for each word of the training document.
# Then we normalize this to generate probability P(T|w_i).
# In the model file, we write the number of training SPAM class files and number of training NOT-SPAM class files; and the all word's probability for each class.
# This probability will be used for testing.
# Also write the top 10 most assosiated word for SPAM and top 10 most assosiated word for NON-SPAM in <DISTINCTIVE_WORDS_FILE>
#
# ============ Decision Tree =============
# In decision Tree method, we created a decision Tree using List. If index 'i' is root we considered index '2*i + 1' as left child of that root and
# index '2*i + 2' as right child.
# ----- Train Decision Tree ---------
# Intially we used the dictionary from the train data set to train this decision Tree. We stored all the unique words into another dictionary.
# We calculated gain for each word from the dictionary in order to find the best attribute. We considered the best_attribute as root
# In order to calculate gain we created a list based on whether a word is present in the document or not. and based on the length of these
# two lists we calculated gain for that word. And find the best_attribute whose gain is the maximum.
# We implemented entropy to calculate gain. After finding the best attribute we updated the train list based on some conditions.
# For Binary feature we updated the list based on whether that best_attribute is in the documents or not.
# We created two branches one is Zero_branch where we stored the documents where the best_attribute is not present,
# and another is a NON_zero_branch where we stored the documents where the best_attribute is present.
# We used ID3 algorithm to implement decision tree.
# after calculating the two updated list we recursively call the ID3 algorithm to create the new branches under the root.
# We implemented the whole decision tree and stored that in a file named 'dt_model_file_binary' and "dt_model_file_cont" respectively for
# Binary and Continuous feature. In this tree the leaf nodes are either "SPAM" or "NON_SPAM"
# We used this model files to test our test files.
# In binary features we created the branches based on whether a word is in the document or not and in continuous features
# we calculated a threshold to decide on the branches of the tree in continuous feature.
# we calculated maximum frequency of a word occured in any of the documents and took a fraction of that number and considered that
# as a threshold. We created two branches if the frequency of a word on any given document is less than the threshold or bigger than the threshold
# we used this condition when creating updated lists while implementing the tree recursively
#
# -------- Test Naive Bayes --------
# First, we read the model file in a dictionary and calculate the prior probability for P(S=SPAM) and P(NON-SPAM) using the number of SPAM and NON-SPAM training file.
# Then for each testing document, tokenize all word of the test document(W) and calculate P(S|W).
#    P(S|W)=P(S,W)P(S)
#    P(S=SPAM|W)=P(S=SPAM,w_1)P(S=SPAM,w_2)... P(S=SPAM,w_n)P(S=SPAM)
#    P(S=NON-SPAM|W)=P(S=NON-SPAM,w_1)P(S=NON-SPAM,w_2)... P(S=NON-SPAM,w_n)P(S=NON-SPAM)
# Then level the test data whose probability is higher.
# And store that in the confusion matrix.
# And found the accuracy 98.003133%(for Binary), 98.003133%(for Continuous).
#
# -------- Test Decision Tree --------
# For testing we first created a list consisting all the words of each documents
# then read the tree from the model file and stored that into a list
# we started checking the tree from the root and based on whether that word(root)  is in the documents or not we split on
# a branch and look for a leaf. And when we find a leaf we return the value whether the document is SPAM or NOT.
# We created confusion matrix for each feature. And found the accuracy 98.7470634299%(for Binary), 98.7470634299%(for Continuous).
# We printed the decision tree for two features up to 4 layers
#
# -------- Best Technique --------
# According to the accuracy, Decision Tree is the best. 
# But Decision Tree takes much longer time to train than Naive Bayes.
#
import sys
import re
import glob
import errno
import operator
import time
import math
import numpy as np
from collections import Counter

start_time = time.time()

# MODE = "train"
# TECHNIQUE = "bayes"
# DATASET_DIRECTORY = ""
# MODEL_FILE = "output_model"

# Global variables
spam_count, nonspam_count, WORD_LIMIT = 0, 0, 10
#nonspam_count = 0
DISTINCTIVE_WORDS_FILE = "distinctive_words.txt"
#WORD_LIMIT = 10
Spam_Word_List_dict=dict()
#Spam_Word_List_dict, NotSpam_Word_List_dict, Attribute_Word_dict = dict(), dict(), dict()
NotSpam_Word_List_dict = dict()
Attribute_Word_dict = dict()
Train_Dataset_list=[]
#Train_Dataset_list, Spam_Train_Dataset_list, NonSpam_Train_Dataset_list, Train_Dataset_list_smaller, word_list, cont_word_list,classifier  = [], [], [], [], [], [],[]
Spam_Train_Dataset_list = []
NonSpam_Train_Dataset_list = []
Topic_Wise_Word_Probability_dict = {"continuous": dict(), "binary": dict()}
Train_Dataset_list_smaller = []
word_list = []
cont_word_list = []
classifier = []
arr = ['-' for n in range(100000)]
cont_arr = ['-' for n in range(100000)]
index, divider = 0, 12
maximum_val = 'True'
#test_list_bin, test_list_cont = [], []
test_list_bin=[]
test_list_cont=[]
confusion_matrix_bin={"TP":0, "FP":0, "FN":0, "TN":0}
#confusion_matrix_bin, confusion_matrix_cont = {"TP":0, "FP":0, "FN":0, "TN":0},{"TP":0, "FP":0, "FN":0, "TN":0}
confusion_matrix_cont = {"TP":0, "FP":0, "FN":0, "TN":0}
#divider=12
Word_Probability_dict= {"continuous": dict(), "binary": dict()}
Number_of_training_file = {"spam": 0, "nonspam": 0}
Number_of_word_occurance = {"spam": dict(), "nonspam": dict()}
NON_NEGATIVE_CONSTANT = 0.05
Probability_SPAM, Probability_NonSPAM = NON_NEGATIVE_CONSTANT, NON_NEGATIVE_CONSTANT
#Probability_NonSPAM = NON_NEGATIVE_CONSTANT
THRESHOLD = 1.0
Confusion_Matrix = {"continuous": np.array([[0.0 for x in range(2)] for y in range(2)]),
                    "binary": np.array([[0.0 for x in range(2)] for y in range(2)])}
NONZERO_FACTOR = .5
maximum_word_counr_dict={}

# Read a file
def read_file(File, Word_List_dict):
    new_dict = dict()
    for line in File:
        data = []
        for w in re.findall(r'([A-Za-z]+[\-_\']?[A-Za-z]+)+', str(line)):
            if Word_List_dict.has_key(w.lower()):
                Word_List_dict[w.lower()] += 1
            else:
                Word_List_dict[w.lower()] = 1

            if new_dict.has_key(w.lower()):
                new_dict[w.lower()] += 1
            else:
                new_dict[w.lower()] = 1

    return new_dict


# Read in training or test data file
def read_data(Mode, Dataset_Directory, Class, Word_List_dict):
    path = Dataset_Directory + Mode + "/" + Class + "/*"
    files = glob.glob(path)

    for file in files:
        try:
            with open(file, 'r') as f:
                Train_Dataset_list_smaller.append(read_file(f, Word_List_dict))
                if Class == "spam":
                    Train_Dataset_list_smaller[-1]["CLASS"] = "spam"
                    if TECHNIQUE == "bayes" and MODE == "train":
                        Number_of_training_file["spam"] += 1
                        for word in Train_Dataset_list_smaller[-1]:
                            if not word == "CLASS":
                                if Number_of_word_occurance["spam"].has_key(word):
                                    Number_of_word_occurance["spam"][word] += 1
                                else:
                                    Number_of_word_occurance["spam"][word] = 1
                    if TECHNIQUE == "dt" and MODE == "train":
                        for word in Train_Dataset_list_smaller[-1]:
                            if not maximum_word_counr_dict.has_key(word):
                                maximum_word_counr_dict[word]=Train_Dataset_list_smaller[-1][word]
                            else:
                                if Train_Dataset_list_smaller[-1][word] > maximum_word_counr_dict[word]:
                                    maximum_word_counr_dict[word]=Train_Dataset_list_smaller[-1][word]
                else:
                    Train_Dataset_list_smaller[-1]["CLASS"] = "nonspam"
                    if TECHNIQUE == "bayes" and MODE == "train":
                        Number_of_training_file["nonspam"] += 1
                        for word in Train_Dataset_list_smaller[-1]:
                            if not word == "CLASS":
                                if Number_of_word_occurance["nonspam"].has_key(word):
                                    Number_of_word_occurance["nonspam"][word] += 1
                                else:
                                    Number_of_word_occurance["nonspam"][word] = 1
                    if TECHNIQUE == "dt" and MODE == "train":
                        for word in Train_Dataset_list_smaller[-1]:

                            if not maximum_word_counr_dict.has_key(word):
                                maximum_word_counr_dict[word]=Train_Dataset_list_smaller[-1][word]
                            else:
                                if Train_Dataset_list_smaller[-1][word] > maximum_word_counr_dict[word]:
                                    maximum_word_counr_dict[word]=Train_Dataset_list_smaller[-1][word]
            f.close()
        except IOError as exc:
            if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
                raise  # Propagate other kinds of IOError.
    return Word_List_dict

def create_train_dataset(Mode, Dataset_Directory, Attribute_Word_dict, Spam_Word_List_dict, NotSpam_Word_List_dict, model_file):
    for key in Spam_Word_List_dict:
        Attribute_Word_dict[key] = 0
    for key in NotSpam_Word_List_dict:
        if not Attribute_Word_dict.has_key(key):
            Attribute_Word_dict[key] = 0
    #print Attribute_Word_dict
    for key in Attribute_Word_dict:
        word_list.append(key)
        cont_word_list.append(key)

    ID_3_binary(Train_Dataset_list_smaller, word_list, arr, 0)
    ID_3_continuous(Train_Dataset_list_smaller, cont_word_list, cont_arr, 0)

    f = open(model_file + '_' +'dt_model_file_binary.txt', 'w')
    for item in arr:
        f.write(str(item))
        f.write(" ")
    f1 = open(model_file + '_' +'dt_model_file_cont.txt', 'w')
    for item in cont_arr:
        f1.write(str(item))
        f1.write(" ")
    f.close()
    f1.close()

def create_train_dataset_table(Mode, Dataset_Directory, Attribute_Word_dict, Spam_Word_List_dict,
                               NotSpam_Word_List_dict):
    for key in Spam_Word_List_dict:
        Attribute_Word_dict[key] = 0
    for key in NotSpam_Word_List_dict:
        if not Attribute_Word_dict.has_key(key):
            Attribute_Word_dict[key] = 0

    for key in Attribute_Word_dict:
        word_list.append(key)

def log_based_2(a, b):
    # print str(a) + " " + str(b)
    if (a == 0 or b == 0):
        return 0
    else:
        return math.log((float(a) / (a + b)), 2)

def spam_classifier(data_list):
    spam = 0
    nonspam = 0
    for item in data_list:
        if item['CLASS'] == "spam":
            spam += 1
        elif item['CLASS'] == "nonspam":
            nonspam += 1
    return (spam, nonspam)

def entropy(a, b):
    ent = 0
    total_val = a + b
    if (a == 0 and b == 0):
        return 0
    else:
        ent = -(float(a) / total_val) * log_based_2(a, b) - (float(b) / total_val) * log_based_2(b, a)
        return ent

def gain_binary(S, w):
    result = 0
    zs, zns = 0, 0
    nzs, nzns = 0, 0
    total_s, total_ns = 0, 0
    for item in S:
        if not item.has_key(w):
            if item["CLASS"] == 'spam':
                zs += 1
            elif item["CLASS"] == 'nonspam':
                zns += 1
        else:
            if item["CLASS"] == 'spam':
                nzs += 1
            elif item["CLASS"] == 'nonspam':
                nzns += 1
    total_s = zs + nzs
    total_ns = zns + nzns
    result = entropy(total_s, total_ns) - float(zs + zns) / (total_ns + total_s) * entropy(zs, zns) - float(
        nzs + nzns) / (total_s + total_ns) * entropy(nzs, nzns)
		
    return result
	
def gain_cont(S, w):
    result = 0
    zs, zns = 0, 0
    nzs, nzns = 0, 0
    total_s, total_ns = 0, 0
    for item in S:
        if not item.has_key(w) or item[w] <= (maximum_word_counr_dict[w]/divider):
            if item["CLASS"] == 'spam':
                zs += 1
            elif item["CLASS"] == 'nonspam':
                zns += 1
        elif item[w] > (maximum_word_counr_dict[w]/divider):
            if item["CLASS"] == 'spam':
                nzs += 1
            elif item["CLASS"] == 'nonspam':
                nzns += 1
    total_s = zs + nzs
    total_ns = zns + nzns
    result = entropy(total_s, total_ns) - float(zs + zns) / (total_ns + total_s) * entropy(zs, zns) - float(
        nzs + nzns) / (total_s + total_ns) * entropy(nzs, nzns)

    return result

def ID_3_continuous(train_list, word_list, tree_arr, index):
    best_gain = -99999
    temp = 0
    best_attribute = ""
    zero_train_datalist = []
    nonzero_train_datalist = []
    spam, non_spam = spam_classifier(train_list)
    if spam == 0:
        tree_arr[index] = "NON_SPAM"
    elif non_spam == 0:
        tree_arr[index] = "SPAM"
    elif len(word_list)==0:
        if spam >= non_spam:
            tree_arr[index] = "SPAM"
        else:
            tree_arr[index] = "NON_SPAM"
    else:
        for word in word_list:
            temp = gain_cont(train_list, word)
            if temp > best_gain:
                best_gain = temp
                best_attribute = word

        #print best_attribute
        tree_arr[index] = best_attribute
        #continuous feature
        for item in train_list:
            if not item.has_key(best_attribute):
                zero_train_datalist.append(item)
            elif item[best_attribute] <= (maximum_word_counr_dict[best_attribute]/divider):
                zero_train_datalist.append(item)
            elif item[best_attribute] > (maximum_word_counr_dict[best_attribute]/divider):
                nonzero_train_datalist.append(item)

        word_list.remove(best_attribute)
        if len(zero_train_datalist) == 0:
            spam, non_spam = spam_classifier(train_list)
            if spam >= non_spam:
                tree_arr[2 * index + 1] = "SPAM"
            else:
                tree_arr[2 * index + 1] = "NON_SPAM"
        else:
            ID_3_continuous(zero_train_datalist, word_list, tree_arr, (2 * index + 1))

        if len(nonzero_train_datalist) == 0:
            spam, non_spam = spam_classifier(train_list)
            if spam >= non_spam:
                tree_arr[2 * index + 2] = "SPAM"
            else:
                tree_arr[2 * index + 2] = "NON_SPAM"
        else:
            ID_3_continuous(nonzero_train_datalist, word_list, tree_arr, (2 * index + 2))
    return tree_arr

def ID_3_binary(train_list, word_list, tree_arr, index):
    best_gain = -99999
    temp = 0
    best_attribute = ""
    zero_train_datalist = []
    nonzero_train_datalist = []
    spam, non_spam = spam_classifier(train_list)

    if spam == 0:
        tree_arr[index] = "NON_SPAM"
    elif non_spam == 0:
        tree_arr[index] = "SPAM"
    elif len(word_list)==0:
        if spam >= non_spam:
            tree_arr[index] = "SPAM"
        else:
            tree_arr[index] = "NON_SPAM"
    else:
        for word in word_list:
            temp = gain_binary(train_list, word)
            if temp > best_gain:
                best_gain = temp
                best_attribute = word

        #print best_attribute
        # Binary feature created
        tree_arr[index] = best_attribute
        for item in train_list:
            if not item.has_key(best_attribute):
                zero_train_datalist.append(item)
            elif item[best_attribute] != 0:
                nonzero_train_datalist.append(item)

        word_list.remove(best_attribute)
        if len(zero_train_datalist) == 0:
            spam, non_spam = spam_classifier(train_list)
            # print " Zero:::: Spam: " + str(spam) + " Non Spam:: " + str(non_spam)
            if spam >= non_spam:
                tree_arr[2 * index + 1] = "SPAM"
            else:
                tree_arr[2 * index + 1] = "NON_SPAM"
        else:
            ID_3_binary(zero_train_datalist, word_list, tree_arr, (2 * index + 1))
        if len(nonzero_train_datalist) == 0:
            spam, non_spam = spam_classifier(train_list)
            if spam >= non_spam:
                tree_arr[2 * index + 2] = "SPAM"
            else:
                tree_arr[2 * index + 2] = "NON_SPAM"
        else:
            ID_3_binary(nonzero_train_datalist, word_list, tree_arr, (2 * index + 2))
    return tree_arr

def ID3_test(train_list, test_list_bin, test_list_cont, index, model_file):
    f = open(model_file + '_' + 'dt_model_file_binary.txt', 'r')
    f1 = open(model_file + '_' + 'dt_model_file_cont.txt', 'r')
    i = 0
    for line in f:
        for word in line.split(" "):
            test_list_bin.insert(i, word)
            i += 1
    for item in train_list:
        category = test_tree(test_list_bin, item, item["CLASS"], 0, confusion_matrix_bin)
    j=0
    for line in f1:
        for word1 in line.split(" "):
            test_list_cont.insert(j, word1)
            j += 1
    print "\n Tree structure: if index i=root, then left_child_index=(2*i + 1) and right_child_index=(2*i +2)"
    print "\n Binary Decision Tree"
    print test_list_bin[0:15]
    print "\n Continuous Decision Tree"
    print test_list_cont[0:15]
    for item in train_list:
        category = test_tree(test_list_cont, item, item["CLASS"], 0,confusion_matrix_cont)

    print "\n Confusion Matrix:: "
    for i in confusion_matrix_bin:
        confusion_matrix_bin[i] = (float(confusion_matrix_bin[i]) / len(train_list)) * 100
    print "Binary Feature :   " + str(confusion_matrix_bin)
    print "Accuracy Binary Feature: " + str(confusion_matrix_bin["TP"]+confusion_matrix_bin["TN"])
    for i in confusion_matrix_cont:
        confusion_matrix_cont[i] = (float(confusion_matrix_cont[i]) / len(train_list)) * 100
    print "Continuos Feature: "+ str(confusion_matrix_cont)
    print "Accuracy Continuous Feature: " + str(confusion_matrix_cont["TP"] + confusion_matrix_cont["TN"])
    return 0


def test_tree(test_list, item, class_s, index, confusion_matrix):
    if test_list[index] == "SPAM":
        if class_s == "spam":
            confusion_matrix["TP"] += 1
        elif class_s == "nonspam":
            confusion_matrix["FN"] += 1
        return "SPAM"
    elif test_list[index] == "NON_SPAM":
        if class_s == "spam":
            confusion_matrix["FP"] += 1
        elif class_s == "nonspam":
            confusion_matrix["TN"] += 1
        return "NON_SPAM"
    elif test_list[index] in item.keys():
        return test_tree(test_list, item, class_s, (2 * index) + 2, confusion_matrix)
    elif test_list[index] not in item.keys():
        return test_tree(test_list, item, class_s, (2 * index) + 1, confusion_matrix)

# tf-idf implementation
def calculate_word_probability(Topics, Word_Freq_list, Word_Probability_dict):
    with open(MODEL_FILE + "_" + TECHNIQUE + "_continuous", 'w') as f:
        f.write(str(Number_of_training_file)[1:-1] + "\n")
    f.close()
    with open(MODEL_FILE + "_" + TECHNIQUE + "_binary", 'w') as f:
        f.write(str(Number_of_training_file)[1:-1] + "\n")
    f.close()

    Word_Probability_dict["continuous"]["spam"] = dict((key, 0) for key in word_list)
    Word_Probability_dict["continuous"]["nonspam"] = dict((key, 0) for key in word_list)

    Word_Probability_dict["binary"]["spam"] = dict((key, 0) for key in word_list)
    Word_Probability_dict["binary"]["nonspam"] = dict((key, 0) for key in word_list)

    for document in Train_Dataset_list_smaller:
        max_freq = max([document[key] for key in document if not key == "CLASS"])
        for word in document:
            if not word == "CLASS":
                tf_cont = 0.5 + 0.5 * float(document[word]) / float(max_freq)
                idf_cont = float(sum(Number_of_training_file.values())) / float(
                    (0 if not Number_of_word_occurance["spam"].has_key(word) else Number_of_word_occurance["spam"][
                        word]) +
                    (
                    0 if not Number_of_word_occurance["nonspam"].has_key(word) else Number_of_word_occurance["nonspam"][
                        word]))
                Word_Probability_dict["continuous"][document["CLASS"]][word] += (tf_cont * idf_cont)

                tf_bin = 0.5 + 0.5
                idf_bin = float(sum(Number_of_training_file.values())) / float(
                    (0 if not Number_of_word_occurance["spam"].has_key(word) else Number_of_word_occurance["spam"][
                        word]) +
                    (
                    0 if not Number_of_word_occurance["nonspam"].has_key(word) else Number_of_word_occurance["nonspam"][
                        word]))
                Word_Probability_dict["binary"][document["CLASS"]][word] += (tf_bin * idf_bin)

    for word in word_list:
        factor = float(
            Word_Probability_dict["continuous"]["spam"][word] + Word_Probability_dict["continuous"]["nonspam"][
                word]) + (2.0 * NONZERO_FACTOR)
        Word_Probability_dict["continuous"]["spam"][word] = float(
            Word_Probability_dict["continuous"]["spam"][word] + NONZERO_FACTOR) / factor
        Word_Probability_dict["continuous"]["nonspam"][word] = float(
            Word_Probability_dict["continuous"]["nonspam"][word] + NONZERO_FACTOR) / factor

        factor = float(
            Word_Probability_dict["binary"]["spam"][word] + Word_Probability_dict["binary"]["nonspam"][word]) + (
                 2.0 * NONZERO_FACTOR)
        Word_Probability_dict["binary"]["spam"][word] = float(
            Word_Probability_dict["binary"]["spam"][word] + NONZERO_FACTOR) / factor
        Word_Probability_dict["binary"]["nonspam"][word] = float(
            Word_Probability_dict["binary"]["nonspam"][word] + NONZERO_FACTOR) / factor

    with open(MODEL_FILE + "_" + TECHNIQUE + "_continuous", 'a') as f:
        f.write("spam-> " + str(Word_Probability_dict["continuous"]["spam"])[1:-1] + "\n")
        f.write("nonspam-> " + str(Word_Probability_dict["continuous"]["nonspam"])[1:-1] + "\n")
    f.close()
    with open(MODEL_FILE + "_" + TECHNIQUE + "_binary", 'a') as f:
        f.write("spam-> " + str(Word_Probability_dict["binary"]["spam"])[1:-1] + "\n")
        f.write("nonspam-> " + str(Word_Probability_dict["binary"]["nonspam"])[1:-1] + "\n")
    f.close()

# Output highest 10 probable word of each topics
def write_distinctive_words(FileName, Topic_list, Topic_Train_Dataset_list, Word_Probability_dict):
    calculate_word_probability(Topic_list, Topic_Train_Dataset_list, Word_Probability_dict)
    with open("continuous_" + FileName, 'w') as f:
        for topic in Topic_list:
            f.write(str(topic) + "-> ")
            counter = 0
            for word in sorted(Word_Probability_dict["continuous"][topic].items(), key=operator.itemgetter(1),
                               reverse=True):
                f.write(str(word) + ",")
                counter += 1
                if counter == WORD_LIMIT: break
            f.write("\n")
    f.close()
    with open("binary_" + FileName, 'w') as f:
        for topic in Topic_list:
            f.write(str(topic) + "-> ")
            counter = 0
            for word in sorted(Word_Probability_dict["binary"][topic].items(), key=operator.itemgetter(1),
                               reverse=True):
                f.write(str(word) + ",")
                counter += 1
                if counter == WORD_LIMIT: break
            f.write("\n")
    f.close()

# Read Bayes probability from file
def read_model_file(FileName, Word_Prob_dict, Number_of_training_file, Model_Type):
    with open(FileName + "_" + TECHNIQUE + "_" + Model_Type, 'r') as f:
        flag_first = True
        for line in f.readlines():

            temp_list = line.split(' ')
            count = 0

            if flag_first:
                while count < len(temp_list):
                    topic = temp_list[count][1:-2]
                    count += 1
                    number = float(temp_list[count][:-1])
                    Number_of_training_file[topic] = number
                    count += 1
                flag_first = False
            else:
                topic = temp_list.pop(0)[:-2]
                temp_dict = dict()
                while count < len(temp_list):
                    word = temp_list[count][1:-2]
                    count += 1
                    prob = float(temp_list[count][:-1])
                    temp_dict[word] = prob
                    count += 1
                Word_Prob_dict[Model_Type][topic] = temp_dict
    f.close()


def bayes_test(Dataset_list, Word_Probability_dict, confusion_matrix):
    for item in Dataset_list:
        predicted_class = naive_predicted_class(item, Word_Probability_dict["continuous"])
        if item["CLASS"] == predicted_class:
            if predicted_class == "spam":
                confusion_matrix["continuous"][0, 0] += 1.0
            else:
                confusion_matrix["continuous"][1, 1] += 1.0
        else:
            if predicted_class == "spam":
                confusion_matrix["continuous"][1, 0] += 1.0
            else:
                confusion_matrix["continuous"][0, 1] += 1.0

        naive_predicted_class(item, Word_Probability_dict["binary"])
        if item["CLASS"] == predicted_class:
            if predicted_class == "spam":
                confusion_matrix["binary"][0, 0] += 1.0
            else:
                confusion_matrix["binary"][1, 1] += 1.0
        else:
            if predicted_class == "spam":
                confusion_matrix["binary"][1, 0] += 1.0
            else:
                confusion_matrix["binary"][0, 1] += 1.0

    confusion_matrix["continuous"] = confusion_matrix["continuous"] / sum(sum(confusion_matrix["continuous"]))
    confusion_matrix["binary"] = confusion_matrix["binary"] / sum(sum(confusion_matrix["binary"]))
    # print test_list
    print "Binary-> TP:"+ str(confusion_matrix["binary"][0,0]*100)+", TN:"+ str(confusion_matrix["binary"][0,1]*100)+", FP:"+ str(confusion_matrix["binary"][1,0]*100)+", FN:"+ str(confusion_matrix["binary"][1,1]*100)
    print "Accuracy Binary: "+str((confusion_matrix["binary"][0,0]+confusion_matrix["binary"][1,1])*100)
    print "Continuous-> TP:"+str(confusion_matrix["continuous"][0,0]*100)+", TN:"+ str(confusion_matrix["continuous"][0,1]*100)+", FP:"+ str(confusion_matrix["continuous"][1,0]*100)+", FN:"+ str(confusion_matrix["continuous"][1,1]*100)
    print "Accuracy Continuous: "+str((confusion_matrix["continuous"][0,0]+confusion_matrix["binary"][1,1])*100)

def naive_predicted_class(word_list, Word_Probability_dict):
    probability_spam = -math.log(float(Number_of_training_file["spam"]) / float(sum(Number_of_training_file.values())))
    probability_nonspam = -math.log(
        float(Number_of_training_file["nonspam"]) / float(sum(Number_of_training_file.values())))

    for word in word_list:
        if not word == "CLASS":
            if not Word_Probability_dict["spam"].has_key(word) and not Word_Probability_dict["nonspam"].has_key(word):
                probability_spam += -math.log(NON_NEGATIVE_CONSTANT)
                probability_nonspam += -math.log(NON_NEGATIVE_CONSTANT)

            elif Word_Probability_dict["spam"].has_key(word) and Word_Probability_dict["nonspam"].has_key(word):
                probability_spam += -math.log(Word_Probability_dict["spam"][word])
                probability_nonspam += -math.log(Word_Probability_dict["nonspam"][word])

            elif Word_Probability_dict["spam"].has_key(word):
                probability_spam += -math.log(Word_Probability_dict["spam"][word])
                probability_nonspam += -math.log(1.0 - Word_Probability_dict["spam"][word])

            elif Word_Probability_dict["nonspam"].has_key(word):
                probability_nonspam += -math.log(Word_Probability_dict["nonspam"][word])
                probability_spam += -math.log(1.0 - Word_Probability_dict["nonspam"][word])

                #     print probability_spam
                #     print probability_nonspam
    if (probability_spam / probability_nonspam) <= (THRESHOLD):
        return "spam"
    else:
        return "nonspam"


# Main function
if len(sys.argv) != 5:
    print "Appropiate number of Argument isn't given"
    sys.exit()
else:
    (MODE, TECHNIQUE, DATASET_DIRECTORY, MODEL_FILE) = sys.argv[1:5]

read_data(MODE, DATASET_DIRECTORY, "spam", Spam_Word_List_dict)
read_data(MODE, DATASET_DIRECTORY, "notspam", NotSpam_Word_List_dict)

if TECHNIQUE == "bayes":
    if MODE == "train":
        create_train_dataset_table(MODE, DATASET_DIRECTORY, Attribute_Word_dict, Spam_Word_List_dict,
                                   NotSpam_Word_List_dict)
        write_distinctive_words(DISTINCTIVE_WORDS_FILE, ['spam', 'nonspam'],
                                [Spam_Word_List_dict, NotSpam_Word_List_dict], Topic_Wise_Word_Probability_dict)
    elif MODE == "test":
        read_model_file(MODEL_FILE, Word_Probability_dict, Number_of_training_file, "continuous")
        read_model_file(MODEL_FILE, Word_Probability_dict, Number_of_training_file, "binary")
        bayes_test(Train_Dataset_list_smaller, Word_Probability_dict, Confusion_Matrix)

if TECHNIQUE == "dt":
    if MODE == "train":
        create_train_dataset(MODE, DATASET_DIRECTORY, Attribute_Word_dict, Spam_Word_List_dict, NotSpam_Word_List_dict, MODEL_FILE)
    elif MODE == "test":
        ID3_test(Train_Dataset_list_smaller, test_list_bin, test_list_cont, 0, MODEL_FILE)

print "Executing Time: {}".format(time.time() - start_time)
