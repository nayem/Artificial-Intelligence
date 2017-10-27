# Report for question 2 Topic Classification

# Process:
# For the training part, first we use regular expression in read_train_data function to get words that are at least 2-letter long, because alphabet letters does not mean much in topics, then we store following information:
# Doc_untrained: a list of documents that are 'untrained', or we can't see their true topics
# Doc_content: to store all words in every document with their relative frequencies in that document
# Topic_Word_dict: to store frequencies of words under every topic for those 'trained' documents, or documents we can actually see their true labels.

# To make information more significant, we use tf-idf metric to define those 'significant' words inside every document(and in every topic as well, we'll use tf-idf for topics later in writing distinctive word file and in testing), so in Doc_content, only those significant words will be stored. The threshold for tf-idf here is set to 0.0005 for full information on true topics, and 0.05 for partial information on true topics. 0.05 is also set so that the program will not run forever, without suffering too much from performance loss.

# Then we use LDA algorithm in Topic Classification to train the model. The algorithm has 2 stages: 
# Inital assign: randomly assign topics to every word in all untrained documents
# Re assign(Collapsed Gibbs Sampling): update tagging of words using this formula: argmax(topic) P(W|T)*P(T|D), P(W|T) comes from Topic_Word_dict, P(T|D) comes from Doc_Topic_dict, which is initiated randomly in init_assign state of LDA. After getting best guess of topic for a single word, we update Topic_Word_dict and Doc_Topic_dict so that calculating next word will use the latest information. This method is also called 'Collapsed Gibbs Sampling'.

# We finish the training part by outputing distinctive words using 10 words for every topic with highest tf-idf values, and outputing model file by writing in Topic_Word_dict.

# For the testing part, we read in model file, do a further filtering on this dictionary with threshold = 0.0015(we need to do it twice because first time it is filtering over those trained documents, and now it's filtering over trained documents + guessing over untrained using LDA). When predicting topics for documents, we predict the most probably topic for every word, then predict based on the most frequent topics we've seen in this document as the prediction result. Further detail can be seen in predict_topic function.

# Results:

# FRACTION = 1 train: 20 secs test: 18 secs
# Accuracy:0.682687201275
# Confusion matrix:
# [[ 198.    0.    0.    0.    2.    1.    1.    0.    6.    1.    0.   12.
#      1.    3.    0.    2.   85.    5.    2.    0.]
#  [   4.  341.    0.    1.    2.   11.    8.    0.    2.    3.    0.    1.
#      1.    2.    7.    7.    1.    3.    2.    0.]
#  [   0.    2.  326.    0.    0.    1.   11.    1.    0.   45.    2.    0.
#      0.    1.    1.    1.    1.    1.    3.    1.]
#  [  12.    1.    3.  133.    0.    3.    1.    1.    1.    3.    0.    7.
#      5.    0.    0.    6.  221.    1.    0.    0.]
#  [   1.    1.    0.    0.  333.    9.    7.    4.   19.    2.    5.    3.
#      0.    0.    0.    6.    1.    1.    2.    2.]
#  [   7.   11.    2.    0.   13.  221.   65.   25.    2.    5.   10.    6.
#      0.    2.    5.    1.    2.    5.    8.    3.]
#  [   1.   15.    0.    0.    0.    8.  331.    2.    2.    0.    9.    7.
#      2.    1.    3.    2.    1.    1.    4.    1.]
#  [   3.    3.    3.    0.    3.    8.   34.  276.    3.    2.    7.    0.
#      1.    5.   11.    1.    0.    4.    8.   17.]
#  [   1.    1.    2.    0.    2.    0.    4.    0.  341.    2.    0.    0.
#      2.    0.    0.    5.    3.    0.    1.    0.]
#  [   0.    0.    2.    0.    0.    0.    2.    0.    2.  390.    0.    0.
#      0.    0.    1.    0.    0.    2.    0.    0.]
#  [   2.    7.    2.    0.    0.   20.   85.   15.    1.    2.  224.    2.
#      2.    0.    7.    0.    2.    3.   10.    1.]
#  [  19.    3.    4.    3.    2.   18.    5.    1.    2.    4.    2.  296.
#      3.    2.    2.   14.    5.    0.   11.    0.]
#  [   0.    1.    1.    0.    1.    1.    1.    0.   14.    7.    2.    0.
#    334.    2.    0.   11.    0.    0.    1.    0.]
#  [   2.   36.    4.    0.    0.    3.    8.    0.    3.    3.    2.    0.
#      0.  332.    0.    1.    0.    1.    3.    0.]
#  [   1.    2.    1.    0.    2.   28.  180.   20.    0.    1.   15.    0.
#      0.    2.  116.    0.    4.    5.   12.    3.]
#  [   5.    1.    1.    0.    1.    0.    1.    0.  117.    8.    1.    6.
#      1.    0.    0.  161.    5.    2.    0.    0.]
#  [  23.    0.    3.    4.    1.    0.    0.    1.   26.    0.    0.    2.
#      2.    0.    0.    6.  178.    4.    1.    0.]
#  [   2.    4.    2.    0.    0.   31.    1.   14.    0.    2.    4.    5.
#      0.    0.    1.    9.    1.  316.    2.    0.]
#  [   2.    2.    7.    1.    5.    0.  147.   83.    0.    2.   31.    0.
#      0.    1.   29.    2.    3.    8.   55.   16.]
#  [   2.    0.    2.    0.    0.    7.   22.   85.    0.    1.    7.    2.
#      0.    3.    3.    1.    1.    9.   10.  240.]]

# FRACTION = .1 train: 262 secs test: 16 secs
# Accuracy:0.191317047265
# Confusion matrix:
# [[  56.   10.   41.    2.    3.   17.   19.   42.   27.   12.    1.    0.
#      1.   26.    5.   19.    0.   23.    4.   11.]
#  [   7.   16.   28.    6.    7.   11.   49.   60.   30.   30.   24.    0.
#      5.   40.   14.   21.    0.   18.   18.   12.]
#  [   4.    3.  148.    1.    8.   25.   24.   25.   12.   51.   10.    0.
#     11.   16.   13.   18.    0.    9.   16.    3.]
#  [  21.   11.   17.    3.    4.   30.   25.   28.   13.   36.   24.    0.
#      2.   51.    8.   50.    0.   29.   25.   21.]
#  [   3.    2.   53.    0.   70.   27.   24.   52.   35.   13.   15.    0.
#      3.   21.    5.   17.    0.   10.   20.   26.]
#  [   3.   10.   35.    0.    8.   29.   48.   56.   21.   40.   23.    0.
#      3.   22.   17.   18.    0.   19.   28.   13.]
#  [   4.    6.   39.    1.    6.   10.  134.   18.   45.   37.   10.    0.
#      4.   14.    6.   14.    0.   14.   16.   12.]
#  [   3.    3.   36.    1.   10.    5.   31.  140.   19.   29.   12.    0.
#      2.    7.    8.    8.    1.   17.   34.   23.]
#  [   4.   14.   35.    1.    6.    7.   13.   31.   87.   22.    6.    0.
#     13.   35.    7.   31.    0.   15.   23.   14.]
#  [   3.    2.   53.    0.    1.    7.   18.   12.   17.  220.    2.    0.
#      1.   23.    4.   17.    0.    7.   10.    2.]
#  [   1.    8.   39.    0.    5.   14.   78.   37.   23.   40.   59.    0.
#      1.   15.   11.   10.    0.    8.   20.   16.]
#  [   5.   18.   33.    1.    8.   23.   20.   72.   27.   50.    8.   13.
#     10.   17.   14.   28.    0.   29.   16.    4.]
#  [   8.   19.   39.    2.    4.   10.    4.   52.   13.   30.   16.    0.
#     75.   16.    6.   39.    0.    8.   20.   15.]
#  [   4.   13.   23.    0.   10.   28.   13.   34.   21.   42.   17.    0.
#      4.   81.    8.   61.    0.   20.   15.    4.]
#  [   2.    5.   42.    0.    0.   18.   64.   77.   50.   20.   12.    0.
#      5.    9.   31.    4.    0.   20.   23.   10.]
#  [  15.    5.   16.    0.    2.    5.   12.   38.   36.   32.    3.    0.
#      4.   20.    8.   76.    0.   13.   17.    8.]
#  [  20.    6.   16.    2.    3.    4.   35.   36.   11.   12.   11.    0.
#      4.   21.    7.   33.    0.   18.    6.    6.]
#  [   4.    7.   31.    0.    2.   32.   50.   78.   27.   27.    5.    1.
#      5.   29.    3.   18.    1.   56.   10.    8.]
#  [   2.    5.   40.    0.    5.    9.   32.   57.   50.   19.    6.    0.
#      2.   14.   27.    9.    0.   20.   71.   26.]
#  [   2.    2.   41.    2.    6.   15.   20.   84.   44.   19.    9.    0.
#      6.   16.    1.   11.    0.   11.   30.   76.]]

# FRACTION = 0.0 train: 298 secs test: 17 secs
# Accuracy:0.0459373340414
# confusion matrix:
# [[   5.    7.    8.    2.    0.    8.   26.    5.   26.   35.    1.    0.
#     56.    2.    3.   84.   15.    9.    4.   23.]
#  [  24.   14.    7.    4.    0.   12.   32.    0.   87.   37.    5.    0.
#     54.    0.    6.   48.   42.   13.    1.   10.]
#  [  30.   20.   28.    7.    0.   19.   13.    1.   33.   45.    3.    0.
#     57.    0.    3.   82.   39.    8.    0.    9.]
#  [   7.   26.   10.    3.    1.   22.   11.    1.   72.   36.    1.    0.
#     49.    0.    6.   89.   25.   13.    2.   24.]
#  [  17.    8.    7.    3.    0.    7.   39.    0.   74.   20.    2.    0.
#     92.    0.    4.   73.   39.    5.    1.    5.]
#  [  11.   17.   12.   12.    0.   21.   11.    1.   50.   44.    2.    0.
#     82.    4.    7.   52.   53.    7.    3.    4.]
#  [  22.   13.    7.    8.    0.   14.   42.    1.   55.   40.    4.    0.
#     23.    1.   12.  103.   15.    9.    4.   17.]
#  [  17.    5.    2.    6.    1.   26.   17.    5.   52.   43.    4.    2.
#     58.    0.    2.   53.   69.   18.    3.    6.]
#  [  13.   10.   29.   17.    0.   18.   38.    0.   49.   24.    5.    0.
#     58.    1.    5.   65.   18.    6.    1.    7.]
#  [  14.   13.   17.    2.    0.   27.   12.    5.   37.   61.    0.    0.
#     41.    0.    3.  118.   34.   14.    0.    1.]
#  [  17.    1.    3.    9.    0.   27.   38.    1.   53.   42.    2.    0.
#     75.    0.    3.   36.   56.    9.    1.   12.]
#  [  18.   23.   14.    8.    4.   24.   13.    4.   31.   58.    2.    0.
#     81.    7.    5.   47.   44.    5.    5.    3.]
#  [  12.   15.   18.    3.    0.   15.   36.    7.   49.   35.    4.    0.
#     53.    1.    2.   66.   41.    6.    3.   10.]
#  [  18.   26.   11.    7.    1.   38.   12.    4.   41.   26.   16.    0.
#     97.    1.    8.   37.   32.   13.    1.    9.]
#  [   8.    7.    0.    2.    0.   12.   49.    1.   95.   31.    1.    0.
#     90.    0.    5.   35.   40.   10.    0.    6.]
#  [  19.   11.   11.   17.    0.   13.   27.   19.   30.   22.    0.    0.
#     54.    0.    2.   32.   42.    4.    0.    7.]
#  [  26.    6.   13.    7.    0.   21.   20.    3.   25.   18.    1.    0.
#     33.    3.    3.   51.   12.    5.    2.    2.]
#  [  21.    6.   13.    3.    1.   16.   17.    6.  110.   56.    4.    0.
#     47.    0.    1.   46.   28.    3.    3.   13.]
#  [   7.   12.    9.    5.    3.   19.   53.    1.   69.   22.    0.    0.
#     53.    3.    3.   59.   46.   10.    5.   15.]
#  [  14.   10.    5.    1.    0.   21.   16.    4.   46.   37.    3.    0.
#     72.    0.    3.   61.   76.   15.    6.    5.]]


import sys
import re
import glob
import errno
import operator
import random
import ast
import time
import math
import numpy as np

# Global variables
FRACTION = 1.0
DISTINCTIVE_WORDS_FILE = "distinctive_words.txt"
WORD_LIMIT = 10
Confusion_Matrix = np.array([[0.0 for x in range(20)] for y in range(20)])

Topic_List = ["atheism", "autos", "baseball", "christian", "crypto", \
          "electronics", "forsale", "graphics", "guns", "hockey", \
          "mac", "medical", "mideast", "motorcycles", "pc", \
          "politics", "religion", "space", "windows", "xwindows"]

Topic_Word_dict = dict() # stores word frequency under different topics for firstly all trained files, then for all training files
Word_Frequency_Dict = dict() # stores word frequency for all documents

# dictionary for tf-idf
Word_in_doc_count = dict()

# dictioanries for LDA
Doc_id = dict() # look up training doc id for corresponding doc path. eg.Doc_id[0] = 'train/atheism/49960'
Doc_content = dict() # stores word frequency given doc id. eg. Doc_content[0][some_word] = (assigned_topic,freq_in_doc)
Doc_untrained = [] # see if given training doc id can see its topic tag or not. eg. Doc_trained[0] = 0 (cannot see its tag)
Doc_Topic_dict = dict() # P(t|d), or distribution of topics for every doc id in training

# Add a new word(key) in a Dictionary
def add_word_in_dictionary(Word_Dictionary, word):
    if Word_Dictionary.has_key(word.lower()):
        Word_Dictionary[word.lower()] += 1
    else:
        Word_Dictionary[word.lower()] = 1

# Read a file
def read_file(File, Word_dict_list):
    for line in File:
        for w in re.findall(r'([A-Za-z]+[\-_\']?[A-Za-z]+)+', str(line)): # including - that appears once
            for Word_dict in Word_dict_list:
                add_word_in_dictionary(Word_dict, w)
                
# Read in training or test data file
def read_train_data(Mode, Dataset_Directory,FRACTION, Doc_untrained, Topic_Word_dict,Doc_content):
    global Word_Frequency_Dict,Topic_List

    document_id = 0 # unique id for all documents
    doc_topic = dict() # stores which topic the doc_id is, for untrained, topic=''
    
    for topic in Topic_List:
        Word_dict = dict() # used for Topic_Word_dict for trained file
        path = Dataset_Directory + Mode + "/" + topic + "/*"
        print path
        files = glob.glob(path)
        for file in files:
            doc_word_freq = dict() # used for Doc_content for both trained and untrained file
            random_selection = random.random()
            
            if random_selection <= FRACTION: # if we can see the label
                doc_topic[document_id] = topic
                try:
                    with open(file, 'r') as f:
                        read_file(f, [doc_word_freq,Word_dict])
                        #read_file(f, [doc_word_freq,Word_dict])
                    f.close()
                except IOError as exc:
                    if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
                        raise  # Propagate other kinds of IOError.
                
            else:
                doc_topic[document_id] = ''
                Doc_untrained += [document_id]
                with open(file, 'r') as f:
                    read_file(f, [doc_word_freq])
                                
            Doc_id[document_id] = file
            Doc_content[document_id] = doc_word_freq

            for w in doc_word_freq.keys():
                if w not in Word_in_doc_count:
                    Word_in_doc_count[w] = 1
                else:
                    Word_in_doc_count[w] += 1

            document_id += 1
            # <=
    
    # use tf-idf to filter out non-dominant words in every document
    doc_count = len(Doc_content.keys())
    if FRACTION == 1:
        Doc_content=tf_idf(Doc_content,Word_in_doc_count,doc_count, 0.0005,remove_freq=0)
    else:
        Doc_content=tf_idf(Doc_content,Word_in_doc_count,doc_count, 0.05,remove_freq=0)
    #Doc_content=tf_idf(Doc_content,Word_in_doc_count,doc_count, 0.05,remove_freq=0) 15% for frac=0.1
    
    # add filtered words into Word_Frequency_Dict
    for doc_id in Doc_content.keys():
        for w,freq in Doc_content[doc_id].items():
            if w not in Word_Frequency_Dict:
                Word_Frequency_Dict[w] = freq
            else:
                Word_Frequency_Dict[w] += freq
    
    # add filtered words into Topic_Word_dict if doc is trained
    for doc_id in Doc_content.keys():
        topic = doc_topic[doc_id]
        if topic != '': # if is trained document
            if topic not in Topic_Word_dict:
                Topic_Word_dict[topic] = dict()
            for w,freq in Doc_content[doc_id].items():
                
                if w not in Topic_Word_dict[topic]:
                    Topic_Word_dict[topic][w] = freq
                else:
                    Topic_Word_dict[topic][w] += freq

    # do tf-idf on Topic_Word_dict again to further filter
    Word_in_topic_count = dict()
    for t in Topic_Word_dict:
        for w in Topic_Word_dict[t]:
            if w not in Word_in_topic_count:
                Word_in_topic_count[w] = 1
            else:
                Word_in_topic_count[w] += 1
    Topic_Word_dict = tf_idf(Topic_Word_dict, Word_in_topic_count, len(Topic_Word_dict.keys()), threshold = 0.0001, remove_freq=0, tfidf_value=False)
    
    return Doc_untrained,Doc_content,Topic_Word_dict,doc_topic
    
def read_test_data(Mode, DATASET_DIRECTORY,Topic_List):
    doc_true_topic = dict()
    doc_content = dict()
    document_id = 0
    for topic in Topic_List:
        path = DATASET_DIRECTORY + Mode + "/" + topic + "/*"
        #print path
        files = glob.glob(path)
        for file in files:
            doc_word_freq = dict() # used for Doc_content for both trained and untrained file
            try:
                with open(file,'r') as f:
                    read_file(f,[doc_word_freq])
                f.close()
            except IOError as exc:
                if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
                    raise  # Propagate other kinds of IOError.
            doc_true_topic[document_id] = topic
            doc_content[document_id] = doc_word_freq
            document_id += 1
    return doc_true_topic,doc_content
    
# LDA functions
def LDA(Doc_untrained, Doc_content,Doc_Topic_dict, Topic_Word_dict):
    init_assign(Doc_untrained,Doc_content,Doc_Topic_dict,Topic_Word_dict)
    count = 1
    continue_assign = True
    # use a list to store documents needs to be reassigned
    reassigned_docs = Doc_untrained
    
    while continue_assign:
        print 'iteration:{},need to reassign {} documents'.format(count,len(reassigned_docs))
        continue_assign,reassigned_docs = re_assign(reassigned_docs,Doc_content,Doc_Topic_dict,Topic_Word_dict)
        count += 1
    
    return Topic_Word_dict

def init_assign(Doc_untrained, Doc_content, Doc_Topic_dict, Topic_Word_dict):
    global Topic_List
    for doc_id in Doc_untrained:
        for word,freq in Doc_content[doc_id].items():
            # randomly assign topic to words in untrained documents
            rnd_topic = Topic_List[random.randrange(len(Topic_List))]
            if rnd_topic not in Topic_Word_dict:
                Topic_Word_dict[rnd_topic] = dict()
            # update Doc_content dictionary
            Doc_content[doc_id][word] = (rnd_topic,freq)
            # update Topic_Word_dict
            if rnd_topic not in Topic_Word_dict:
                Topic_Word_dict[rnd_topic] = dict()
            if word in Topic_Word_dict[rnd_topic]:
                Topic_Word_dict[rnd_topic][word] += freq
            else:
                Topic_Word_dict[rnd_topic][word] = freq
            # update Doc_Topic_dict
            if doc_id not in Doc_Topic_dict:
                Doc_Topic_dict[doc_id] = dict()
            if rnd_topic not in Doc_Topic_dict[doc_id]:
                Doc_Topic_dict[doc_id][rnd_topic] = freq
            else:
                Doc_Topic_dict[doc_id][rnd_topic] += freq

def re_assign(reassigned_docs, Doc_content, Doc_Topic_dict, Topic_Word_dict,):
    reassigned_n = 0 # total reassigned words in all documents
    new_reassigned_docs = []
    for doc_id in reassigned_docs:
        reassigned_words = 0 # total reassigned words in this document
    
        for word,(pre_assigned,pre_freq) in Doc_content[doc_id].items():
            max_prob,new_assigned = 0.0,pre_assigned
            total_topic_freq = sum(Doc_Topic_dict[doc_id].values())
            for topic in Doc_Topic_dict[doc_id]:

                topic_freq = Doc_Topic_dict[doc_id][topic]
                # calculate P(topic|document_id)
                topic_prob = float(topic_freq)/total_topic_freq
                # calculate P(word|topic)
                if word in Topic_Word_dict[topic]:
                    word_freq = Topic_Word_dict[topic][word]

                    total_freq = sum(Topic_Word_dict[topic].values())
                    word_prob = float(word_freq)/total_freq
                    
                    prob = topic_prob * word_prob
                    if prob > max_prob:
                        max_prob = prob
                        new_assigned = topic
            # update dictionaries after reassigning new topic for every word
            if new_assigned != pre_assigned:
                reassigned_n += 1
                reassigned_words += 1
                # assign new topic to this word if new != pre
                Doc_content[doc_id][word] = (new_assigned,pre_freq)
                # update Doc_Topic_dict 
                Doc_Topic_dict[doc_id][new_assigned] += pre_freq
                Doc_Topic_dict[doc_id][pre_assigned] -= pre_freq
                # update Topic_Word_dict
                Topic_Word_dict[new_assigned][word] += pre_freq
                Topic_Word_dict[pre_assigned][word] -= pre_freq
        
        # put document into reassigned_doc list if its word is reassigned
        if reassigned_words > 0:
            new_reassigned_docs += [doc_id]
       
    print 'Total reassigned words: {}'.format(reassigned_n)
    if reassigned_n == 0:
        return False,new_reassigned_docs
    else:
        return True,new_reassigned_docs
# End of LDA functions

# Calculate distinctive words using tf-idf
def tf_idf(Topic_Word_dict, Word_in_topic_count, Topic_Count, threshold = 0, remove_freq=0, tfidf_value=False):
    # tfidf_value is used in write_distinctive_words
    new_Topic_Word_dict = dict()
    for topic in Topic_Word_dict: 
        
        new_dict = dict() # store results after filtering low tf_idfs
        
        # total freq of words in topic/document
        total_tf = float(sum(Topic_Word_dict[topic].values()))
        for word in Topic_Word_dict[topic]:
            freq = Topic_Word_dict[topic][word]
            tf = freq/total_tf
            word_in_topic_count = Word_in_topic_count[word]

            idf = Topic_Count/float(word_in_topic_count)
            tfidf = tf*math.log(idf)
            if tfidf >= threshold:
                if freq > remove_freq: # used when writing distinctive words for topics
                    if not tfidf_value: # just need to store freqs
                        new_dict[word] = Topic_Word_dict[topic][word]
                    else:
                        new_dict[word] = tfidf
                else: # used when filtering words for documents
                    if Word_in_topic_count[word] != 1:
                        if not tfidf_value:
                            new_dict[word] = Topic_Word_dict[topic][word]
                        else:
                            new_dict[word] = tfidf
        new_Topic_Word_dict[topic] = new_dict
    return new_Topic_Word_dict

# Predict word topics for a single document
def predict_topic(words_dict, topic_word_dict, maj_topic):
    tagging = dict() # stores topics for all words in words_list
    total_freqs = dict()
    
    # here just choose the topic reaching maximum P(T|W) for every word
    for w,freq in words_dict.items():
        total_freq = 0 # total frequency of w in all topics
        for t in topic_word_dict:
            if w in topic_word_dict[t]:
                total_freq += topic_word_dict[t][w]
        total_freqs[w] = total_freq
        best_t,best_prob = '',0
        
        for t in topic_word_dict:
            if w in topic_word_dict[t]:
                topic_prob = topic_word_dict[t][w] / float(total_freqs[w])
                if topic_prob > best_prob:
                    best_t = t
        if best_t not in tagging and best_t != '':
            tagging[best_t] = 1
        elif best_t != '':
            tagging[best_t] += 1 # this has highest accuracy 70.5921, with total_words in Topic model 12234
    
    if tagging.keys():
        return max(tagging,key=tagging.get)
    else:
        return maj_topic

# Test the model performance and print accuracy
def test_model(doc_true_topic,doc_content, Topic_Word_dict,confusion_matrix):
    global Topic_List

    maj_topic,tmp = '',0
    for t in Topic_Word_dict:
        if sum(Topic_Word_dict[t].values()) > tmp:
            tmp = sum(Topic_Word_dict[t].values()) 
            maj_topic = t
    for doc_id in doc_true_topic:
        true = doc_true_topic[doc_id]
        predicted = predict_topic(doc_content[doc_id],Topic_Word_dict,maj_topic)
        confusion_matrix[Topic_List.index(true), Topic_List.index(predicted)] += 1.0

    total = float( sum(sum(confusion_matrix)) )
    # for x in range(len(Topic_List)):
    #     print 'topic {} correct predictions: {}'.format(Topic_List[x],confusion_matrix[x,x])
    correct = sum([ confusion_matrix[x, x]for x in range(len(Topic_List))])
    print 'Total correct:{}, out of total:{} predictions'.format(correct,total)
    print 'Total accuracy:{}'.format(float(correct)/total)

# Output highest 10 disticative words of each topics
def write_distinctive_words(FileName, Topic_Word_dict, Word_Frequency_Dict):
    global Topic_List
    Topic_Count = len(Topic_List)
    Word_in_topic_count = dict()
    for w in Word_Frequency_Dict.keys():
        for t in Topic_List:
            if w in Topic_Word_dict[t]:
                if w in Word_in_topic_count:
                    Word_in_topic_count[w] += 1
                else:
                    Word_in_topic_count[w] = 1
    
    Word_Probability_dict=tf_idf(Topic_Word_dict, Word_in_topic_count, Topic_Count, tfidf_value=True)
    
    with open(FileName, 'w') as f:
        for topic in Topic_List: 
            f.write(str(topic) + "-> ")
            counter = 0
            for word in sorted(Word_Probability_dict[topic].items(), key=operator.itemgetter(1) , reverse=True):
                f.write(str(word) + ",")
                counter += 1
                if counter == WORD_LIMIT: break
            f.write("\n")
    f.close()

# I/O of model file
def write_model_file(FileName, Topic_Word_dict):
    with open(FileName, 'w') as f:

        for topic in Topic_List: 
            f.write(str(topic) + "-> " + str(Topic_Word_dict[topic]) + '\n')
    f.close()

# I/O of model file
def read_model_file(FileName):
    Topic_Word_dict = dict()
    with open(FileName, 'r') as f:
        
        for line in f.readlines(): 
            temp_list = line.split('->')
            topic,content = temp_list[0],temp_list[1]
            Topic_Word_dict[topic] = eval(content)
    f.close()
    return Topic_Word_dict


# Main function
if len(sys.argv) < 4 or len(sys.argv) > 5:
    print "Appropiate number of Argument isn't given"
    print 'Example usage: mode, dataset_directory, model_file,fraction'
    sys.exit()
else:
    (MODE, DATASET_DIRECTORY, MODEL_FILE) = sys.argv[1:4]
    if len(sys.argv) == 5:
        FRACTION = float(sys.argv[4])
    else:
        FRACTION = 1


start_time = time.time()
print (MODE, DATASET_DIRECTORY, MODEL_FILE, FRACTION)

if MODE == 'train':
    # read in training data and build basic dictionaries
    Doc_untrained,Doc_content,Topic_Word_dict,Doc_Topic=read_train_data(MODE,DATASET_DIRECTORY, FRACTION, Doc_untrained, Topic_Word_dict, Doc_content)
    with open(MODEL_FILE, 'w') as f:
        f.write('')
    f.close()
    # run LDA and write model file
    print 'untrained documents:{}'.format(len(Doc_untrained))
    print 'Running LDA on training data...'
    
    Topic_Word_dict = LDA(Doc_untrained,Doc_content,Doc_Topic_dict,Topic_Word_dict)
    print 'Writing distinctive words into distinctive_words.txt'
    write_distinctive_words(DISTINCTIVE_WORDS_FILE, Topic_Word_dict, Word_Frequency_Dict)
    write_model_file(MODEL_FILE, Topic_Word_dict)
elif MODE == 'test':
    # read model file, or P(W|T)
    Topic_Word_dict = read_model_file(MODEL_FILE)
    total_words = 0
    for t in Topic_Word_dict:
        total_words += len(Topic_Word_dict[t].keys())
    print 'total_words in model file:',total_words

    # do tf-idf on Topic_Word_dict again to further filter
    Word_in_topic_count = dict()
    for t in Topic_Word_dict:
        for w in Topic_Word_dict[t].keys():
            if Topic_Word_dict[t][w] == 0:
                del Topic_Word_dict[t][w]
            else:
                if w not in Word_in_topic_count:
                    Word_in_topic_count[w] = 1
                else:
                    Word_in_topic_count[w] += 1
    Topic_Word_dict = tf_idf(Topic_Word_dict, Word_in_topic_count, len(Topic_Word_dict.keys()), threshold = 0.0015, remove_freq=0, tfidf_value=False)
    
    total_words = 0
    for t in Topic_Word_dict:
        total_words += len(Topic_Word_dict[t].keys())
    print 'after filtering, total_words in model file:',total_words
    # read test data
    doc_true_topic,doc_content = read_test_data(MODE, DATASET_DIRECTORY, Topic_List)
    # do prediction and print confusion matrix
    test_model(doc_true_topic, doc_content, Topic_Word_dict, Confusion_Matrix)
    print Confusion_Matrix
else:
    print "Invalid mode. Mode should be 'train' or 'test'"
    sys.exit()
print "Executing Time: {}".format(time.time() - start_time)
