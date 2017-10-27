###################################
# CS B551 Fall 2016, Assignment #3
#
# takter-knayem-bxiong
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Performances:
# ==> So far scored 2000 sentences with 29442 words.
#                    Words correct:     Sentences correct:
#    0. Ground truth:      100.00%              100.00%
#      1. Simplified:       91.73%               37.75%
#             2. HMM:       93.09%               44.25%
#         3. Complex:       94.28%               49.15%
#
# Time required: Approximate 472 seconds
#
# Here we implemented four functions train(), Posterior(), Simplified(), HMM() and complex().
# 1. train(): Here we trained the data found from bc.train. Here we calculated P(S_1), P(S_i), P(S_i+1|S_i) and P(W_i|S_i)
#   In P(S_1): we calculated prior probability. The probability of each pos as first word.
#   P(S_i): we calculated class probability. The probability of each pos at any position.
#   P(S_i+1|S_i): Transition Probability.
#   P(W_i|S_i): Emission probability
#
# 2. Posterior(): In this function we calculated the posterior for the words of a sentence given the label for each word.
#   We used this equation= P(S0) * product(transition_prob_i) * product(emission_prob_i)
#   and take the log of the final calculation
#
# 3. Simplified(): In this function we calculated the most probable tag for S_i for each word W_i
#   Here we multiplied the emission probability with each probability of P(S_i) and returned the maximum probability.
#   we returned the label and the probability
#
# 4. HMM(): In this function, we used viterbi algorithm in order to calculate maximum a posteriori (MAP) labeling of a sentence.
#   here assume i is the set of 12 parts of speech
#   In this case we first considered the first word. For the first word we multiplied the emission probabilty with the prior probability.
#   We calculated all possible probability for all 12 parts of speech. If the first word is not available in the training data then we
#   assumed the emission probability of that word given that parts of speech is very minimum. And thus saved this probability values into
#   a dictionary inorder to use these values for next word
#   For the words other than first word, we calculated the V_i(word) by multiplying the probability values of previous word (V_i-1) and transition probability
#   for all 12 combinations of probability of previous word. Then we take the maximum value from them and multiply that with the emission probability of that word
#   given the parts of speech.
#   We stored all the 12 possible values for each word into a dictionary and then use that dictionary inorder to calculate the probability
#   of next word.
#   After calculating the probability V_i(word) of all words in a sentence we take the maximum probability value of the last word. Then we backtracked
#   in order to calculate the maximum likely path from that label of the word. While calculating each V_i for each word we stored the maximum probability of the previous
#   word for which the value of current probability has become maximum into a dictionary. So while backtracking from the last word we looked into the dictionary
#   for the maximum value of the previous word and take that label and value as most likely label and probability for the previous word. So when the backtrack is finished
#   we get the most likely label for each sentence.
#
# 5. complex(): In this function we used variable elimination algorithm to choose maximum marginal for each word
#   Since we build the algorithm based on 1(c), where every word depends not only on its previous one, but also on the one
#   before the previous one, we need the variable elimination starting from the 3rd word, so we build a helper function, called
#   getFirstTwoWordsState(), to get tags of first two words, or if the sentence only has 1 word, we write in the body of complex()
#   to specifically deal with that, using a simple emission probability(dic_observed) and prior probability(b1, or 1-gram
#   probability dictionary).
#   The pattern we found generating tau function in variable elimination is as follows:
#   P(s_i|W) = P(w_i|s_i) * sum_over_(s_i-1){ P(w_i-1|s_i-1) * sum_over_(s_i-2) {( P(w_i-2|s_i-2)*P(s_i,s_i-1,s_i-2) )} }
#   tau(s_i,s_i-1) = sum_over_(s_i-2) {( P(w_i-2|s_i-2)*P(s_i,s_i-1,s_i-2) )}
#   The way we deal with new words that only appear in the test set is setting tag of that new word as the majority tag in
#   the training set, i.e. noun, with emission probability = 1
#   Interestingly, use of b0 and b1 in helper function getFirstTwoWordsState() makes a difference in complex(). Intuitively if we use b0
#   to calculate most possible tagging of first word in the sentence, the result will be better than using b1 since b1 ignores the specific
#   location of words. However when we run the program, b1 actually does slightly better than b0, especially for sentence accuracy, using
#   b0 generates 49.15%, while using b1 generates 49.45%. Anyway, in order to be consistent with our other algorithms(since HMM also uses the helper function)
#   ,and also the performance are very close,we still uses b0

####

import random
import math
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

###### GLOBAL VARIABLES #####
'''b1,b2,b3: dictionaries counting frequencies of tags of 1-gram,2-gram and 3-gram
dic_observed: frequency of word w given tag t
dic_ems: initiation for dic_observed when new word comes in during training'''
b0, b1, b2, b3 = {}, {}, {}, {}
dic_observed = {}
dic_ems = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0, 'num': 0, 'pron': 0, 'prt': 0, 'verb': 0,
           'x': 0, '.': 0}
mj_tag = ''
dic_trans = {'adj': 0, 'adv': 1, 'adp': 2, 'conj': 3, 'det': 4, 'noun': 5, 'num': 6, 'pron': 7, 'prt': 8, 'verb': 9,
             'x': 10, '.': 11}
trans_prob = np.zeros((12, 12))
minimum = 9999999


class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        global dic_observed
        p = 0
        for i in range(len(sentence)):
            word = sentence[i]
            tag = label[i]
            if word in dic_observed:
                if dic_observed[word][tag] != 0:
                    p += math.log(dic_observed[word][tag])
                else:
                    p += math.log(0.00000001)
            else:
                p += math.log(0.00000001)
        return p

    # Do the training!
    #
    def train(self, data):
        ''' data: [(s1_words,s1_tags),(s2_words,s2_tags),...] '''
        global dic_ems, b0, b1, b2, b3, dic_observed, trans_prob, mj_tag

        b1, b2, b3, dic_observed = {}, {}, {}, {}
        b0 = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0, 'num': 0, 'pron': 0, 'prt': 0, 'verb': 0,
              'x': 0, '.': 0}
        b1 = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0, 'num': 0, 'pron': 0, 'prt': 0, 'verb': 0,
              'x': 0, '.': 0}
        total_trans = [0 for x in range(12)]
        n = len(data)
        T = []  # stores actual tags
        T += [data[i][1] for i in range(n)]
        sentences = []  # stores words
        sentences += [data[i][0] for i in range(n)]

        # create dictionary for P(w|s)
        for i, s in enumerate(sentences):  # i is index, s is sentence
            if i % 10000 == 0:
                print str(i) + '/' + str(n) + ' sentences processed so far'
            for word_i in range(0, len(s)):
                word = s[word_i]
                tag = T[i][word_i]

                if word not in dic_observed:
                    dic_observed[word] = dict(dic_ems)
                dic_observed[word][tag] += 1

        seq1 = []
        seq2 = []
        seq3 = []
        count = 0

        # create dictionaries for P(S),P((S_1,S_2)),P((S_1,S_2,S_3))
        for t in T:

            count += 1
            notfirst = 0
            if count % 1000 == 0:
                print str(count) + '/' + str(n) + ' tags processed so far'
            b0[t[0]] = b0[t[0]] + 1

            seq1 = [t[i] for i in range(len(t))]

            for s in seq1:
                b1[s] = 1 if s not in b1 else b1[s] + 1
            seq2 = [str(t[i:(i + 2)]) for i in range(len(t) - 1)]

            for s in seq2:
                b2[s] = 1 if s not in b2 else b2[s] + 1

            seq3 = [str(t[i:(i + 3)]) for i in range(len(t) - 2)]
            for s in seq3:
                b3[s] = 1 if s not in b3 else b3[s] + 1

        '''Calculating emission probability'''
        for key, values in dic_observed.items():
            for k in values:
                if b1[k] == 0:
                    dic_observed[key][k] = 1e-10
                else:
                    if dic_observed[key][k] == 0:
                        dic_observed[key][k] = 1e-10
                    else:
                        dic_observed[key][k] = (float(dic_observed[key][k]) / b1[k])

        mj_tag = max(b1, key=b1.get)

        '''calculating prior probability b0'''
        N_0 = float(sum(b0.values()))
        for k in b0.keys():
            b0[k] /= N_0

        N_1 = float(sum(b1.values()))
        for k in b1.keys():
            b1[k] /= N_1

        N_2 = float(sum(b2.values()))
        for k in b2.keys():
            b2[k] /= N_2

        N_3 = float(sum(b3.values()))
        for k in b3.keys():
            b3[k] /= N_3

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        global b0
        global dic_observed

        prob_simple = {}
        total_simple = 0
        list_pos = []
        list_prob = []

        for word in sentence:
            if word not in dic_observed:
                pos = max(b0, key=b0.get)
                maximum = (pos, b0[pos])
            else:
                total_simple = float(sum(dic_observed[word].values()))
                maximum = (0, 0)
                for p in dic_observed[word].keys():  # p: one of 12 states
                    if p in b0:
                        tmp = (dic_observed[word][p]) * b0[p]
                        if tmp > maximum[1]:
                            maximum = (p, tmp)

            list_pos.append(maximum[0])
            list_prob.append(round(maximum[1], 2))

        return [[list_pos], [list_prob]]

    def hmm(self, sentence):
        ''' viterbi algorithm '''
        global b1, b0, b2, dic_observed, mj_tag

        dic_prev, dic_new, dic_intermediate, dic_maximum, dic_vvprev, dic_vprev, dic = {}, {}, {}, {}, {}, {}, {}
        notfirst, emission_prob, k = 0, 0, 1
        list_pos, list_prob = [], []
        length = len(sentence)

        for word in sentence:
            if notfirst == 0:
                for i in b0:

                    if word not in dic_observed:
                        emission_prob = minimum
                    else:
                        emission_prob = float(dic_observed[word][i])
                    dic_prev[i] = b0[i] * emission_prob
                notfirst += 1
            else:
                for i in b1:
                    dic_vprev = {}
                    dic_new = {}
                    for j in b1:
                        trans_prob = b2[str((j, i))] / b1[j] if str((j, i)) in b2 else 1e-10
                        dic_new[j] = dic_prev[j] * trans_prob
                    if word not in dic_observed:
                        emission_prob = minimum
                    else:
                        emission_prob = float(dic_observed[word][i])
                    maximum_key = max(dic_new, key=dic_new.get)
                    dic_intermediate[i] = dic_new[maximum_key] * emission_prob
                    dic_vprev[maximum_key] = (dic_prev[maximum_key])
                    dic_vvprev[i] = dict(dic_vprev)
                    dic_maximum[k] = dict(dic_vvprev)
                k += 1
                for key in dic_intermediate:
                    dic_prev[key] = dic_intermediate[key]

        if length > 0:
            maximum_key = max(dic_prev, key=dic_prev.get)
            list_pos.append(maximum_key)
            list_prob.append(dic_prev[maximum_key])
            for i in (range(len(sentence) - 1, 0, -1)):
                dic[i] = dic_maximum[i][maximum_key]
                for key in dic[i]:
                    list_pos.append(key)
                    list_prob.append(dic[i][key])
        tag = list_pos[::-1]
        if length >= 2:
            p1, p2 = self.getFirstTwoWordsState(sentence[0:2], dic_observed, b0, b1, b2)
            tag1, tag2 = p1[1], p2[1]
            prob1, prob2 = p1[0], p2[0]

            new_tag = [tag1, tag2] + tag[2:]
            return [new_tag], []
        else:
            return [tag], []

    def complex(self, sentence):
        ''' two dependencies model '''
        global b0, b1, b2, b3, dic_observed, mj_tag

        W_S = dic_observed
        S = b1
        SS = b2
        SSS = b3
        tagging = []  # [(prob1,s1),(prob2,s2)...]

        if len(sentence) < 2:
            w1 = sentence[0]

            if w1 in W_S:
                TAGS_w1 = W_S[w1].keys()
            else:
                TAGS_w1 = [mj_tag]

            # calculate p(word|state) for every state here
            p1 = (0, 0)
            for s1 in TAGS_w1:
                e1 = W_S[w1][s1] if w1 in W_S else 1
                tmp = (e1 * (S[s1]))
                if tmp > p1[0]:
                    p1 = (tmp, s1)
            p1 = (round(p1[0], 2), p1[1])
            tagging.append(p1)
        else:
            s1, s2 = self.getFirstTwoWordsState(sentence[0:2], W_S, b0, S, SS)
            tagging.append(s1)
            tagging.append(s2)

            i = 0
            tau = {}  # tau[(s_i-1,s_i)]: t() in Variable Elimination
            pre_tau = {}

            while i <= len(sentence) - 3:
                w1, w2, w3 = sentence[i:(i + 3)]  # 3-grams
                dic_p = {}  # to store probability of s|w for all values of s3 in SSS.keys()...

                # calculate tau
                for sss in SSS.keys():
                    s1, s2, s3 = eval(sss)

                    e1 = W_S[w1][s1] if w1 in W_S else 1
                    e2 = W_S[w2][s2] if w2 in W_S else 1
                    e3 = W_S[w3][s3] if w3 in W_S else 1

                    if (s2, s3) not in tau:
                        tau[(s2, s3)] = 0

                    if pre_tau == {}:  # if is initiation,then, w1 starts as 3rd word in sentence
                        tau[(s2, s3)] += (e1 * SSS[sss])
                    else:
                        if (s1, s2) in pre_tau:
                            tau[(s2, s3)] += e1 * (SSS[sss] / SS[str((s1, s2))]) * pre_tau[(s1, s2)]

                    if s3 not in dic_p:
                        dic_p[s3] = 0
                    dic_p[s3] += e3 * e2 * tau[(s2, s3)]

                # choose best tag of w3
                if w3 in W_S:
                    best_tag = max(dic_p, key=dic_p.get)
                    best_prob = round(float(dic_p[best_tag]) / sum(dic_p.values()), 2)
                else:
                    best_tag = mj_tag
                    best_prob = 1.0
                tagging.append((best_prob, best_tag))

                i += 1
                pre_tau = tau
                tau = {}
        tags = [tagging[i][1] for i in range(len(tagging))]
        probs = [tagging[i][0] for i in range(len(tagging))]
        return [[tags], [probs]]
        #return [[["noun"] * len(sentence)], [[0] * len(sentence), ]]

    # Helper functions
    def getFirstTwoWordsState(self, words, W_S, b0, S, SS):
        ''' this function helps choose s1 and s2 of complex model, if the sentence
        has length greater than 2(so that complex model exists)
        output: ([prob1,state1],[prob2,state2]) '''
        global mj_tag
        w1, w2 = words[0], words[1]
        if w1 in W_S:
            TAGS_w1 = W_S[w1].keys()
        else:
            TAGS_w1 = [mj_tag]

        if w2 in W_S:
            TAGS_w2 = W_S[w2].keys()
        else:
            TAGS_w2 = [mj_tag]

        # calculate p(word|state) for every state here
        p1, p2 = (0, 0), (0, 0)

        for s1 in TAGS_w1:
            # e1 = W_S[w1][s1] / float(N_w1) if w1 in W_S else 1
            e1 = W_S[w1][s1] if w1 in W_S else 1
            tmp = (e1 * (b0[s1]))
            if tmp > p1[0]:
                p1 = (tmp, s1)
        p1 = (round(p1[0], 2), p1[1])

        for s2 in TAGS_w2:
            for s1 in TAGS_w1:
                e1 = W_S[w1][s1] if w1 in W_S else 1
                if str((s1, s2)) in SS:
                    p = e1 * SS[str((s1, s2))]
                    e2 = W_S[w2][s2] if w2 in W_S else 1
                    tmp = (e2 * p)
                    if tmp > p2[0]:
                        p2 = (tmp, s2)
        p2 = (round(p2[0], 2), p2[1])
        return p1, p2

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"
