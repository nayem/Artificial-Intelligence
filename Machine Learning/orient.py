# assignment 5_ takter_knayem
#--------------- Nearest Neighbor --------------------------

# We used K nearest neighbor methods but instead of considering K members we selected the nodes which gives
# the minimum distance among all the nodes.
# Here, for each test images we compared all the pixels of the test image with the same pixel of each
# train image using euclidean distance.
# We calculated all the distances for each train image for the test image and selected the minimum distance.
# We choose that image as the nearest member of the test image and assigned the orientation of that
# image as the class of the test image.
# We created a file named "nearest_output.txt" where we stored all the image_name with their predicted orientation.
# We also created a 4*4 confusion matrix to calculate the accuracy.
#
# For this algorithm we found the Accuracy:  0.670201484624
# Execution Time:  996.853266001
#
# # ------------------------ Ada Boost --------------------------
#
# In ada boost we considered K decision stumps to classify the test images. We considered 1000 random
# pixels instead of all possible (192*192) pixels.
# We randomly generated 2*1000 numbers and used them as a pixel of the train images. We created 4 "1 vs all" ensemblers (0-270).
# In each ensembler we choose K decision stumps based on minimum error among all the pixels set that we considered.
# For two sets of pixels we compare the pixels and and based on the results we considerd them as True or False
# and found for possible combinations.
# Such as for 0 ensembler we found four combinations: True and 0, False and 0, True not 0, False not 0.
# Then we calculated all the errors for each ensembler (For 0: error for 0 and not 0 and choose the lowest
# error among the errors of all possible pixels).
# We then selected the pixels with minimum error as stumps. We selected k stumps for each ensembler using the same procedure.
#
# Then for each ensembler we updated the weights of the training data whose class is rightly assumed by the ensembler.
# We give them less weight than the unclassified one. So in the next iteration the next stump will try to classify
# the unclassified data.
# We also gave weight to each stumps according to their weight. If the error rate is low the weight of the stump is high.
# After calculating all the stumps and their class for 4 ensembler we then calculated the confidence level
# for each classifier for each test image.
# We calculated the confidence according to the weights. For each ensembler suppose 0 we calculated the percentage of
# the weights of 0 class over the non 0 classes. We selected the ensembler with the highest confidence
# scores among all the ensemblers.
# We assigned the test image with the class of the higest confidence scores ensemblers.
# We calculated confusion matrix for checking the accuracy. Adaboost is much slow as it considers 
# all training data and compare them for 1000 times. We uploaded the adaboost_output.txt file for K=5
# k=5
# Accuracy :: 0.509554140127
# For, k=10
# Accuracy:: 0.53821543477
# Executing Time: 865.644000053
# For, k=50
# Accuracy:: 0.478767451756
# Executing Time: 3808.22100019
#
#
############################ Step 3: Neural Net ###########################
# In neural network, we consider all the RGB value as input, X and the orientation as output, Y.
# For ease, we normalize X by dividing it by 255(maximum RGB value), and Y is converted as a 4 valued vector,
# where 0=0 degree, 1=90 degree, 2= 180 degree, 3= 270 degree.
#
# ===== Structure of the NN ==========
# Input Layer = 192 node
# Hidden Layer = 200 node
# Output Layer = 4 node
# So the first layer weight is W1 (192 X 200) and bias B1 (200).
# In the second layer weight is W2 (200 X 4) and bias B2 (4).
# Learning Rate = 0.01
# Error Constant = 0.001
# ======Parameters=======
# TOTAL_ITERATION = 1000
# BATCH_PROBABILITY = 0.005
# BATCH_ITERATION = 100
# INDIVIDUAL_ITERATION = 2000
#
# ===== Train NN ==========
# To Train the huge dataset, we batch the train dataset in size of 200(using probability).
# Then each batch is trained for 500 times, where each batch is first randomely organized to remove the bias of dataset order.
# In the each batch, we use stochastic gradient descent to train each instance for 2000 times.
#
# In the feed-forward pass, in the first layer we calculate z1=(W1.T)*X (with bias).
# Then we use tanh() fucntion as activation fucntion and get, a1.
# In the second layer we calculate z2=(W2.T)*a1 (with bias), and use softmax function to get a probability to predict orientation.
#
# In the backpropagation phase, we calculate dW2 and dB2 the correction on 2nd layer. Also calculate dW1 and dB1 for first layer correction.
# Then undate the weight and bias vector using learning rate in each example training.
#
# ===== Test NN ==========
# In testing, we just using the feed-forward pass using the last weight (W1, W2) and bias( B1, B2).
# Using this we construct a confusion matrix, and here
#
# | Predicted -> 0 , 90 , 180 , 270
# V Actual
# 0             0.1825 , 0.25 , 0.23 , 0.30
# 90            0.28 , 0.1825 , 0.24 , 0.41
# 180           0.21, 0.34 , 0.1604 , 0.34
# 270           0.355 , 0.27 , 0.25 , 0.203
# Accuracy: 68.8405%
############################ Step 4: Best Neural Net ###########################
# In NNet we get best result for,
# TOTAL_ITERATION = 1000
# BATCH_PROBABILITY = 0.005
# BATCH_ITERATION = 100
# INDIVIDUAL_ITERATION = 2000
# Parameter: TOTAL_ITERATION, BATCH_PROBABILITY, BATCH_ITERATION, INDIVIDUAL_ITERATION
# Running Time(1000, 0.005, 100, 2000)=816s
# Accuraccy(100, 0.05, 10, 20)=28.06%
# Running Time(100, 0.05, 10, 20)=6560s
# Accuraccy(500, 0.05, 50, 200)=49.45%
# Running Time(1000, 0.005, 100, 2000)=11560s
# Accuraccy(1000, 0.005, 100, 2000)= 68.8405%

import numpy as np
import time
import sys
import errno
import operator
import math
import random


start_time = time.time()

ALGORITHM_NEAREST = "nearest"
ALGORITHM_ADABOOST = "adaboost"
ALGORITHM_NNET = "nnet"
ALGORITHM_BEST = "best"

OUTPUT_FILE_NAME_NNET = "nnet_output.txt"
MODEL_FILE_NAME = "model_file.txt"

# Variables for NNet
HIDDEN_COUNT = 300
LEARNING_RATE = 0.01
RANDOMIZE_CONSTANT = 0.25
train_orientation = []
model = dict()
TOTAL_ITERATION = 1000
BATCH_PROBABILITY = 0.005
BATCH_ITERATION = 100
INDIVIDUAL_ITERATION = 2000

# Global variables
image_file = []
orientation = []
pixel = []
attribute_len = 1000
confusion_matrix = np.array([[0.0 for x in range(4)] for y in range(4)])

## ********** Adaboost ****************
def adaboost_0(train_pixel, train_orientation, stump_count, test_list, first_sample, second_sample):
    error = 0
    # Initial Weights for each ensembler
    weight = np.array([(1.0 / len(train_pixel)) for x in range(len(train_pixel))])
    weight_90 = np.array([(1.0 / len(train_pixel)) for x in range(len(train_pixel))])
    weight_180 = np.array([(1.0 / len(train_pixel)) for x in range(len(train_pixel))])
    weight_270 = np.array([(1.0 / len(train_pixel)) for x in range(len(train_pixel))])

    hypothesis, hypothesis_90, hypothesis_180, hypothesis_270 = [], [], [], []
    hyp_w, hyp_w_90, hyp_w_180, hyp_w_270 = [], [], [], []
    matrix_zero = {'T0':0, 'TN0':0, 'F0':0, 'FN0':0}

    for k in range(stump_count):
        print "New stump k: " + str(k)
        data_set = {k: dict(matrix_zero) for k in range(len(first_sample))}
        data_set_90 = {k: dict(matrix_zero) for k in range(len(first_sample))}
        data_set_180 = {k: dict(matrix_zero) for k in range(len(first_sample))}
        data_set_270 = {k: dict(matrix_zero) for k in range(len(first_sample))}
        true_error = np.zeros(len(first_sample))
        false_error = np.zeros(len(first_sample))
        true_error_90 = np.zeros(len(first_sample))
        false_error_90 = np.zeros(len(first_sample))
        true_error_180 = np.zeros(len(first_sample))
        false_error_180 = np.zeros(len(first_sample))
        true_error_270 = np.zeros(len(first_sample))
        false_error_270 = np.zeros(len(first_sample))

        for i in range(len(train_pixel)):
            # print i

            for j in range(len(first_sample)):
                # Compare two pixels and calculate the all possible combinations for each ensembler
                if train_pixel[i][first_sample[j]] > train_pixel[i][second_sample[j]]:
                    if train_orientation[i] == 0:
                        data_set[j]['T0'] += weight[i]
                        data_set_90[j]['TN0'] += weight_90[i]
                        data_set_180[j]['TN0'] += weight_180[i]
                        data_set_270[j]['TN0'] += weight_270[i]
                    elif train_orientation[i] == 90:
                        data_set_90[j]['T0'] += weight_90[i]
                        data_set[j]['TN0'] += weight[i]
                        data_set_180[j]['TN0'] += weight_180[i]
                        data_set_270[j]['TN0'] += weight_270[i]
                    elif train_orientation[i] == 180:
                        data_set_180[j]['T0'] += weight_180[i]
                        data_set[j]['TN0'] += weight[i]
                        data_set_90[j]['TN0'] += weight_90[i]
                        data_set_270[j]['TN0'] += weight_270[i]
                    elif train_orientation[i] == 270:
                        data_set_270[j]['T0'] += weight_270[i]
                        data_set[j]['TN0'] += weight[i]
                        data_set_90[j]['TN0'] += weight_90[i]
                        data_set_180[j]['TN0'] += weight_180[i]
                else:
                    if train_orientation[i] == 0:
                        data_set[j]['F0'] += weight[i]
                        data_set_90[j]['FN0'] += weight_90[i]
                        data_set_180[j]['FN0'] += weight_180[i]
                        data_set_270[j]['FN0'] += weight_270[i]
                    elif train_orientation[i] == 90:
                        data_set_90[j]['F0'] += weight_90[i]
                        data_set[j]['FN0'] += weight[i]
                        data_set_180[j]['FN0'] += weight_180[i]
                        data_set_270[j]['FN0'] += weight_270[i]
                    elif train_orientation[i] == 180:
                        data_set_180[j]['F0'] += weight_180[i]
                        data_set[j]['FN0'] += weight[i]
                        data_set_90[j]['FN0'] += weight_90[i]
                        data_set_270[j]['FN0'] += weight_270[i]
                    elif train_orientation[i] == 270:
                        data_set_270[j]['F0'] += weight_270[i]
                        data_set[j]['FN0'] += weight[i]
                        data_set_90[j]['FN0'] += weight_90[i]
                        data_set_180[j]['FN0'] += weight_180[i]
        # print "Finish Train"
        # Create two error matrix one for True Values another for False values for each ensembler
        for j in range(len(first_sample)):
            true_error[j] = float(data_set[j]['TN0'] + data_set[j]['F0']) / len(train_pixel)
            false_error[j] = float(data_set[j]['T0'] + data_set[j]['FN0']) / len(train_pixel)

            true_error_90[j] = float(data_set_90[j]['TN0'] + data_set_90[j]['F0']) / len(train_pixel)
            false_error_90[j] = float(data_set_90[j]['T0'] + data_set_90[j]['FN0']) / len(train_pixel)

            true_error_180[j] = float(data_set_180[j]['TN0'] + data_set_180[j]['F0']) / len(train_pixel)
            false_error_180[j] = float(data_set_180[j]['T0'] + data_set_180[j]['FN0']) / len(train_pixel)

            true_error_270[j] = float(data_set_270[j]['TN0'] + data_set_270[j]['F0']) / len(train_pixel)
            false_error_270[j] = float(data_set_270[j]['T0'] + data_set_270[j]['FN0']) / len(train_pixel)

        # Chose the minimum error among true and false error matrix
        ## --------------------- 0- ensemble -----------------------
        true_min_err = true_error.min()
        false_min_err = false_error.min()
        true_min_index = np.argmin(true_error)
        false_min_index = np.argmin(false_error)
        if true_min_err < false_min_err:
            min_index = true_min_index
            error = true_min_err
            class_orientation = "zero"
            false_class = "non_zero"
        else:
            min_index = false_min_index
            error = false_min_err
            class_orientation = "non_zero"
            false_class = "zero"
        ## ----------------- 90 ---------------------------------
        true_min_err_90 = true_error_90.min()
        false_min_err_90 = false_error_90.min()
        true_min_index_90 = np.argmin(true_error_90)
        false_min_index_90 = np.argmin(false_error_90)
        if true_min_err_90 < false_min_err_90:
            min_index_90 = true_min_index_90
            error_90 = true_min_err_90
            class_orientation_90 = "90"
            false_class_90 = "non_90"
        else:
            min_index_90 = false_min_index_90
            error_90 = false_min_err_90
            class_orientation_90 = "non_90"
            false_class_90 = "90"

        ## ------------------------------ 180 ------------------------
        true_min_err_180 = true_error_180.min()
        false_min_err_180 = false_error_180.min()
        true_min_index_180 = np.argmin(true_error_180)
        false_min_index_180 = np.argmin(false_error_180)
        if true_min_err_180 < false_min_err_180:
            min_index_180 = true_min_index_180
            error_180 = true_min_err_180
            class_orientation_180 = "180"
            false_class_180 = "non_180"
        else:
            min_index_180 = false_min_index_180
            error_180 = false_min_err_180
            class_orientation_180 = "non_180"
            false_class_180 = "180"
            ## ------------------------------ 270 ------------------------
        true_min_err_270 = true_error_270.min()
        false_min_err_270 = false_error_270.min()
        true_min_index_270 = np.argmin(true_error_270)
        false_min_index_270 = np.argmin(false_error_270)
        if true_min_err_270 < false_min_err_270:
            min_index_270 = true_min_index_270
            error_270 = true_min_err_270
            class_orientation_270 = "270"
            false_class_270 = "non_270"
        else:
            min_index_270 = false_min_index_270
            error_270 = false_min_err_270
            class_orientation_270 = "non_270"
            false_class_270 = "270"
        # update the weights who are truly classified (give less weight)
        for i in range(len(train_pixel)):
            if train_pixel[i][first_sample[min_index]] > train_pixel[i][second_sample[min_index]]:
                if train_orientation[i] == 0 and class_orientation == 'zero':
                    weight[i] = weight[i] * (error / (1 - error))
                elif train_orientation[i] != 0 and class_orientation == 'non_zero':
                    weight[i] = weight[i] * (error / (1 - error))

            if train_pixel[i][first_sample[min_index_90]] > train_pixel[i][second_sample[min_index_90]]:
                if train_orientation[i] == 90 and class_orientation_90 == '90':
                    weight_90[i] = weight_90[i] * (error_90 / (1 - error_90))
                elif train_orientation[i] != 90 and class_orientation_90 == 'non_90':
                    weight_90[i] = weight_90[i] * (error_90 / (1 - error_90))

            if train_pixel[i][first_sample[min_index_180]] > train_pixel[i][second_sample[min_index_180]]:
                if train_orientation[i] == 180 and class_orientation == '180':
                    weight_180[i] = weight_180[i] * (error_180 / (1 - error_180))
                elif train_orientation[i] != 180 and class_orientation == 'non_180':
                    weight_180[i] = weight_180[i] * (error_180 / (1 - error_180))

            if train_pixel[i][first_sample[min_index_270]] > train_pixel[i][second_sample[min_index_270]]:
                if train_orientation[i] == 270 and class_orientation == '270':
                    weight_270[i] = weight_270[i] * (error_270 / (1 - error_270))
                elif train_orientation[i] != 270 and class_orientation == 'non_270':
                    weight_270[i] = weight_270[i] * (error_270 / (1 - error_270))
        total_sum = np.sum(weight)
        total_sum_90 = np.sum(weight_90)
        total_sum_180 = np.sum(weight_180)
        total_sum_270 = np.sum(weight_270)
        # Normalized the weights
        for i in range(len(weight)):
            weight[i] = weight[i] / total_sum
            weight_90[i] = weight_90[i] / total_sum_90
            weight_180[i] = weight_180[i] / total_sum_180
            weight_270[i] = weight_270[i] / total_sum_270

        hypothesis.append([first_sample[min_index], second_sample[min_index], error, class_orientation, false_class])
        hypothesis_90.append([first_sample[min_index_90], second_sample[min_index_90], error_90, class_orientation_90, false_class_90])
        hypothesis_180.append([first_sample[min_index_180], second_sample[min_index_180], error_180, class_orientation_180, false_class_180])
        hypothesis_270.append([first_sample[min_index_270], second_sample[min_index_270], error_270, class_orientation_270, false_class_270])

        # update the weights of the hypothesis
        hyp_w.append(math.log((1 - error)) / error)
        hyp_w_90.append(math.log((1 - error_90)) / error_90)
        hyp_w_180.append(math.log((1 - error_180)) / error_180)
        hyp_w_270.append(math.log((1 - error_270)) / error_270)


        del first_sample[min_index]
        del second_sample[min_index]
        del first_sample[min_index_90]
        del second_sample[min_index_90]
        del first_sample[min_index_180]
        del second_sample[min_index_180]
        del first_sample[min_index_270]
        del second_sample[min_index_270]


    # print hypothesis
    # print hyp_w
    # ============== test image 0000================================
    percentage_0 = []
    percentage_90 = []
    percentage_180 = []
    percentage_270 = []
    for data in test_list:
        test_output = []
        test_output_90 = []
        test_output_180 = []
        test_output_270 = []
        for i in range(len(hypothesis)):
            # print "hypthesis 1: "+str(hypothesis[i][0]) + " "+ str(data[hypothesis[i][0]]) + " " +"hypthesis 1: "+str(hypothesis[i][1]) + " "+ str( data[hypothesis[i][1]])
            if data[hypothesis[i][0]] > data[hypothesis[i][1]]:
                test_output.append([hypothesis[i][3], hyp_w[i]])
            elif data[hypothesis[i][0]] <= data[hypothesis[i][1]]:
                test_output.append([hypothesis[i][4], hyp_w[i]])

            if data[hypothesis_90[i][0]] > data[hypothesis_90[i][1]]:
                test_output_90.append([hypothesis_90[i][3], hyp_w_90[i]])
            elif data[hypothesis_90[i][0]] <= data[hypothesis_90[i][1]]:
                test_output_90.append([hypothesis_90[i][4], hyp_w_90[i]])

            if data[hypothesis_180[i][0]] > data[hypothesis_180[i][1]]:
                test_output_180.append([hypothesis_180[i][3], hyp_w_180[i]])
            elif data[hypothesis_180[i][0]] <= data[hypothesis_180[i][1]]:
                test_output_180.append([hypothesis_180[i][4], hyp_w_180[i]])

            if data[hypothesis_270[i][0]] > data[hypothesis_270[i][1]]:
                test_output_270.append([hypothesis_270[i][3], hyp_w_270[i]])
            elif data[hypothesis_270[i][0]] <= data[hypothesis_270[i][1]]:
                test_output_270.append([hypothesis_270[i][4], hyp_w_270[i]])


        zero_count = 0
        total_count = 0
        for val in test_output:
            total_count += val[1]
            if val[0] == "zero":
                zero_count += val[1]
        percentage_0.append((float(zero_count) / total_count) * 100)
        ### =============== 90 -------------------------------
        zero_count = 0
        total_count = 0
        for val in test_output_90:
            total_count += val[1]
            if val[0] == "90":
                zero_count += val[1]
        percentage_90.append((float(zero_count) / total_count) * 100)
        ### =============== 180 -------------------------------
        zero_count = 0
        total_count = 0
        for val in test_output_180:
            total_count += val[1]
            if val[0] == "180":
                zero_count += val[1]
        percentage_180.append((float(zero_count) / total_count) * 100)
        ### =============== 270 -------------------------------
        zero_count = 0
        total_count = 0
        for val in test_output_270:
            total_count += val[1]
            if val[0] == "270":
                zero_count += val[1]
        percentage_270.append((float(zero_count) / total_count) * 100)

    return percentage_0, percentage_90, percentage_180, percentage_270


def adaboost(train_pixel, train_orientation, stump_count, test_file):
    file = open(test_file, 'r');
    f1 = open("adaboost_output.txt", 'w');
    count = 0
    accuracy = 0
    list_pixel = []
    image_name = []
    original_class = []
    first_sample = []
    second_sample = []
    weight = np.array([(1.0 / len(train_pixel)) for x in range(len(train_pixel))])
    for line in file:
        # print count
        data = tuple([w for w in line.split()])
        list_pixel.append(list(map(int, data[2:])))
        image_name.append(data[0])
        original_class.append(data[1])
    for i in range(attribute_len):
        [x, y] = random.sample(range(192), 2)
        first_sample.insert(i, x)
        second_sample.insert(i, y)
    print "Adaboost Started:  please wait...."
    test_op_0, test_op_90, test_op_180, test_op_270 = adaboost_0(train_pixel, train_orientation, stump_count, list_pixel, first_sample, second_sample)
    # test the accuracy of adaboost
    for index in range(len(image_name)):
        maximum = max(test_op_0[index], test_op_90[index], test_op_180[index], test_op_270[index])
        if test_op_0[index] == maximum:
            pred_class = 0
        elif test_op_90[index] == maximum:
            pred_class = 90
        elif test_op_180[index] == maximum:
            pred_class = 180
        elif test_op_270[index] == maximum:
            pred_class = 270
        f1.write(image_name[index] + " " + str(pred_class) + "\n")
        confusion_matrix[(int(original_class[index]) / 90), (pred_class) / 90] += 1
        for i in range(4):
            for j in range(4):
                confusion_matrix[i, j] = confusion_matrix[i, j] / len(image_name)
                if i == j:
                    accuracy += confusion_matrix[i, j]

    print "Accuracy:  ",
    print accuracy

####  Nearest Nighbor **************
def nearest_neighbor(file_name, train_pixel, orient):
    global confusion_matrix
    accuracy = 0
    predicted_orient = []
    distance = 0
    list_pixel = []
    image_name = []
    file = open(file_name, 'r');
    f1 = open("nearest_output.txt", 'w');
    count = 0
    for line in file:
        data = tuple([w for w in line.split()])
        list_pixel = (list(map(int, data[2:])))
        minimum = []
        for i in train_pixel:
            for r in range(len(i)):
                distance += ((i[r]) - (list_pixel[r])) ** 2  # used euclidean distance
            distance = math.sqrt(distance)
            minimum.append(distance)
        minimum_index = minimum.index(min(minimum))  # chose the minimum distance
        predicted_orient.append(orient[minimum_index])
        image_name.append(data[0])
        confusion_matrix[(int(data[1]) / 90), (orient[minimum_index]) / 90] += 1
    for i in range(4):
        for j in range(4):
            confusion_matrix[i, j] = confusion_matrix[i, j] / len(predicted_orient)
            if i == j:
                accuracy += confusion_matrix[i, j]
    for index in range(len(predicted_orient)):
        f1.write(image_name[index] + " " + str(predicted_orient[index]) + "\n")
    print "Accuracy:: ",
    print accuracy


# Read the training file and seperate the image name , orientation, pixel values.
def read_data(fname):
    global pixel
    file = open(fname, 'r');
    for line in file:
        data = tuple([w for w in line.split()])
        image_file.append(data[0])
        orientation.append(int(data[1]))
        pixel.append(list(map(int, data[2:])))


#########################################

class Config:
    nn_input_dim = 192  # input layer 
    nn_output_dim = 4  # output layer 
    epsilon = LEARNING_RATE  
    reg_lambda = 0.001 
    nn_hdim = HIDDEN_COUNT

    np.random.seed(0)
    W1 = []
    b1 = []
    W2 = []
    b2 = []

def calculate_loss(model, X, y):
    num_examples = len(X)  
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Feed-Forward
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)

    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward Pass 
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def build_model(X, y, nn_hdim, num_passes=INDIVIDUAL_ITERATION, print_loss=False):
    num_examples = len(X)

    for i in range(0, num_passes):
        # Forward propagation
        z1 = X.dot(Config.W1) + Config.b1
        a1 = np.tanh(z1)
        z2 = a1.dot(Config.W2) + Config.b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(Config.W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        dW2 += Config.reg_lambda * Config.W2
        dW1 += Config.reg_lambda * Config.W1

        Config.W1 += -Config.epsilon * dW1
        Config.b1 += -Config.epsilon * db1
        Config.W2 += -Config.epsilon * dW2
        Config.b2 += -Config.epsilon * db2

        model ['W1'] = Config.W1
        model ['b1'] = Config.b1
        model ['W2'] = Config.W2
        model ['b2'] = Config.b2

        if print_loss and i % 1000 == 0:
            print("Iteration %i, Loss: %f" % (i, calculate_loss(model, X, y)))

    return model


def write_model(FileName, model):
       
    with open(FileName, 'w') as f:
        for topic in sorted(model.keys(), key=operator.itemgetter(0) , reverse=True):
            f.write(topic + " " + str(int(model[topic].shape[0])) + " " + str(int(model[topic].shape[1])) + "\n")
            flat_list = [item for sublist in model[topic] for item in sublist]
            n = 0
            while n < len(flat_list):
                if n == len(flat_list) - 1:
                    f.write(str(float(flat_list[n])))
                else:
                    f.write(str(float(flat_list[n])) + " ")
                n += 1
            f.write("\n")
    f.close()

def read_model_file(FileName, model):
    temp_list = []
    with open(FileName, 'r') as f:
        for line in f.readlines(): 
            temp_list.append(line.split(' '))
    f.close()
    
    t = 0
    while t < len(temp_list) :
        topic = temp_list[t][0]
        row = int (temp_list[t][1])
        col = int (temp_list[t][2])
        
        t += 1
        model[topic] = []
        values = []
        num = 0
        while num < len(temp_list[t]):
            if num != 0 and num % col == 0:
                model[topic].append(values)
                values = []
            values.append(float(temp_list[t][num]))
            num += 1
            
        model[topic].append(values)
        t += 1    

        model[topic] = np.array(model[topic])
    f.close()


def tester(test_file_name, output_file_name):
    count = 0
    try:
        with open(test_file_name, 'r') as file_test:
            with open(output_file_name, 'w') as file_output:
                for line in file_test:
                    print count
                    data = tuple([w.lower() for w in line.split()])
                    image_name = data[0]
                    original_class = data[1]
                    list_pixel = list(data[2:])
                    X = map(lambda x: float(x), list_pixel)
                    X = np.array([X])
        
                    z1 = X.dot(model["W1"]) + model ['b1']
                    a1 = np.tanh(z1)
                    z2 = a1.dot(model ['W2']) + model ['b2']
                    exp_scores = np.exp(z2)
                    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                    predicted_class = np.argmax(probs)
                    
                    confusion_matrix[(int(data[1]) / 90), predicted_class ] += 1
                    file_output.write(image_name + " " + str(predicted_class*90) +"\n")
#                     file_output.write(image_name + " predicted:" + str(predicted_class * 90) + " - original:" + str(original_class) + "\n")
                    count += 1

                file_output.close()
            file_test.close()
            
    except IOError as exc:
        if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
            raise  # Propagate other kinds of IOError.
    
    print "Confusion Matrix:(freq)"
    for x in range(4):
        for y in range(4):
            confusion_matrix[ x, y] = float(confusion_matrix[ x, y])
            print "{}, ".format(confusion_matrix[ x, y]),
        print "\n"

    print "Confusion Matrix:"
    total = float(sum(sum(confusion_matrix)))
    for x in range(4):
        for y in range(4):
            confusion_matrix[ x, y] = float(confusion_matrix[ x, y]) / total
            print "{}, ".format(confusion_matrix[ x, y]),
        print "\n"
    
    print "Accuracy: {}".format(sum([ confusion_matrix[x, x]for x in range(4)]) * 100)


def randomize_train_dataset(X, Y):
    number_of_train_data = int(X.shape[0])
    random_index_list = np.random.randint(number_of_train_data, size=number_of_train_data)
    for instance in range(number_of_train_data):
        X[instance], X[random_index_list[instance]] = X[random_index_list[instance]], X[instance]
        Y[instance], Y[random_index_list[instance]] = Y[random_index_list[instance]], Y[instance]

def read_data_nnet(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w.lower() for w in line.split()])
        exemplars += [ (data[0], data[1], data[2:]) ]
    return exemplars


# Main function
if len(sys.argv) == 4:
    (train_file, test_file, algorithm) = sys.argv[1:4]
elif len(sys.argv) == 5:
    (train_file, test_file, algorithm, count) = sys.argv[1:5]
else:
    print "Appropiate number of Argument isn't given"
    sys.exit()



# calling algorithms
if  algorithm == ALGORITHM_NEAREST:
    # # train file reading
    read_data(train_file)
    nearest_neighbor(test_file, pixel, orientation)
    
elif  algorithm == ALGORITHM_ADABOOST:
    # # train file reading
    read_data(train_file)
    adaboost(pixel, orientation, int(count), test_file)
    
elif algorithm == ALGORITHM_NNET and len(sys.argv) == 5:
    HIDDEN_COUNT = int(count)
    TOTAL_ITERATION = 500
    BATCH_PROBABILITY = 0.005
    BATCH_ITERATION = 50
    INDIVIDUAL_ITERATION = 200
        
    Config.W1 = (1 * np.random.randn(Config.nn_input_dim, Config.nn_hdim) + 0) / np.sqrt(Config.nn_input_dim)
    Config.b1 = np.zeros((1, Config.nn_hdim))
    Config.W2 = (1 * np.random.randn(Config.nn_hdim, Config.nn_output_dim) + 0) / np.sqrt(Config.nn_hdim) 
    Config.b2 = np.zeros((1, Config.nn_output_dim))
    model ['W1'] = Config.W1
    model ['b1'] = Config.b1
    model ['W2'] = Config.W2
    model ['b2'] = Config.b2
    
    train_data = read_data_nnet(train_file)
     
    for data in train_data:
        image_file.append(data[0])
        orientation.append(int(data[1]) / 90)
        rgb = map(lambda x: float(x), data[2])
        pixel.append(rgb)
    pixel = np.array(pixel)
    
    for iter in range(TOTAL_ITERATION):
        randomize_train_dataset(pixel, orientation)
        rgb_pixel = []
        train_orientation = []
        for t in range(len(orientation)):
            random_index = np.random.rand(1)[0]
            if random_index <= BATCH_PROBABILITY:
                rgb_pixel.append(pixel[t] / 255)
                train_orientation.append(orientation[t])
         
        rgb_pixel = np.array(rgb_pixel)
        for batch_iter in range(BATCH_ITERATION):
            randomize_train_dataset(rgb_pixel, train_orientation)
            for instance in range(len(rgb_pixel)):  
                model = build_model(np.array([rgb_pixel[instance]]), [train_orientation[instance]], INDIVIDUAL_ITERATION, print_loss=True)
                          
    write_model(MODEL_FILE_NAME, model)
    tester(test_file, OUTPUT_FILE_NAME_NNET)
        
elif algorithm == ALGORITHM_BEST:
    
    if len(sys.argv) == 4:
        TOTAL_ITERATION = 1000
        BATCH_PROBABILITY = 0.005
        BATCH_ITERATION = 100
        INDIVIDUAL_ITERATION = 2000
        HIDDEN_COUNT = 300
        
        Config.W1 = (1 * np.random.randn(Config.nn_input_dim, Config.nn_hdim) + 0) / np.sqrt(Config.nn_input_dim)
        Config.b1 = np.zeros((1, Config.nn_hdim))
        Config.W2 = (1 * np.random.randn(Config.nn_hdim, Config.nn_output_dim) + 0) / np.sqrt(Config.nn_hdim) 
        Config.b2 = np.zeros((1, Config.nn_output_dim))
        model ['W1'] = Config.W1
        model ['b1'] = Config.b1
        model ['W2'] = Config.W2
        model ['b2'] = Config.b2
        
        train_data = read_data_nnet(train_file)
         
        for data in train_data:
            image_file.append(data[0])
            orientation.append(int(data[1]) / 90)
            rgb = map(lambda x: float(x), data[2])
            pixel.append(rgb)
        pixel = np.array(pixel)
        
        for iter in range(TOTAL_ITERATION):
            randomize_train_dataset(pixel, orientation)
            rgb_pixel = []
            train_orientation = []
            for t in range(len(orientation)):
                random_index = np.random.rand(1)[0]
                if random_index <= BATCH_PROBABILITY:
                    rgb_pixel.append(pixel[t] / 255)
                    train_orientation.append(orientation[t])
             
            rgb_pixel = np.array(rgb_pixel)
            for batch_iter in range(BATCH_ITERATION):
                randomize_train_dataset(rgb_pixel, train_orientation)
                for instance in range(len(rgb_pixel)):  
                    model = build_model(np.array([rgb_pixel[instance]]), [train_orientation[instance]], INDIVIDUAL_ITERATION, print_loss=True)
        
        write_model(MODEL_FILE_NAME, model)
        
    elif len(sys.argv) == 5:
        MODEL_FILE_NAME = count
        read_model_file(MODEL_FILE_NAME, model)
    
    tester(test_file, OUTPUT_FILE_NAME_NNET)

print "Executing Time: {}".format(time.time() - start_time)
