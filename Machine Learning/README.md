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
