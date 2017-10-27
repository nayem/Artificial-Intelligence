#### Assignment 3: Problem 2
#### Red line -> simple model
#### Blue line -> HMM model
#### Green line -> A point given in HMM model
# 
#### Use 3000 Gibbes sampling, Time apporx. 10 min to run
# 
# =========================================================================================================================
# (1) a description of how you formulated the problem, including precisely defining the abstractions (e.g. HMM formulation)
# =========================================================================================================================
# Problem 1: This problem is modeled on 1b simplified model. In this problem, edge_strength matrix's each row values of a column is consided as evidence(w)
#  and each column refers to a hidden state(S). Here, no S is depended on any other hidden states S. S refers to the row number of the mountain
#  ridgeline in each column. Higher the gradient value, the greater chance for that row to be on the ridgeline.
#  
# Problem 2 & 3: This problem is modeled on 1a HMM model. The differnce between the previous model is that here  S is depended on previous hidden states S. 
# So in the time of evaluating ridgeline, a ridge-point of a column also influences the selection of other ridge-point.
# 
# ==================================================
# (2) a brief description of how your program works
# ==================================================
# For the Problem 1, simply I calculate the max gradient value of each column in edge_strength matrix. This means I assign Hidden state(S) that
# value for which evidence gradient value is higher.
# 
# For the Problem 2, first I normalize each column of the edge_strength matrix and this is now the Emission Probabilty matrix.
# This matrix has the property that higher gradient intutively has higher probabilty. 
# 
# Now, for MCMC gibbs sampling, I make the initial sampling with the row numbers which has the maximum gradient value in each column.
# Then for each column(unwatched variables), I calculated the probabilty P(Si|s_1...s_i-1,s_i+1...s_n, w_1...w_n). For this, I calcuate all the 
# possible values S_i can take(index of each row). Then from the probabilty distribution of S_i, I choose a new s_i for the new sample. After
# calculating for each unwatched variables, I get the new samples and add this in the total sample list.
# 
# In calculating P(Si|s_1...s_i-1,s_i+1...s_n, w_1...w_n), I use Gausian Distribution for assinging Transitional probabily.
# This means that P(s_i+1|s_i) is higher if s_i+1=s_i and similarly P(s_i|s_i-1) is higher if s_i=s_i-1. Here s_i+1 refers to the row value of
# next column ridgeline. And s_i-1 refers to the row value of previous column ridgeline.
# 
# After sampling of 3000 times, I calcuate P(Si) for each column and the row value gives higher probabilty, I choose that one as a ridge-point.
# To calculate this, I evaluate P(Si) I have to select which row index has the higher frequency of occurance in all the samples.
# 
# Problem 3 is mostly like the problem 2,just now I have a point (x,y) on the ridgeline. To maintain uniformity, I use the same function of
# problem 2 just with parameter (gt_row, gt_col) with default value -1. Negative value means no (x,y) point is given, otherwise a point is given. 
# Now each sample the (x,y) point will be present. To do this this point will be used in calculating P(Si|s_1...s_i-1,s_i+1...s_n, w_1...w_n).
# In choosing a row value y column, x will always picked. And in the time of choosing a row value y-1 and y+1 column, Transitional Probabilty
# P(s_i|s_i-1=x) and P(s_i+1=x|s_i) has the near 1 probability. So the chance of choosing (x,y) point in ridgeline is almost 1. To elevate this
#  Transitional Probabilty, I use Gausian Distribution with narrower standard deviation. Besides this, rest this same as Problem 2.
#  
# =======================================================================================
# (3) a discussion of any problems, assumptions, simplifications, and/or design decisions
# =======================================================================================
# Problem 1 part is faster and gives good result when mountain is darker and sky is ligher. For ligher mountain or white mountain or if other much 
# darker object is present in the picture, then the ridgeline is not very good.
#   
# Problem 2 and 3 are very slow for sampling reason and takes around 10minutes to sample. 
# Higher the sampling rate and better the gaussian distribution model, the better the reason.
# This gives considerable good result than Problem 1, but lighter mountain peak is also a problem here.


#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import sys
import random
import collections
from scipy.stats import norm

# Global Variable
NON_ZERO_CONSTANT = 1E-10
TRAINING_TIME = 3000

# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    return filtered_y ** 2

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range(max(y - thickness / 2, 0), min(y + thickness / 2, image.size[1] - 1)):
            image.putpixel((x, t), color)
    return image

# Draw line in the image
def simple_draw_edge(image, gradient_image, color, thickness):
    # Gradient_image = [row, col]
    # Image = [col,row]
    row_len, col_len = gradient_image.shape[0], gradient_image.shape[1]
    
    for pix in range(col_len):
        max_gradient = max(gradient_image[:, pix])
        max_index = where(gradient_image[:, pix] == max_gradient)[0][0]
        for t in range(max(max_index - thickness / 2, 0), min(max_index + thickness / 2, image.size[1] - 1)):
            image.putpixel((pix, t), color)
    return image

# Calculate ridge-points
def draw_line(image, array, color, row_len, col_len):
    a = zeros(shape=(row_len, col_len))
    for col in range(col_len):
        d = collections.Counter(array[:, col])
        sum_dict = sum(d.values())
        for row in range(row_len):
            if row in d: a[row, col] = float(d[row]) / sum_dict
    return simple_draw_edge(image, a, color, 3)

# Transitional Probability Function-using Inverse Log
def inverse_log_probability(sample, s_i):
    if abs(sample - s_i) == 0: return 0.9
    elif abs(sample - s_i) <= 1: return 0.8
    elif abs(sample - s_i) <= 2: return 0.7
    else: return (1.0 / math.log(abs(sample - s_i), 2))
    
# Transitional Probability Function-using normpdf
def normpdf(x, mean, sd):
    var = float(sd) ** 2
    pi = 3.1415926
    denom = (2 * pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom

# Transitional Probability
def trans_probability(s_i, mean, std_dev):
#     return inverse_log_probability(mean, s_i)
    return normpdf(s_i, mean, std_dev)
#     return norm.pdf(s_i,mean, std_dev)

# MCMC Sampling using Gibbs method
def mcmc_draw_edge(image, gradient_image, color, thickness, gt_row=-1, gt_col=-1):
    samples = []
    row_len, col_len = gradient_image.shape[0], gradient_image.shape[1]
    # Normalize gradient_image
    evidence = gradient_image / (sum(gradient_image, axis=0) + NON_ZERO_CONSTANT)
    
    # Pick initial sample
    sample_instance = []
    for pix in xrange(col_len):
        row_index = where(evidence[:, pix] == max(evidence[:, pix]))[0][0]
        sample_instance.append(row_index)     
    samples.append(sample_instance)
#     print samples
    
    # Start Sampling
    for K in xrange(TRAINING_TIME):
        if K % 50:
            print K,
        else:
            print K
            
        for pix in xrange(col_len):
            if gt_col >= 0 and gt_row >= 0 and gt_col == pix:
                sample_instance[pix] = gt_row
                continue
            
            s_dist = []
            for s in xrange(row_len):
                if pix == 0:
                    if gt_col >= 0 and gt_row >= 0 and gt_col == pix + 1:
                        prob = trans_probability(sample_instance[pix + 1], s, 1) * (1.0 / row_len)
                    else:
                        prob = trans_probability(sample_instance[pix + 1], s, 10) * (1.0 / row_len)
                        
                elif pix == (col_len - 1):
                    if gt_col >= 0 and gt_row >= 0 and gt_col == pix - 1:
                        prob = trans_probability(s, sample_instance[pix - 1], 1) * (1.0 / row_len)
                    else:
                        prob = trans_probability(s, sample_instance[pix - 1], 10) * (1.0 / row_len)
                else:
                    if gt_col >= 0 and gt_row >= 0 and (gt_col == pix - 1 or gt_col == pix - 1):
                        prob = trans_probability(s, sample_instance[pix - 1], 1) * trans_probability(sample_instance[pix + 1], s, 1)
                    else:
                        prob = trans_probability(s, sample_instance[pix - 1], 10) * trans_probability(sample_instance[pix + 1], s, 10)
                
#                 if samples[K][pix] == s: s_dist.append(evidence[s, pix] * prob)
#                 else: s_dist.append((1.0 - evidence[s, pix]) * prob)
                s_dist.append(evidence[s, pix] * prob)
            
            # Normalize probabilty distribution             
            s_dist = array(s_dist)
            s_dist = s_dist / (sum(s_dist) + NON_ZERO_CONSTANT)
                
            r = random.random()
            for p in xrange(len(s_dist)):
                if s_dist[p] >= r:
                    e = p
                    break
                else:
                    r -= s_dist[p]

            sample_instance[pix] = e
            samples.append(sample_instance)
    
    return draw_line(image, array(samples), color, row_len, col_len)
            
    
# main program
#
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
# imsave('edges.jpg', edge_strength)

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
ridge = [ edge_strength.shape[0] / 2 ] * edge_strength.shape[1]

# output answer
imsave(output_filename, simple_draw_edge(input_image, edge_strength, (255,0, 0), 3) )
imsave(output_filename, mcmc_draw_edge(input_image, edge_strength, (0, 255, 0), 3))
imsave(output_filename, mcmc_draw_edge(input_image, edge_strength, (0, 0, 255), 3, gt_row, gt_col))
# imsave(output_filename, draw_edge(input_image, ridge, (255, 0, 0), 5))
    