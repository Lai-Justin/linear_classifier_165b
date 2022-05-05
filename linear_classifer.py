# Starter code for CS 165B HW2 Spring 2022
from calendar import c
from pickletools import TAKEN_FROM_ARGUMENT1
import numpy as np
import math

from setuptools import find_namespace_packages

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition. 
    You are permitted to use the numpy library but you must write 
    your own code for the linear classifier. 

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values 

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """

    #if(i <= testing_input[0][1]):
    #        val = 0
    #elif((i > testing_input[0][1]) & (i <= (testing_input[0][1] + testing_input[0][2]))):
    #    val = 1
    #else:
    #    val = 2

    centroids = np.zeros(((len(training_input[0]) - 1), training_input[0][0]))
    rangeKeeper = 0

    for i in range(1, len(training_input)):
        if(i <= testing_input[0][1]):
            rangeKeeper = 0
        elif((i > testing_input[0][1]) & (i <= (testing_input[0][1] + testing_input[0][2]))):
            rangeKeeper = 1
        else:
            rangeKeeper = 2
        for x in range(training_input[0][0]):
            centroids[rangeKeeper][x] += training_input[i][x]    
    
    

    #rangeKeeper = training_input[0][1] 
    #start = 1
    #for i in range(len(training_input[0]) - 1):
    #    for j in range(start, rangeKeeper + 1):
    #        for x in range(training_input[0][0]):
    #            centroids[i][x] += training_input[j][x]        
    #            
    #    start += training_input[0][1] 
    #    rangeKeeper += training_input[0][1] 
   


    for i in range(len(centroids)):
        for j in range(len(centroids[i])):
            centroids[i][j] /= training_input[0][1]
    
    print(centroids)
    
    weights = np.zeros((int(((len(training_input[0]) - 1) * (len(training_input[0]) - 2)) / 2), training_input[0][0]))
    count = 0
    
    for i in range(len(centroids) - 1):
        for j in range(i + 1, len(centroids)):
            for k in range(int(centroids[0][0])):
                weights[count][k] = centroids[i][k] - centroids[j][k]
            count += 1
    


    weights_opposite = np.zeros((int(((len(training_input[0]) - 1) * (len(training_input[0]) - 2)) / 2), training_input[0][0]))
    count1 = 0
    
    for i in range(len(centroids) - 1):
        for j in range(i + 1, len(centroids)):
            for k in range(int(centroids[0][0])):
                weights_opposite[count1][k] = centroids[i][k] + centroids[j][k]
            count1 += 1

    
    thresholds = np.zeros((int(((len(training_input[0]) - 1) * (len(training_input[0]) - 2)) / 2), 1))
    for i in range(len(weights)):
        thresholds[i] = (np.dot(weights[i], weights_opposite[i])) / 2



    final_classes = np.zeros(((len(testing_input[0]) - 1),(len(testing_input[0]) - 1)))
    
    val = 0


    for i in range(1, len(testing_input)):
        if(i <= testing_input[0][1]):
            val = 0
        elif((i > testing_input[0][1]) & (i <= (testing_input[0][1] + testing_input[0][2]))):
            val = 1
        else:
            val = 2
        if(np.dot(testing_input[i], weights[0]) >= thresholds[0]):
            if(np.dot(testing_input[i], weights[1]) >= thresholds[1]):
                final_classes[val][0] += 1
            else:
                final_classes[val][2] += 1
        else:
            if(np.dot(testing_input[i], weights[2]) >= thresholds[2]):
                final_classes[val][1] += 1
            else:
                final_classes[val][2] += 1


    tp_a = final_classes[0][0]
    tp_b = final_classes[1][1]
    tp_c = final_classes[2][2]

    tn_a = final_classes[1][1] + final_classes[2][1] + final_classes[1][2] + final_classes[2][2]
    tn_b = final_classes[0][0] + final_classes[0][2] + final_classes[2][0] + final_classes[2][2]
    tn_c = final_classes[0][0] + final_classes[0][1] + final_classes[1][0] + final_classes[1][1]

    fp_a = final_classes[1][0] + final_classes[2][0]
    fp_b = final_classes[0][1] + final_classes[2][1]
    fp_c = final_classes[0][2] + final_classes[1][2]

    fn_a = final_classes[0][1] + final_classes[0][2]
    fn_b = final_classes[1][0] + final_classes[1][2]
    fn_c = final_classes[2][0] + final_classes[2][1]

    tpr_a = tp_a / testing_input[0][1]
    tpr_b = tp_b / testing_input[0][2]
    tpr_c = tp_c / testing_input[0][3]
    tpr = (tpr_a + tpr_b + tpr_c) / 3

    fpr_a = fp_a / (testing_input[0][2] + testing_input[0][3])
    fpr_b = fp_b / (testing_input[0][3] + testing_input[0][1])
    fpr_c = fp_c / (testing_input[0][1] + testing_input[0][2])
    fpr = (fpr_a + fpr_b + fpr_c) / 3
    
    error_a = (fp_a + fn_a) / (testing_input[0][1] + testing_input[0][2]+ testing_input[0][3])
    error_b = (fp_b + fn_b) / (testing_input[0][1] + testing_input[0][2]+ testing_input[0][3])
    error_c = (fp_c + fn_c) / (testing_input[0][1] + testing_input[0][2]+ testing_input[0][3])

    error = (error_a + error_b + error_c) / 3

    accuracy_a = (tp_a + tn_a) / (testing_input[0][1] + testing_input[0][2] + testing_input[0][3])
    accuracy_b = (tp_b + tn_b) / (testing_input[0][1] + testing_input[0][2] + testing_input[0][3])
    accuracy_c = (tp_c + tn_c) / (testing_input[0][1] + testing_input[0][2] + testing_input[0][3])
    accuracy = (accuracy_a + accuracy_b + accuracy_c) / 3

    predicted_a_counter = final_classes[0][0] + final_classes[1][0] + final_classes[2][0]
    predicted_b_counter = final_classes[0][1] + final_classes[1][1] + final_classes[2][1]
    predicted_c_counter = final_classes[0][2] + final_classes[1][2] + final_classes[2][2]
   

    precision_a = tp_a / predicted_a_counter
    precision_b = tp_b / predicted_b_counter
    precision_c = tp_c / predicted_c_counter
    precision = (precision_a + precision_b + precision_c) / 3

    #print(tpr)
    #print(fpr)
    #print(error)
    #print(accuracy)
    #print(precision)
    #print(final_classes)

    return {
                "tpr": tpr,
                "fpr": fpr,
                "error_rate": error,
                "accuracy": accuracy, 
                "precision": precision
            }


    # TODO: IMPLEMENT
    pass



#######
# The following functions are provided for you to test your classifier.
######
def parse_file(filename):
    """
    This function is provided to you as an example of the preprocessing we do
    prior to calling run_train_test
    """
    with open(filename, "r") as f:
        data = [[float(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]
        

        return data

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw2.py [training file path] [testing file path]
    """
    import sys

    training_input = parse_file(sys.argv[1])
    testing_input = parse_file(sys.argv[2])
    

    run_train_test(training_input, testing_input)

