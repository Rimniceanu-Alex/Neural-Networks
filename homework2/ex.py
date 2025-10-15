import pickle
import os
import pandas as pd
import numpy as np

train_file = "extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train=pickle.load(fp)

lsit1=[]
list_labels=[]
for i1, i2 in train:
    lsit1.append(np.array(i1.flatten()))
    list_labels.append(i2)

matrixes=np.array(lsit1)

list_bias=np.load("bias.npy")
list_weights=np.load("weights.npy")
list_classifciations=np.load("classifications.npy")

def chainPerceptron(matrixes:any, list_labels:any, list_weights:any, list_bias:any):
    correct=0
    false_poisitive=0
    false_negatives=0
    for x, y in zip(matrixes, list_labels):
        result=0
        digit_index=0
        while(result==0):
            if digit_index==10:
                false_negatives+=1
                break
            product=np.dot(x, list_weights[digit_index])+list_bias[digit_index]
            if product>=0:
                result=1
            else:
                result=0
            if result==1:
                if y==digit_index:
                    correct+=1
                    break
                else:
                    false_poisitive+=1
                    break
            else:
                digit_index+=1

            
    print(f"Out of the {len(matrixes)} we have {correct} classifications =>{correct*100/len(matrixes)}% Accuracy\nFalsePositives={false_poisitive}\nFalse Negatives={false_negatives}")


# Using the most Certain Perceptron for a specific Digit
def certainPerceptron(matrixes:any, list_labels:any, list_weights:any, list_bias:any):
    correct=0
    false_poisitive=0
    false_negatives=0
    for x, y in zip(matrixes, list_labels):
        list_products=[]
        digit=0
        for weight, bias in zip(list_weights, list_bias):
            product=np.dot(x, weight)+bias
            list_products.append((product, digit))
            digit+=1
        product, classified_digit=max(list_products)
        if product>=0 :  
            if classified_digit==y:
                correct+=1
            else:
                false_poisitive+=1
        else:
            false_negatives+=1
    print(f"Out of the {len(matrixes)} we have {correct} classifications =>{correct*100/len(matrixes)}% Accuracy\nFalsePositives={false_poisitive}\nFalse Negatives={false_negatives}")

certainPerceptron(matrixes, list_labels, list_weights, list_bias)
print("\n\n\n")
chainPerceptron(matrixes, list_labels, list_weights, list_bias)
# Make functions for both possible ways