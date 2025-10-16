import pickle
import os
import pandas as pd
import numpy as np

test_file="extended_mnist_test.pkl"

with open(test_file, "rb") as fp:
    test=pickle.load(fp)

list1=[]
list2=[]
for matrix, label in test:
    list1.append(np.array(matrix.flatten())/255)
    zeros=np.zeros(10)
    zeros[label]=1
    list2.append(zeros)

matrixes=np.array(list1)
labels=np.array(list2)

weights=np.load("weights.npy")
biases=np.load("bias.npy")
correct=0
for x, y in zip(matrixes, labels):
    z=np.dot(weights, x)+biases
    above=np.exp(z)
    bellow=np.sum(above)
    activators=above/bellow
    chosen_value=np.max(activators)
    classified_digit=np.where(activators==chosen_value)[0]
    classified_digit=np.argmax(activators)
    digit=np.argmax(y)
    if classified_digit==digit:
        correct+=1
print(f"Out of {len(matrixes)} samples, {correct} where classified correctly\nAccuracy={correct*100/len(matrixes)}")