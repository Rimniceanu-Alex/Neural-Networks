import pickle
import os
import pandas as pd
import numpy as np

train_file = "extended_mnist_train.pkl"
test_file="extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train=pickle.load(fp)

with open(test_file, "rb") as fp:
    test=pickle.load(fp)

list1=[]
list2=[]
for matrix, label in train:
    list1.append(np.array(matrix.flatten()))
    list2.append(label)

matrixes=np.array(list1)
labels=np.array(list2)

list_weights=[]
list_bias=[]
list_classifications=[]
for i in range(10):
    calssification=np.where(labels==i, 1, 0)
    list_classifications.append(calssification)
    weights=np.random.randn(784)*0.1
    learning_term=0.0001
    bias=0
    epochs=1000
    for epoch in range(epochs):
        print(f"epoch {epoch}\{epochs} started for digit {i} :")
        for x, category in zip(matrixes, calssification):
            product=np.dot(x, weights)+bias
            if product >=0:
                response=1 
            else:
                response=0
            update=(category-response)*learning_term
            weights+=update*x
            bias+=update
    list_weights.append(weights)
    list_bias.append(bias)
np.save("weights.npy", np.array(list_weights))
np.save("bias.npy", np.array(list_bias))
np.save("classifications", np.array(list_classifications))
for i in range(10):
    correct=0
    for x, y in zip(matrixes, list_classifications[i]):
        product=np.dot(x, list_weights[i])+list_bias[i]
        if product>=0:
            result=1
        else:
            result=0
        if y ^ result==0:
            correct+=1
    print(f"For {i}: Out of the {len(matrixes)} we have {correct} classifications =>{correct*100/len(matrixes)}% Accuracy")