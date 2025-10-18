import pickle
import os
import pandas as pd
import numpy as np

train_file = "extended_mnist_train.pkl"

with open(train_file, "rb") as fp:
    train=pickle.load(fp)

list1=[]
list2=[]
for matrix, label in train:
    list1.append(np.array(matrix.flatten())/255)
    zeros=np.zeros(10)
    zeros[label]=1
    list2.append(zeros)

matrixes=np.array(list1)
labels=np.array(list2)


weights=np.random.randn(10, 784)*0.1
biases=np.zeros(10)


epochs=100
learning_rate=0.001

for epoch in range(epochs):
    print(f"{epoch}/{epochs}: AverageCrossEntropy= ",end="")
    crossEntropyAverage=0
    for x , target in zip(matrixes,labels):
        z=np.dot(weights, x)+biases
        above=np.exp(z)
        bellow=np.sum(above)
        activators=above/bellow
        ## For the Cross Entropy/Error
        crossEntropy=-np.sum(np.dot(target, np.log10(activators)))
        crossEntropyAverage+=crossEntropy
        ##Training
        weights+=learning_rate*np.outer((target-activators),x)
        biases+=learning_rate*(target-activators)
    print(f"{crossEntropyAverage/len(matrixes)}")


np.save("weights.npy", np.array(weights))
np.save("bias.npy", np.array(biases))
## Test
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
