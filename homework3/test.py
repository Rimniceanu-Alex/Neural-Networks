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

weights_i_h=np.load("weights_i_h.npy")
biases_i_h=np.load("bias_i_h.npy")
weights_h_o=np.load("weights_h_o.npy")
biases_h_o=np.load("bias_h_o.npy")
predictions=[]
correct=0
for x, y in zip(matrixes, labels):
    z1=np.dot(weights_i_h, x)+biases_i_h
    activation1= np.where(z1<0 , 0.2*z1, z1)
    z2=np.dot(weights_h_o, activation1)+ biases_h_o
    above=np.exp(z2)
    bellow=np.sum(above)
    activation2=above/bellow
    chosen_value=np.max(activation2)
    classified_digit=np.where(activation2==chosen_value)[0]
    classified_digit=np.argmax(activation2)
    predictions.append(classified_digit)
    digit=np.argmax(y)
    if classified_digit==digit:
        correct+=1
print(f"Out of {len(matrixes)} samples, {correct} where classified correctly\nAccuracy={correct*100/len(matrixes)}")

predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)