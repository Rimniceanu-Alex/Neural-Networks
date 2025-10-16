import pickle
import os
import pandas as pd
import numpy as np

train_file = "extended_mnist_train.pkl"
test_file="extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train=pickle.load(fp)

list1=[]
list2=[]
for matrix, label in train:
    list1.append(np.array(matrix.flatten())/255)
    list2.append(label)

matrixes=np.array(list1)
labels=np.array(list2)

weights=np.zeros(784, 10)

list_bias=[]
list_classifications=[]
learning_term=0.001
epochs=100
for i in range(10):
        calssification=np.where(labels==i, 1, 0)
        list_classifications.append(calssification)
        weights=np.zeros(784)
        bias=0
        list_weights.append(weights)
        list_bias.append(bias)

for epoch in range(epochs):
        print(f"epoch {epoch}\{epochs} started:")
        for x, category in zip(matrixes, labels):
            list_activators=[]
            list_product=[]
            for i in range(10):
                product=np.dot(x, list_weights[i])+list_bias[i]
                list_product.append(product)
            for i in range(10):
                pr_sum=0
                for pr in list_product:
                     pr_sum+=np.exp(pr)
                activator=np.exp(list_product[i])/pr_sum
                list_activators.append(activator)
            chosen=max(list_activators)
            classification=np.zeros(10)
            classification[category]=1
            crossEntropy=0
            for i in range(10):
                crossEntropy-=classification[i]*np.log10(list_activators[i])
            for i in range(10):
                 weights+=learning_term*classification[i]-list_activators[i]*x
np.save("weights.npy", np.array(list_weights))
np.save("bias.npy", np.array(list_bias))
np.save("classifications", np.array(list_classifications))
for i in range(10):
    correct=0
    for x, y in zip(matrixes, list_classifications[i]):
        product=np.dot(x, list_weights[i])+list_bias[i]
        if product>=0.5:
            result=1
        else:
            result=0
        if y ^ result==0:
            correct+=1
    print(f"For {i}: Out of the {len(matrixes)} we have {correct} classifications =>{correct*100/len(matrixes)}% Accuracy")