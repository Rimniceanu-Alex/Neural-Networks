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


weights_i_h=np.random.randn(100, 784)*0.1
biases_i_h=np.zeros(100)
weights_h_o=np.random.randn(10, 100)*0.1
biases_h_o=np.zeros(10)

epochs=100
learning_rate=0.001

for epoch in range(epochs):
    print(f"{epoch}/{epochs}: AverageCrossEntropy= ",end="")
    crossEntropyAverage=0
    # combination=list(zip(matrixes, labels))
    # np.random.shuffle(combination)
    # matrixes, labels=zip(*combination)
    # matrixes=np.array(matrixes)
    # labels=np.array(labels)
    for x , target in zip(matrixes,labels):
        # Input->Hidden
        z1=np.dot(weights_i_h, x)+biases_i_h # avem un vector cu toate dot-producturile hident layer-elor
        # Leaky Relu
        activation1=np.where(z1<0 , 0.2*z1, z1)
        # Hidden->Output
        z2=np.dot(weights_h_o, activation1)+ biases_h_o
         # SoftMax
        above=np.exp(z2)
        bellow=np.sum(above)
        activation2=above/bellow
        ## For the Cross Entropy/Error
        crossEntropy=-np.sum(np.dot(target, np.log10(activation2)))
        crossEntropyAverage+=crossEntropy
        ##Training Hiddent->Output
        delta2=activation2-target
        weights_h_o-=learning_rate*np.outer(delta2,activation1)
        biases_h_o-=learning_rate*delta2
        ##Backpropagation Training Input->Hidden
        leaky_relu_deriv=np.where(z1>0, 1, 0.2)
        weights_h_oT=np.transpose(weights_h_o.copy())
        delta1=np.dot(weights_h_oT, delta2)*leaky_relu_deriv
        weights_i_h-=learning_rate*np.outer(delta1,x)
        biases_i_h-=learning_rate*delta1
    print(f"{crossEntropyAverage/len(matrixes)}")


np.save("weights_i_h.npy", np.array(weights_i_h))
np.save("bias_i_h.npy", np.array(biases_i_h))
np.save("weights_h_o.npy", np.array(weights_h_o))
np.save("bias_h_o.npy", np.array(biases_h_o))
## Test
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
    digit=np.argmax(y)
    if classified_digit==digit:
        correct+=1
print(f"Out of {len(matrixes)} samples, {correct} where classified correctly\nAccuracy={correct*100/len(matrixes)}")