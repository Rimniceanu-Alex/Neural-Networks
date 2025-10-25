import pickle
import os
import pandas as pd
import numpy as np
import time

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

epochs=27
learning_rate=0.01
batch_size=64
plateau_size=3
plateau_count=0
best_crossEntropy=1
epsilon=0.01
indices=np.arange(len(matrixes))
total_time=480
epoch_counter=0
max_duration=0.0
start=time.perf_counter()
while total_time>0:
    np.random.shuffle(indices)
    matrixes=matrixes[indices]
    labels=labels[indices]
    epoch_start=time.perf_counter()
    print(f"{epoch_counter}|{max_duration} secs per epoch: AverageCrossEntropy= ",end="")
    crossEntropySum=0
    for i in range(0, len(matrixes), batch_size):
        j=i+batch_size
        matrixes_batch=matrixes[i:j]
        labels_batch=labels[i:j]
        batch_weights_i_h=np.zeros_like(weights_i_h)
        batch_weights_h_o=np.zeros_like(weights_h_o)
        batch_biases_i_h=np.zeros_like(biases_i_h)
        batch_biases_h_o=np.zeros_like(biases_h_o)
        weights_h_oT=np.transpose(weights_h_o.copy())
        for x , target in zip(matrixes_batch,labels_batch):
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
            crossEntropySum+=crossEntropy
            ##Training Hiddent->Output
            delta2=activation2-target
            batch_weights_h_o+=np.outer(delta2,activation1)
            batch_biases_h_o+=delta2
            ##Backpropagation Training Input->Hidden
            leaky_relu_deriv=np.where(z1>0, 1, 0.2)
            delta1=np.dot(weights_h_oT, delta2)*leaky_relu_deriv
            batch_weights_i_h+=np.outer(delta1,x)
            batch_biases_i_h+=delta1
        weights_h_o-=learning_rate*batch_weights_h_o
        biases_h_o-=learning_rate*batch_biases_h_o
        weights_i_h-=learning_rate*batch_weights_i_h
        biases_i_h-=learning_rate*batch_biases_i_h  
    crossEntropyAverage=crossEntropySum/len(matrixes)
    print(f"{crossEntropyAverage}", end="")
    if crossEntropyAverage < best_crossEntropy-epsilon:
        plateau_count=0
        best_crossEntropy=crossEntropyAverage
    else:
        plateau_count+=1
        if plateau_count >=plateau_size:
            learning_rate*=0.5
            plateau_count=0
            print("\nLearning Rate Changed")
    epoch_end=time.perf_counter()
    print(f"=>{(epoch_end-epoch_start):.6f} seconds")
    epoch_time=epoch_end-epoch_start
    total_time-=epoch_time
    if epoch_time>max_duration:
        max_duration=epoch_time
    print(f"{total_time} seconds remaining")
    epoch_counter+=1
end=time.perf_counter()
final_time=int(end-start)
print(f"Training took {final_time//60}:{final_time-final_time//60*60} ")
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