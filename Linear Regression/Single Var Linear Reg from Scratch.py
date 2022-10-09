# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
w = 0.0 
b = 0.0
Lr = 0.01 #Learning Rate
x_train = np.array([1.0, 1.2, 1.6, 2.0, 2.9, 3.0, 4.6, 5.0]) #size of houses in 1000 sqft, 1.2-> 1200sqft, 3.0->3000sqft
y_train = np.array([300.0, 340.0, 420.0, 500.0, 680.0, 700.0, 1020.0, 1100.0]) #price of houses in $1000s, 420.0 -> $420,000
#x_train = x
#y_train = y
m = x_train.shape[0] #Actual Training Examples
ep = 100  #no. of Epoches required to run the model(can be changed)
sum_w = 0 #temp_w
sum_b = 0 #temp_b
for epoches in range(ep):
    #Calculating the Gradient Descent...
    for i in range(m):
        sum_w += ((w*x_train[i]+b) - y_train[i])* x_train[i]   # ((mx+c) - y) * x ---> (1/m) * ((w.x_i + b) - y_i) * x_i
        sum_b += ((w*x_train[i]+b) - y_train[i])   # 1/m ((mx+c) - y) * x ---> (1/m) * ((w.x_i + b) - y_i)
    sum_w *= 1/m #1/m ____
    sum_b *= 1/m #1/m ____
    
    w -= Lr * sum_w # w = w - Learning Rate * temp_w
    b -= Lr * sum_b # b = b - Learning Rate * temp_b
    print("w at ", epoches+1, "th epoch: ", w) #Printing w at every epoch
    print("b at ", epoches+1, "th epoch: ", b) #Printing b at every epoch
    
    Y_pred = (w*x_train) +b # The Prediction of Model in that Iternation
    print("cost at ", epoches+1,": ", (1/(2*m)* (Y_pred-y_train[epoches%m])**2).mean())
    clear_output(wait=True) # Clearing the output of graph
    plt.xlim([min(x_train)-1, max(x_train)+1]) #setting the Limits of X on graph
    plt.ylim([min(y_train)-50, max(y_train)+50]) #setting the limits of Y on graph
    plt.scatter(x_train, y_train) #scattering the points on graph
    plt.plot(x_train, Y_pred, 'red') #plotting the prediction line in that iteration/epoch
    plt.show()
print('Finished Training the Model')
print('-------------------------------------------')
error = np.subtract(y_train,Y_pred).mean()
print("Error: ", error)
print('-------------------------------------------')
val = float(input("Enter the size of house to predice it's Price: "))
newpred = w*(val)+b
print ("The Price of House is Approximately ---> $",round(newpred,0), "k")