import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
#size of houses in 1000 sqft, 1.2-> 1200sqft, 3.0->3000sqft
X_train = np.array([[1.24e+03, 3.00e+00, 1.00e+00, 6.40e+01],
 [1.95e+03, 3.00e+00, 2.00e+00, 1.70e+01],
 [1.72e+03, 3.00e+00, 2.00e+00, 4.20e+01],
 [1.96e+03, 3.00e+00, 2.00e+00, 1.50e+01],
 [1.31e+03, 2.00e+00, 1.00e+00, 1.40e+01],
 [8.64e+02, 2.00e+00, 1.00e+00, 6.60e+01],
 [1.84e+03, 3.00e+00, 1.00e+00, 1.70e+01],
 [1.03e+03, 3.00e+00, 1.00e+00, 4.30e+01],
 [3.19e+03, 4.00e+00, 2.00e+00, 8.70e+01],
 [7.88e+02, 2.00e+00, 1.00e+00, 8.00e+01],
 [1.20e+03, 2.00e+00, 2.00e+00, 1.70e+01],
 [1.56e+03, 2.00e+00, 1.00e+00, 1.80e+01],
 [1.43e+03, 3.00e+00, 1.00e+00, 2.00e+01],
 [1.22e+03, 2.00e+00, 1.00e+00, 1.50e+01],
 [1.09e+03, 2.00e+00, 1.00e+00, 6.40e+01],
 [8.48e+02, 1.00e+00, 1.00e+00, 1.70e+01],
 [1.68e+03, 3.00e+00, 2.00e+00, 2.30e+01],
 [1.77e+03, 3.00e+00, 2.00e+00, 1.80e+01],
 [1.04e+03, 3.00e+00, 1.00e+00, 4.40e+01],
 [1.65e+03, 2.00e+00, 1.00e+00, 2.10e+01],
 [1.09e+03, 2.00e+00, 1.00e+00, 3.50e+01],
 [1.32e+03, 3.00e+00, 1.00e+00, 1.40e+01],
 [1.59e+03, 0.00e+00, 1.00e+00, 2.00e+01],
 [9.72e+02, 2.00e+00, 1.00e+00, 7.30e+01],
 [1.10e+03, 3.00e+00, 1.00e+00, 3.70e+01],
 [1.00e+03, 2.00e+00, 1.00e+00, 5.10e+01],
 [9.04e+02, 3.00e+00, 1.00e+00, 5.50e+01],
 [1.69e+03, 3.00e+00, 1.00e+00, 1.30e+01],
 [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02],
 [1.42e+03, 3.00e+00, 2.00e+00, 1.90e+01],
 [1.16e+03, 3.00e+00, 1.00e+00, 5.20e+01],
 [1.94e+03, 3.00e+00, 2.00e+00, 1.20e+01],
 [1.22e+03, 2.00e+00, 2.00e+00, 7.40e+01],
 [2.48e+03, 4.00e+00, 2.00e+00, 1.60e+01],
 [1.20e+03, 2.00e+00, 1.00e+00, 1.80e+01],
 [1.84e+03, 3.00e+00, 2.00e+00, 2.00e+01],
 [1.85e+03, 3.00e+00, 2.00e+00, 5.70e+01],
 [1.66e+03, 3.00e+00, 2.00e+00, 1.90e+01],
 [1.10e+03, 2.00e+00, 2.00e+00, 9.70e+01],
 [1.78e+03, 3.00e+00, 2.00e+00, 2.80e+01],
 [2.03e+03, 4.00e+00, 2.00e+00, 4.50e+01],
 [1.78e+03, 4.00e+00, 2.00e+00, 1.07e+02],
 [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02],
 [1.55e+03, 3.00e+00, 1.00e+00, 1.60e+01],
 [1.95e+03, 3.00e+00, 2.00e+00, 1.60e+01],
 [1.22e+03, 2.00e+00, 2.00e+00, 1.20e+01],
 [1.62e+03, 3.00e+00, 1.00e+00, 1.60e+01],
 [8.16e+02, 2.00e+00, 1.00e+00, 5.80e+01],
 [1.35e+03, 3.00e+00, 1.00e+00, 2.10e+01],
 [1.57e+03, 3.00e+00, 1.00e+00, 1.40e+01],
 [1.49e+03, 3.00e+00, 1.00e+00, 5.70e+01],
 [1.51e+03, 2.00e+00, 1.00e+00, 1.60e+01],
 [1.10e+03, 3.00e+00, 1.00e+00, 2.70e+01],
 [1.76e+03, 3.00e+00, 2.00e+00, 2.40e+01],
 [1.21e+03, 2.00e+00, 1.00e+00, 1.40e+01],
 [1.47e+03, 3.00e+00, 2.00e+00, 2.40e+01],
 [1.77e+03, 3.00e+00, 2.00e+00, 8.40e+01],
 [1.65e+03, 3.00e+00, 1.00e+00, 1.90e+01],
 [1.03e+03, 3.00e+00, 1.00e+00, 6.00e+01],
 [1.12e+03, 2.00e+00, 2.00e+00, 1.60e+01],
 [1.15e+03, 3.00e+00, 1.00e+00, 6.20e+01],
 [8.16e+02, 2.00e+00, 1.00e+00, 3.90e+01],
 [1.04e+03, 3.00e+00, 1.00e+00, 2.50e+01],
 [1.39e+03, 3.00e+00, 1.00e+00, 6.40e+01],
 [1.60e+03, 3.00e+00, 2.00e+00, 2.90e+01],
 [1.22e+03, 3.00e+00, 1.00e+00, 6.30e+01],
 [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02],
 [2.60e+03, 4.00e+00, 2.00e+00, 2.20e+01],
 [1.43e+03, 3.00e+00, 1.00e+00, 5.90e+01],
 [2.09e+03, 3.00e+00, 2.00e+00, 2.60e+01],
 [1.79e+03, 4.00e+00, 2.00e+00, 4.90e+01],
 [1.48e+03, 3.00e+00, 2.00e+00, 1.60e+01],
 [1.04e+03, 3.00e+00, 1.00e+00, 2.50e+01],
 [1.43e+03, 3.00e+00, 1.00e+00, 2.20e+01],
 [1.16e+03, 3.00e+00, 1.00e+00, 5.30e+01],
 [1.55e+03, 3.00e+00, 2.00e+00, 1.20e+01],
 [1.98e+03, 3.00e+00, 2.00e+00, 2.20e+01],
 [1.06e+03, 3.00e+00, 1.00e+00, 5.30e+01],
 [1.18e+03, 2.00e+00, 1.00e+00, 9.90e+01],
 [1.36e+03, 2.00e+00, 1.00e+00, 1.70e+01],
 [9.60e+02, 3.00e+00, 1.00e+00, 5.10e+01],
 [1.46e+03, 3.00e+00, 2.00e+00, 1.60e+01],
 [1.45e+03, 3.00e+00, 2.00e+00, 2.50e+01],
 [1.21e+03, 2.00e+00, 1.00e+00, 1.50e+01],
 [1.55e+03, 3.00e+00, 2.00e+00, 1.60e+01],
 [8.82e+02, 3.00e+00, 1.00e+00, 4.90e+01],
 [2.03e+03, 4.00e+00, 2.00e+00, 4.50e+01],
 [1.04e+03, 3.00e+00, 1.00e+00, 6.20e+01],
 [1.62e+03, 3.00e+00, 1.00e+00, 1.60e+01],
 [8.03e+02, 2.00e+00, 1.00e+00, 8.00e+01],
 [1.43e+03, 3.00e+00, 2.00e+00, 2.10e+01],
 [1.66e+03, 3.00e+00, 1.00e+00, 6.10e+01],
 [1.54e+03, 3.00e+00, 1.00e+00, 1.60e+01],
 [9.48e+02, 3.00e+00, 1.00e+00, 5.30e+01],
 [1.22e+03, 2.00e+00, 2.00e+00, 1.20e+01],
 [1.43e+03, 2.00e+00, 1.00e+00, 4.30e+01],
 [1.66e+03, 3.00e+00, 2.00e+00, 1.90e+01],
 [1.21e+03, 3.00e+00, 1.00e+00, 2.00e+01],
 [1.05e+03, 2.00e+00, 1.00e+00, 6.50e+01]])
y_train = np.array([300.0, 340.0, 420.0, 500.0, 680.0, 700.0, 1020.0, 1100.0]) #price of houses in $1000s, 420.0 -> $420,000
Y_train = np.array([300.,   509.8,  394.,   540.,   415.,   230.,   560.,   294.,   718.2,  200.,
 302.,   468.,   374.2,  388.  , 282. ,  311.8,  401. ,  449.8  ,301. ,  502.,
 340.,   400.28, 572. ,  264.  , 304. ,  298. ,  219.8,  490.7  ,216.96, 368.2,
 280.,   526.87, 237. ,  562.43, 369.8,  460. ,  374. ,  390.   ,158.  , 426.,
 390.,   277.77, 216.96, 425.8 , 504. ,  329. ,  464. ,  220.   ,358.  , 478.,
 334.,   426.98, 290.  , 463.  , 390.8,  354. ,  350. ,  460.   ,237.  , 288.3,
 282.,   249.,   304.  , 332.  , 351.8,  310. ,  216.96, 666.34 ,330.  , 480.,
 330.3,  348.,   304.  , 384.  , 316. ,  430.4,  450.  , 284.   ,275.  , 414.,
 258.,   378.,   350.  , 412.  , 373. ,  225. ,  390.  , 267.4  ,464.  , 174.,
 340.,   430.,   440.  , 216.  , 329. ,  388. ,  390.  , 356.   ,257.8 ])
scalar = StandardScaler()
X_norm = scalar.fit_transform(X_train)
#print('x before scaling--> ', X_train )
#print('x after scaling--> ', x_norm )
sgdr = SGDRegressor(max_iter = 1000) #getting the regression model from lib
sgdr.fit(X_norm, Y_train) #fitting the model with given dataset
print(sgdr)
print(f"No of iterations completed: {sgdr.n_iter_}, number of weights updated: {sgdr.t_} ") 
w_norm = sgdr.coef_ #getting the values of 'W' after fitting the data
b_norm = sgdr.intercept_ #getting the values of 'B' after fitting the data
print(f"w after fitting the data: {w_norm}, b after fitting the data: {b_norm}")
x_pred = np.array([[-4.33623238e-01,  4.33808841e-01, -7.89272336e-01 , 9.93726440e-01],
 [ 1.29217810e+00,  4.33808841e-01,  1.26698980e+00, -8.29542143e-01],
 [ 7.33115693e-01 , 4.33808841e-01,  1.26698980e+00,  1.40281572e-01],
 [ 1.31648516e+00,  4.33808841e-01,  1.26698980e+00, -9.07128040e-01]]) #array for predicting the price of houses with model
pred = sgdr.predict(x_pred) #calling the predict fucntion
print(f"houses prediction---> {pred}")