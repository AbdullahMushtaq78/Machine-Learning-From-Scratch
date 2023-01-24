# import all necessary libraries 
import numpy as np                                    # For matrices and MATLAB like functions                  
from sklearn.model_selection import train_test_split  # To split data into train and test set
# for plotting graphs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Viz_Data = True

def plot_predictions(testY, y_pred, output_type):
    plt.figure(figsize=(15,8))
    plt.plot(testY.squeeze(), linewidth=2 , label="True")
    plt.plot(y_pred.squeeze(), linestyle="--",  label="Predicted")
    plt.title(output_type)
    plt.legend()
    plt.show()
    
def plot_losses(epoch_loss,output_type):
    plt.plot(epoch_loss, linewidth=2)
    plt.title(output_type)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.show()

# ### Data Generation
# 
# Input Feature Vector : X = [x_1 ,  x_2]^T 
# Target Variable : Y$ 
# Y = x_1^2 + x_2^3 + x_1x_2
def generate_data(n_samples=1500):

    X = np.random.uniform(-5,5, (n_samples, 5) ).astype(np.float32)

    Y = (X[:,0]**2 + X[:,1]**3 + X[:,0]*X[:, 1]).astype(np.float32)
    return X, Y


def visualize_data(X, Y):
    # ### Visualize Mapping from Input Feature Vector X = [x_1 ,  x_2]^T  to target variable Y
    fig = plt.figure(figsize=(20,10))
    ax = plt.axes(projection='3d')

    ax.plot_trisurf(X[:, 0],X[:, 1],Y, cmap='twilight_shifted')
    ax.view_init(30, -40)

    ax.set_xlabel(r'$x_1$', fontsize=20)
    ax.set_ylabel(r'$x_2$', fontsize=20)
    ax.set_zlabel(r'$y$'  , fontsize=20)
    plt.show()


class Network(object):        
    
    
    def __init__(self,  n_features = 2):        
        
        self.n_features  = n_features
        self.w = np.random.normal(size=(1,n_features)) 
        self.b = np.random.normal(size=(1,n_features))
    
    def __str__(self):
        
        msg = "Linear Regression:\n\nSize of Input = " + str(self.n_features)                
        return  msg

    def forward(self, x):

        y_hat = ((self.w*x) +self.b).sum(axis=1) 
        return y_hat

    def loss(self, y, y_hat):
        n=y.size
        loss =  ((y_hat-y)**2).mean()
        # Another optimised loss function to further scale the loss value (Double Mean instead of 1 Mean + 1 Sum)
        # loss = (1/(2*self.n_features)* ((y_hat-y)**2).mean())
        return loss
     
    def backward(self , y_hat , x , y , lr):

        batch_size = y_hat.shape[0]
        #Calculating gradients for achieving minima and for convergence.
        sum_w = ((y_hat - y)  * x.T).sum(axis=1) # ((mx+c) - y) * x ---> (1/m) * ((w.x_i + b) - y_i) * x_i^T
        sum_b = (y_hat- y).sum()   # 1/m ((mx+c) - y) * x ---> (1/m) * ((w.x_i + b) - y_i)
        sum_w *= 1/batch_size #1/m 
        sum_b *= 1/batch_size #1/m 

        #Updating both W and B values.
        self.w -= lr * sum_w # w = w - Learning Rate * temp_w
        self.b -= lr * sum_b # b = b - Learning Rate * temp_b 



def train(model, n_epochs, lr, trainX, trainY, valX, valY):
    k,l=trainX.shape
    model = Network( n_features = l  )
    print(model)
    lr = lr
    n_epochs = n_epochs
    n_examples = trainX.shape[0]
    epoch_loss = []
    val_loss= []
    print("\n\nTraining...")
    for epoch in range(n_epochs): # training the model for each epoch
        loss=0 # zeroing the loss at each epoch
        # Your implementation ???
        y_hat = model.forward(trainX) # Saving the predicted outcomes to use in loss
        loss = model.loss(trainY, y_hat) # Calulating loss 
        epoch_loss.append(loss) # saving loss at each epoch for plotting graph
        model.backward(y_hat, trainX, trainY,lr) # Back propagation for both paramenters
        #Validation
        val_hat = model.forward(valX) # predicting the outcomes of validation dataset.
        val_L = model.loss(valY, val_hat)  # calculating the loss for validation dataset.
        val_loss.append(val_L) # saving the loss for plotting the val_loss graph
    #final prediction for validation data
    val_preds = model.forward(valX) # final prediction for validation dataset.
    plot_predictions(valY, val_preds,"validation_pred_without_batch")   
    print("\nDone.")    

    return model, epoch_loss, val_loss


def main():
    X, Y = generate_data() # Generating data
    if Viz_Data:
        visualize_data(X, Y)
    # Split the dataset into training and testing and validation
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.15, random_state=42) # split training and testing data to train and test the model.
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.15, random_state=42)  # 0.25 x 0.8 = 0.2 #extracting the validation dataset frm the training dataset for validation purposes




    print("Shape of Train Data:")
    print("TrainX: " , trainX.shape)
    print("TrainY: " , trainY.shape)
    print("\nShape of Test Data:")
    print("TestX: " , testX.shape)
    print("TestY: " , testY.shape)
    
    
    # ### Creating a Randomly Initialized Neural Network
    k,l=trainX.shape
    model = Network( n_features = 5 )
    print(model)
    # ### Printing Randomly Initialized theta
    print("Theta = \n", model.w)
    # Take 10 exampls to check if everything is working
    x = trainX[:10]
    y = trainY[:10]
    lr = 0.1
    y_hat = model.forward( x )
    print("theta matrix before weight update:")
    print("\ntheta = \n", model.w)
    model.backward( y_hat , x , y , lr)
    print("\n\ntheta matrix after weight update:")
    print("\ntheta = \n", model.w)
    
    
    # ## Train Network using gradient descent
    model = Network( n_features = k )
    lr = 0.1 #defining Learning Rate
    n_epochs=100 # defining Epochs of training
    model, epoch_loss, val_loss = train(model, n_epochs, lr, trainX, trainY, valX, valY) # Starting the training procedure
    plot_losses(epoch_loss,"train_loss_without_batch") #plotting all values of loss after all epochs on training dataset
    plot_losses(val_loss,"val_loss_without_batch") # plotting all values of loss after all epochs on validation dataset
    # ## Test prediction Prediction
    y_pred = model.forward(testX) # predicting the outcome of the model on Testing dataset.
    plot_predictions(testY, y_pred,"test_pred_without_batch") # plotting the predictions and acutal outcomes on graph for comparison purposes
    print("Loss at the end of training: ", epoch_loss[-1])
 

if __name__ == '__main__':
    main() # calling the main function
    



