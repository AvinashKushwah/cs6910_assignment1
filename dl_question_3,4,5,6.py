# -*- coding: utf-8 -*-


from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

!pip install wandb
import wandb
!wandb login

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(y_train.shape)

x_train, x_validate, y_train, y_validate = train_test_split( x_train, y_train, test_size=0.1, random_state=42)
# Define the class names

# Define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize the image pixel intesity within a range of 0-1
x_trainFlat = x_train.reshape(len(x_train),784).astype('float32')/255
print(x_trainFlat.shape)

x_validateFlat = x_validate.reshape(len(x_validate),784).astype('float32')/255
print(x_trainFlat.shape)

x_testFlat = x_test.reshape(len(x_test),784).astype('float32')/255
print(x_testFlat.shape)


#activation Functions

def ReLU(z):
    return np.maximum(0, z)

def ReLU_deriv(z):
    return np.where(z > 0, 1, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_diff(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.square(np.tanh(z))

def softmax(z):
   # print(z[0].shape)
    shift_z = z - np.max(z, axis=0, keepdims=True)  # avoiding overflow issue
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)  # avoiding underflow issue



np.random.seed(50)
def Initalize_Wb(layers, numberOfNeurons):
    W = []
    b = []
    for i in range(layers):
        if i == 0:
            # Input layer
            w_temp = np.random.randn(numberOfNeurons[i], 784) * np.sqrt(2/784)
            b_temp = np.zeros((numberOfNeurons[i], 1))
        else:
            # Hidden layers
            w_temp = np.random.randn(numberOfNeurons[i], numberOfNeurons[i-1])*np.sqrt(2/numberOfNeurons[i-1])
            b_temp = np.zeros((numberOfNeurons[i], 1))
        W.append(w_temp)
        b.append(b_temp)
    # Output layer
    output_w = np.random.randn(10, numberOfNeurons[-1]) * np.sqrt(2/numberOfNeurons[-1])
    output_b = np.zeros((10, 1))
    W.append(output_w)
    b.append(output_b)
    return W, b





def xavier_init(layers, numberOfNeurons):
    W = []
    b = []
    for i in range(layers):
        if i == 0:
            # Input layer
            w_temp = np.random.randn(numberOfNeurons[i], 784) * np.sqrt(1/784)
            b_temp = np.zeros((numberOfNeurons[i], 1))
        else:
            # Hidden layers
            w_temp = np.random.randn(numberOfNeurons[i], numberOfNeurons[i-1]) * np.sqrt(1/numberOfNeurons[i-1])
            b_temp = np.zeros((numberOfNeurons[i], 1))
        W.append(w_temp)
        b.append(b_temp)
    # Output layer
    output_w = np.random.randn(10, numberOfNeurons[-1]) * np.sqrt(1/numberOfNeurons[-1])
    output_b = np.zeros((10, 1))
    W.append(output_w)
    b.append(output_b)
    return W, b


def feedForward(input_data,W,b,layers,activation):
  
   output =[]
   temp =[]
   A =[]
  #  print("hello",b[0].shape)
  #  print(b[0].reshape(-1,1).shape)
   Y =np.dot(W[0],input_data) + b[0]
   temp.append(Y)
   if(activation == 'ReLU'):
      Y=ReLU(Y)
   elif(activation == 'sigmoid'):
      Y=sigmoid(Y)
   else:
      Y = tanh(Y)
        
   A.append(Y)
   for i in range(1,layers):
      Y =np.dot(W[i],Y) + b[i]
      temp.append(Y)
      if(activation == 'ReLU'):
        Y=ReLU(Y)
        A.append(Y)
      elif(activation == 'sigmoid'):
        Y=sigmoid(Y)
        A.append(Y)
      else:
        Y = tanh(Y)
        A.append(Y)

   temp2=np.dot(W[layers],Y) + b[layers]
   temp.append(temp2)
  #  if(activation == 'ReLU'):
  #    output=(softmax(temp2))
  #  elif(activation == 'sigmoid'):
  #    output=(softmax(sigmoid(temp2)))
  #  else:
  #    output=(softmax(tanh(temp2)))
   output=softmax(temp2)
   #print("softmaxt",output[:,1])
   return output,temp,A


def find_Actuall_Y():
  X,Y = x_trainFlat.shape
  # print(X)
  Y_actual=[]
  for i in range(X):
    tempX =x_trainFlat[i]
    tempY = y_train[i]
    vector = np.zeros(10)             # creating a vector of size 10 and assigning '1' to the correct index of trainFlat[i]
    vector[tempY] =1
    Y_actual.append(vector)
    # #outputY=softmax(feedForward(x_trainFlat[i]))
    # Y_predicted.append(outputY)
    # loss = -np.mean(np.sum(Y_actual * np.log(Y_predicted),axis =1))    
          #crossEntropy loss Caluclation
  
  return Y_actual


Y_actual = find_Actuall_Y()
Y_actual = np.array(Y_actual)
Y_actual = Y_actual.T
print(Y_actual.shape)


def find(x_flat,y_train):
  X,Y = x_flat.shape
  Y_actual =[]
  for i in range(X):
    tempX =x_flat[i]
    tempY = y_train[i]
    vector = np.zeros(10)             # creating a vector of size 10 and assigning '1' to the correct index of trainFlat[i]
    vector[tempY] =1
    Y_actual.append(vector)
  Y_actual = np.array(Y_actual)
  Y_actual = Y_actual.T
  return Y_actual


def prediction_Test(input_data,W,b,num_layers,activation):
  Y_predicted, _, _ = feedForward(input_data.T, W, b,num_layers,activation)
  predictions = get_predictions(Y_predicted)
  accuracy = get_accuracy(predictions, y_test) * 100
  print(f"Test_Accuracy = {accuracy:.2f}%")




def calculateWsquare(W):
  W_norm = 0.0
  for w in W:
      W_norm += np.sum(w ** 2)
  return W_norm

  

def loss_func(Y_actual, Y_predicted, loss_function, weight_decay, W):
    if loss_function == 'mse':
        N = Y_actual.shape[1] # number of samples
        loss = 1/(2*N) * np.sum((Y_actual - Y_predicted)**2)
        loss = loss + weight_decay/2 * calculateWsquare(W)
        loss = np.clip(loss,-5,7)
    elif loss_function == 'cross_entropy':
       # Y_predicted = np.clip(Y_predicted, 1e-9, 1 - 1e-9)
        N = Y_actual.shape[1] # number of samples
        loss = -1/N * np.sum(Y_actual * np.log(Y_predicted))
        loss = loss + weight_decay/2 * calculateWsquare(W)
        loss = np.clip(loss,-5,7)
    else:
        print("Invalid error")
    return loss


#Backpropogation Function

def backpropagation(X, Y_actual, Y_predicted, Intermediate_Activation,A,W,b,activation,weight_decay):
    dW = [0] * (len(W)) # gradient of weight matrix at each layer
    db = [0] * (len(b)) # gradient of bias vector at each layer
    
   
    dL_dZ = Y_predicted - Y_actual
    
   
   #dW[-1] = 1/X.shape[0] * np.dot(dL_dZ, Intermediate_Activation[-1].T)
    dW[-1] = 1/X.shape[0] * np.dot(dL_dZ, A[-1].T)
    db[-1] = 1/X.shape[0] * np.sum(dL_dZ, axis=1, keepdims=True)
   # print(dL_dZ.shape)
    #print(db[-1].shape)
    
    # Backpropagate in hidden layers:
    for l in reversed(range(1, len(W)-1)):
        dL_dY = np.dot(W[l+1].T, dL_dZ)
        # print(dL_dY.shape)
        # print(Intermediate_Activation[l].shape)
        # zy= sigmoid_diff(Intermediate_Activation[l-1])
        # print(zy.shape)
        dl_dz=[]
        if(activation == 'ReLU'):
          dL_dZ = dL_dY * ReLU_deriv(Intermediate_Activation[l])
        elif(activation == 'sigmoid'):
          dL_dZ = dL_dY * sigmoid_diff(Intermediate_Activation[l])
        else:
          dL_dZ = dL_dY * tanh_deriv(Intermediate_Activation[l])
        # dL_dZ = dL_dY * ReLU_deriv(Intermediate_Activation[l])
        dW[l] = 1/X.shape[0] * np.dot(dL_dZ, Intermediate_Activation[l-1].T) +  weight_decay * W[l]                  #hell01
        db[l] = 1/X.shape[0] * np.sum(dL_dZ, axis=1, keepdims=True) 

    
    # Compute gradients for input layer
    dL_dY = np.dot(W[1].T, dL_dZ)
    #dL_dZ = dL_dY * ReLU_deriv(Intermediate_Activation[0])
    if(activation == 'ReLU'):
      dL_dZ = dL_dY * ReLU_deriv(Intermediate_Activation[0])
    elif(activation == 'sigmoid'):
      dL_dZ = dL_dY * sigmoid_diff(Intermediate_Activation[0])
    else:
      dL_dZ = dL_dY * tanh_deriv(Intermediate_Activation[0])
    dW[0] = 1/X.shape[0] * np.dot(dL_dZ, X)
    db[0] = 1/X.shape[0] * np.sum(dL_dZ, axis=1, keepdims=True)
    # print("insside update",len(db))
    # print(db[0].shape)
    # print(db[1].shape)
    # print(db[2].shape)
    return dW, db


def Update(W,b,dW,db,learningRate):

  for i in range (len(W)):
    W[i] = W[i] -learningRate*dW[i]
   # print("before update",b[i].shape,db[i].shape)
    b[i] = b[i] -learningRate* db[i]
    #print("after update",b[i].shape, W[i].shape)
  
  return W,b


#W,b=Update(W,b,dW,db,0.1)

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


#optimization algorithms:
def stochastic_gradient_descent(x_trainFlat, Y_actual, y_train,epochs,learning_rate, batch_size,W,b,numberOfNeurons,num_layers,activation,loss_function,weight_decay):
   # W, b = Initalize_Wb()
    
    numberofBatch=len(x_trainFlat)/batch_size
    for i in range(epochs+1):
        # Randomly shuffle the training data and split it into batches
        idx = np.random.permutation(len(x_trainFlat))
        x_trainFlat_shuffled = x_trainFlat[idx]
        Y_actual_shuffled = Y_actual[:, idx]
        y_train_shuffled = np.array(y_train)[idx]
        for j in range(0, len(x_trainFlat), batch_size):
            # Select a batch of training examples
            x_batch = x_trainFlat_shuffled[j:j+batch_size]
            Y_actual_batch = Y_actual_shuffled[:, j:j+batch_size]
            y_train_batch = y_train_shuffled[j:j+batch_size]
            # Compute the forward pass and backpropagation for the batch
            Y_predicted_batch, Intermediate_Activation_batch, A_batch = feedForward(x_batch.T, W, b,num_layers,activation)
            dW_batch, db_batch = backpropagation(x_batch, Y_actual_batch, Y_predicted_batch, Intermediate_Activation_batch, A_batch, W, b,activation,weight_decay)
            # Update the parameters using the batch gradient
            W, b = Update(W, b, dW_batch, db_batch, learning_rate)
      
            # Compute the accuracy on the entire training set
        Y_predicted, _, _ = feedForward(x_trainFlat.T, W, b,num_layers,activation)
        predictions = get_predictions(Y_predicted)
        accuracy = get_accuracy(predictions, y_train) *100
        print(f"Iteration {i}: Train accuracy = {accuracy:.2f}")

        Y_actual_train =find(x_trainFlat,y_train)
        train_loss = loss_func(Y_actual_train, Y_predicted,loss_function,weight_decay,W)              
        print("train_loss:",train_loss)

        Y_predicted_validate, _, _ = feedForward(x_validateFlat.T, W, b,num_layers,activation)
        predictions_validate = get_predictions(Y_predicted_validate)
        accuracy_val = get_accuracy(predictions_validate, y_validate) *100
        print(f"Iteration {i}: Validation accuracy = {accuracy_val:.2f}")
        
        
        Y_actual_validate = find(x_validateFlat,y_validate)
        validation_loss = loss_func(Y_actual_validate,Y_predicted_validate,loss_function,weight_decay,W)
        print("validation_loss:",validation_loss)

        

        wandb.log({"validation_accuracy": accuracy_val, "train_accuracy": accuracy,"train_loss": train_loss, "val_loss":validation_loss})
      

    return W, b


#W,b = stochastic_gradient_descent(x_trainFlat,Y_actual,y_train,100,0.1,64)


def momentum_gradient_descent(x_trainFlat, Y_actual, y_train, epochs, learning_rate, batch_size, momentum,W,b,numberOfNeurons,num_layers,activation,loss_function,weight_decay):
    #W, b = Initalize_Wb()
    # initialize velocities to zero
    v_dW = [np.zeros_like(w) for w in W]
    v_db = [np.zeros_like(b) for b in b]
    
    numberofBatch = len(x_trainFlat)/batch_size
    # print(len(v_dW))
    # print(len(v_db))
    
    for i in range(epochs+1):
        # Randomly shuffle the training data and split it into batches
        idx = np.random.permutation(len(x_trainFlat))
        x_trainFlat_shuffled = x_trainFlat[idx]
        Y_actual_shuffled = Y_actual[:, idx]
        y_train_shuffled = np.array(y_train)[idx]
        
        for j in range(0, len(x_trainFlat), batch_size):
            # Select a batch of training examples
            x_batch = x_trainFlat_shuffled[j:j+batch_size]
            Y_actual_batch = Y_actual_shuffled[:, j:j+batch_size]
            y_train_batch = y_train_shuffled[j:j+batch_size]
            
            # Compute the forward pass and backpropagation for the batch
            Y_predicted_batch, Intermediate_Activation_batch, A_batch = feedForward(x_batch.T, W, b,num_layers,activation)
            dW_batch, db_batch = backpropagation(x_batch, Y_actual_batch, Y_predicted_batch, Intermediate_Activation_batch, A_batch, W, b,activation,weight_decay)
            
            # Update velocities
            v_dW = [momentum * v_dw + (1-momentum) * dw for v_dw, dw in zip(v_dW, dW_batch)]
            v_db = [momentum * v_db + (1-momentum) * db for v_db, db in zip(v_db, db_batch)]
            
            # Update the parameters using the velocity
            W, b = Update(W, b, v_dW, v_db, learning_rate)
            
        # if i % 5 == 0:
        #     # Compute the accuracy on the entire training set
        #     Y_predicted, _, _ = feedForward(x_trainFlat.T, W, b,num_layers,activation)
        #     predictions = get_predictions(Y_predicted)
        #     accuracy = get_accuracy(predictions, y_train) * 100
        #     print(f"Iteration {i}: accuracy = {accuracy:.2f}%")
        
        Y_predicted, _, _ = feedForward(x_trainFlat.T, W, b,num_layers,activation)
        predictions = get_predictions(Y_predicted)
        accuracy = get_accuracy(predictions, y_train) *100
        print(f"Iteration {i}: accuracy = {accuracy:.2f}")

        #wandb.log({"train_accuracy":accuracy} )
        Y_predicted_validate, _, _ = feedForward(x_validateFlat.T, W, b,num_layers,activation)
        predictions_validate = get_predictions(Y_predicted_validate)
        accuracy_val = get_accuracy(predictions_validate, y_validate) *100
        print(f"Iteration {i}: accuracy = {accuracy_val:.2f}")

        Y_actual_train =find(x_trainFlat,y_train)
        
        train_loss = loss_func(Y_actual_train, Y_predicted,loss_function,weight_decay,W)           
        print("train_loss:",train_loss)
        Y_actual_validate = find(x_validateFlat,y_validate)
        validation_loss = loss_func(Y_actual_validate,Y_predicted_validate,loss_function,weight_decay,W)
        print("validation_loss:",validation_loss)
        wandb.log({"validation_accuracy": accuracy_val, "train_accuracy": accuracy,"train_loss": train_loss, "val_loss":validation_loss})
        #wandb.log({"validation_accuracy":accuracy_val} )
       # wandb.log({"validation_accuracy": accuracy_val, "train_accuracy": accuracy})
    return W, b
#W, b = momentum_gradient_descent(x_trainFlat, Y_actual, y_train, iteration=1000, learning_rate=0.1,batch_size=64, momentum=0.9)

def nesterov_gradient_descent(x_trainFlat, Y_actual, y_train,epochs,learning_rate,batch_size,momentum,W,b,numberOfNeurons,num_layers,activation,loss_function,weight_decay):
    #W, b = Initalize_Wb()
    # initialize velocities to zero
    v_dW = [np.zeros_like(w) for w in W]
    v_db = [np.zeros_like(b) for b in b]
    numberofBatch = len(x_trainFlat)/batch_size
    # print(len(v_dW))
    # print(len(v_db))
    
    for i in range(epochs+1):
        # Randomly shuffle the training data and split it into batches
        idx = np.random.permutation(len(x_trainFlat))
        x_trainFlat_shuffled = x_trainFlat[idx]
        Y_actual_shuffled = Y_actual[:, idx]
        y_train_shuffled = np.array(y_train)[idx]
        
        for j in range(0, len(x_trainFlat), batch_size):
            # Select a batch of training examples
            x_batch = x_trainFlat_shuffled[j:j+batch_size]
            Y_actual_batch = Y_actual_shuffled[:, j:j+batch_size]
            y_train_batch = y_train_shuffled[j:j+batch_size]
            
            # Compute the forward pass and backpropagation for the batch
            #Y_predicted_batch, Intermediate_Activation_batch, A_batch = feedForward(x_batch.T, W, b,num_layers,activation)

            # Compute the gradients at the lookahead position
            #for all layers using loop
            W_lookahead = [w - momentum * v_dw for w, v_dw in zip(W, v_dW)]
            b_lookahead = [b1 - momentum * j for b1, j in zip(b, v_db)]
            Y_predicted_batch, Intermediate_Activation_batch, A_batch= feedForward(x_batch.T, W_lookahead, b_lookahead,num_layers,activation)
            dW_batch_lookahead, db_batch_lookahead = backpropagation(x_batch, Y_actual_batch, Y_predicted_batch, Intermediate_Activation_batch, A_batch, W_lookahead, b_lookahead,activation,weight_decay)
            
            # Update velocities
            v_dW = [momentum * v_dw + (1-momentum) * dw for v_dw, dw in zip(v_dW, dW_batch_lookahead)]
            v_db = [momentum * j + (1-momentum) * db for j, db in zip(v_db, db_batch_lookahead)]
            
            # Update the parameters using the velocity
            W, b = Update(W, b, v_dW, v_db, learning_rate)
            
        # if i % 5 == 0:
        #     # Compute the accuracy on the entire training set
        #     Y_predicted, _, _ = feedForward(x_trainFlat.T, W, b,num_layers,activation)
        #     predictions = get_predictions(Y_predicted)
        #     accuracy = get_accuracy(predictions, y_train) * 100
        #     print(f"Iteration {i}: accuracy = {accuracy:.2f}%")
        Y_predicted, _, _ = feedForward(x_trainFlat.T, W, b,num_layers,activation)
        predictions = get_predictions(Y_predicted)
        accuracy = get_accuracy(predictions, y_train) *100
        print(f"Iteration {i}: Train accuracy = {accuracy:.2f}")
        # wandb.log({"train_accuracy":accuracy} )
        Y_predicted_validate, _, _ = feedForward(x_validateFlat.T, W, b,num_layers,activation)
        predictions_validate = get_predictions(Y_predicted_validate)
        accuracy_val = get_accuracy(predictions_validate, y_validate) *100
        print(f"Iteration {i}: Validation accuracy = {accuracy_val:.2f}")

        Y_actual_train =find(x_trainFlat,y_train)
       
        train_loss = loss_func(Y_actual_train, Y_predicted,loss_function,weight_decay,W)        
        print("train_loss:",train_loss)
        Y_actual_validate = find(x_validateFlat,y_validate)
        validation_loss = loss_func(Y_actual_validate,Y_predicted_validate,loss_function,weight_decay,W)
        print("validation_loss:",validation_loss)
        
        wandb.log({"validation_accuracy": accuracy_val, "train_accuracy": accuracy,"train_loss": train_loss, "val_loss":validation_loss})
        #wandb.log({"validation_accuracy":accuracy_val} )
       # wandb.log({'validation_accuracy': accuracy_val, 'train_accuracy': accuracy})
    return W, b

def rmsprop(x_trainFlat, Y_actual, y_train,epochs,learning_rate,batch_size,momentum,W,b,numberOfNeurons,num_layers,activation,beta,beta1,beta2,epsilon,loss_function,weight_decay):
   #W, b = Initalize_Wb()
    # initialize accumulated gradients to zero
    s_dW = [np.zeros_like(w) for w in W]
    s_db = [np.zeros_like(b) for b in b]
    numberofBatch=len(x_trainFlat)/batch_size 
    # set a small constant to avoid division by zero
    #epsilon = 1e-8
    
    for i in range(epochs+1):
        # Randomly shuffle the training data and split it into batches
        idx = np.random.permutation(len(x_trainFlat))
        x_trainFlat_shuffled = x_trainFlat[idx]
        Y_actual_shuffled = Y_actual[:, idx]
        y_train_shuffled = np.array(y_train)[idx]
        
        for j in range(0, len(x_trainFlat), batch_size):
            # Select a batch of training examples
            x_batch = x_trainFlat_shuffled[j:j+batch_size]
            Y_actual_batch = Y_actual_shuffled[:, j:j+batch_size]
            y_train_batch = y_train_shuffled[j:j+batch_size]
            
            # Compute the forward pass and backpropagation for the batch
            Y_predicted_batch, Intermediate_Activation_batch, A_batch = feedForward(x_batch.T, W, b,num_layers,activation)
            dW_batch, db_batch = backpropagation(x_batch, Y_actual_batch, Y_predicted_batch, Intermediate_Activation_batch, A_batch, W, b,activation,weight_decay)
            
            # Accumulate the squared gradients
            s_dW = [beta * s_dw + (1-beta) * dw**2 for s_dw, dw in zip(s_dW, dW_batch)]
            s_db = [beta * s_db1 + (1-beta) * j**2 for s_db1, j in zip(s_db, db_batch)]
            
            # Update the parameters using the accumulated gradients
            W = [w - learning_rate * dw / np.sqrt(s_dw + epsilon) for w, dw, s_dw in zip(W, dW_batch, s_dW)]
            b = [b1 - learning_rate * db / np.sqrt(s_db1 + epsilon) for b1, db, s_db1 in zip(b, db_batch, s_db)]
            
        
            # Compute the accuracy on the entire training set
        Y_predicted, _, _ = feedForward(x_trainFlat.T, W, b,num_layers,activation)
        predictions = get_predictions(Y_predicted)
        accuracy = get_accuracy(predictions, y_train) *100
        print(f"Iteration {i}:Train accuracy = {accuracy:.2f}")

        #wandb.log({"train_accuracy":accuracy} )
        Y_predicted_validate, _, _ = feedForward(x_validateFlat.T, W, b,num_layers,activation)
        predictions_validate = get_predictions(Y_predicted_validate)
        accuracy_val = get_accuracy(predictions_validate, y_validate) *100
        print(f"Iteration {i}: Validation accuracy = {accuracy_val:.2f}")

        Y_actual_train =find(x_trainFlat,y_train)
        
        train_loss = loss_func(Y_actual_train, Y_predicted,loss_function,weight_decay,W)            
        print("train_loss:",train_loss)
        Y_actual_validate = find(x_validateFlat,y_validate)
        validation_loss = loss_func(Y_actual_validate,Y_predicted_validate,loss_function,weight_decay,W)
        print("validation_loss:",validation_loss)

        wandb.log({"validation_accuracy": accuracy_val, "train_accuracy": accuracy,"train_loss":train_loss ,"val_loss":validation_loss})
        #wandb.log({"validation_accuracy":accuracy_val} )
       # wandb.log({"validation_accuracy": accuracy_val, "train_accuracy": accuracy})
    
    return W, b

def adam(x_trainFlat, Y_actual, y_train,epochs, learning_rate, batch_size, beta1, beta2, epsilon,W,b,numberOfNeurons,num_layers,activation,loss_function,weight_decay):
    #W, b = Initalize_Wb()
    # initialize velocities and squared gradients to zero
    v_dW = [np.zeros_like(w) for w in W]
    v_db = [np.zeros_like(b) for b in b]
    s_dW = [np.zeros_like(w) for w in W]
    s_db = [np.zeros_like(b) for b in b]
    
    numberofBatch = len(x_trainFlat)/batch_size
    for i in range(epochs+1):
        # Randomly shuffle the training data and split it into batches
        idx = np.random.permutation(len(x_trainFlat))
        x_trainFlat_shuffled = x_trainFlat[idx]
        Y_actual_shuffled = Y_actual[:, idx]
        y_train_shuffled = np.array(y_train)[idx]

        for j in range(0, len(x_trainFlat), batch_size):
            # Select a batch of training examples
            x_batch = x_trainFlat_shuffled[j:j+batch_size]
            Y_actual_batch = Y_actual_shuffled[:, j:j+batch_size]
            y_train_batch = y_train_shuffled[j:j+batch_size]

            # Compute the forward pass and backpropagation for the batch
            Y_predicted_batch, Intermediate_Activation_batch, A_batch = feedForward(x_batch.T, W, b,num_layers,activation)
            dW_batch, db_batch = backpropagation(x_batch, Y_actual_batch, Y_predicted_batch, Intermediate_Activation_batch, A_batch, W, b,activation,weight_decay)

            # Update velocities and squared gradients
            v_dW = [beta1 * v_dw + (1 - beta1) * dw for v_dw, dw in zip(v_dW, dW_batch)]
            v_db = [beta1 * v_db + (1 - beta1) * db for v_db, db in zip(v_db, db_batch)]
            s_dW = [beta2 * s_dw + (1 - beta2) * np.square(dw) for s_dw, dw in zip(s_dW, dW_batch)]
            s_db = [beta2 * s_db1 + (1 - beta2) * np.square(db) for s_db1, db in zip(s_db, db_batch)]

            # Bias correction for velocities and squared gradients
            v_dW_corrected = [v_dw / (1 - beta1**(i+1)) for i, v_dw in enumerate(v_dW)]
            v_db_corrected = [v_db1 / (1 - beta1**(i+1)) for i, v_db1 in enumerate(v_db)]
            s_dW_corrected = [s_dw / (1 - beta2**(i+1)) for i, s_dw in enumerate(s_dW)]
            s_db_corrected = [s_db1 / (1 - beta2**(i+1)) for i, s_db1 in enumerate(s_db)]

            # Update the parameters
            W, b = Update(W, b, [(learning_rate * v_dW_corrected[i]) / (np.sqrt(s_dW_corrected[i]) + epsilon) for i in range(len(W))], [(learning_rate * v_db_corrected[i]) / (np.sqrt(s_db_corrected[i]) + epsilon) for i in range(len(b))],learning_rate)

        Y_predicted, _, _ = feedForward(x_trainFlat.T, W, b,num_layers,activation)
        predictions = get_predictions(Y_predicted)
        accuracy = get_accuracy(predictions, y_train) *100
        print(f"Iteration {i}: Train accuracy = {accuracy:.2f}")
       # wandb.log({"train_accuracy":accuracy} )
        Y_predicted_validate, _, _ = feedForward(x_validateFlat.T, W, b,num_layers,activation)
        predictions_validate = get_predictions(Y_predicted_validate)
        accuracy_val = get_accuracy(predictions_validate, y_validate) *100
        print(f"Iteration {i}:Validation accuracy = {accuracy_val:.2f}")

        Y_actual_train =find(x_trainFlat,y_train)
       
        train_loss = loss_func(Y_actual_train, Y_predicted,loss_function,weight_decay,W)   
        print("train_loss:",train_loss)
        Y_actual_validate = find(x_validateFlat,y_validate)
        validation_loss = loss_func(Y_actual_validate,Y_predicted_validate,loss_function,weight_decay,W)   
        print("validation_loss:",validation_loss)
        
        wandb.log({"validation_accuracy": accuracy_val, "train_accuracy": accuracy,"train_loss": train_loss, "val_loss":validation_loss})

        #wandb.log({"validation_accuracy":accuracy_val} )
       # wandb.log({"validation_accuracy": accuracy_val, "train_accuracy": accuracy})
    
    return W, b

def nadam(x_trainFlat, Y_actual, y_train,epochs, learning_rate, batch_size, beta1, beta2, epsilon,W,b,numberOfNeurons,num_layers,activation,loss_function,weight_decay):
   # W, b = Initalize_Wb()
    # initialize velocities and squared gradients to zero
    v_dW = [np.zeros_like(w) for w in W]
    v_db = [np.zeros_like(b) for b in b]
    s_dW = [np.zeros_like(w) for w in W]
    s_db = [np.zeros_like(b) for b in b]

    # initialize the bias correction terms for velocities
    m_dW = [np.zeros_like(w) for w in W]
    m_db = [np.zeros_like(b) for b in b]
    numberofBatch = len(x_trainFlat)/batch_size
    for i in range(epochs):
        # Randomly shuffle the training data and split it into batches
        idx = np.random.permutation(len(x_trainFlat))
        x_trainFlat_shuffled = x_trainFlat[idx]
        Y_actual_shuffled = Y_actual[:, idx]
        y_train_shuffled = np.array(y_train)[idx]

        for j in range(0, len(x_trainFlat), batch_size):
            # Select a batch of training examples
            x_batch = x_trainFlat_shuffled[j:j+batch_size]
            Y_actual_batch = Y_actual_shuffled[:, j:j+batch_size]
            y_train_batch = y_train_shuffled[j:j+batch_size]

            # Compute the forward pass and backpropagation for the batch
            Y_predicted_batch, Intermediate_Activation_batch, A_batch = feedForward(x_batch.T, W, b,num_layers,activation)
            dW_batch, db_batch = backpropagation(x_batch, Y_actual_batch, Y_predicted_batch, Intermediate_Activation_batch, A_batch, W, b,activation,weight_decay)

            # Update velocities and squared gradients
            v_dW = [beta1 * v_dw + (1 - beta1) * dw for v_dw, dw in zip(v_dW, dW_batch)]
            v_db = [beta1 * v_db1 + (1 - beta1) * db for v_db1, db in zip(v_db, db_batch)]
            s_dW = [beta2 * s_dw + (1 - beta2) * np.square(dw) for s_dw, dw in zip(s_dW, dW_batch)]
            s_db = [beta2 * s_db1 + (1 - beta2) * np.square(db) for s_db1, db in zip(s_db, db_batch)]

            # Bias correction for velocities and squared gradients
            v_dW_corrected = [v_dw / (1 - beta1**(i+1)) for i, v_dw in enumerate(v_dW)]
            v_db_corrected = [v_db1 / (1 - beta1**(i+1)) for i, v_db1 in enumerate(v_db)]
            s_dW_corrected = [s_dw / (1 - beta2**(i+1)) for i, s_dw in enumerate(s_dW)]
            s_db_corrected = [s_db1 / (1 - beta2**(i+1)) for i, s_db1 in enumerate(s_db)]

            # Compute the Nesterov momentum update
            m_dW = [(beta1 * v_dW_corrected[i] + (1 - beta1) * dw) / (1 - beta1**(i+1)) for i, dw in enumerate(dW_batch)]
            m_db = [(beta1 * v_db_corrected[i] + (1 - beta1) * db1) / (1 - beta1**(i+1)) for i, db1 in enumerate(db_batch)]

          
            m_dW_corrected = [(beta1 * v_dW_corrected[i] + (1 - beta1) * dw) / (1 - beta1**(i+1)) for i, dw in enumerate(m_dW)]
            m_db_corrected = [(beta1 * v_db_corrected[i] + (1 - beta1) * db) / (1 - beta1**(i+1)) for i, db in enumerate(m_db)]
            v_dW_corrected = [(beta2 * v_dW_corrected[i] + (1 - beta2) * np.square(dw)) for i, dw in enumerate(dW_batch)]
            v_db_corrected = [(beta2 * v_db_corrected[i] + (1 - beta2) * np.square(db)) for i, db in enumerate(db_batch)]
            v_dW_corrected = [v_dW_corrected[i] / (1 - beta2**(i+1)) for i in range(len(v_dW_corrected))]
            v_db_corrected = [v_db_corrected[i] / (1 - beta2**(i+1)) for i in range(len(v_db_corrected))]

            W, b = Update(W, b, [(learning_rate * v_dW_corrected[i]) / (np.sqrt(s_dW_corrected[i]) + epsilon) for i in range(len(W))], [(learning_rate * v_db_corrected[i]) / (np.sqrt(s_db_corrected[i]) + epsilon) for i in range(len(b))],learning_rate)


        Y_predicted, _, _ = feedForward(x_trainFlat.T, W, b,num_layers,activation)
        predictions = get_predictions(Y_predicted)
        accuracy = get_accuracy(predictions, y_train) *100
        print(f"Iteration {i}: Train accuracy = {accuracy:.2f}")
       # wandb.log({"train_accuracy":accuracy} )
        Y_predicted_validate, _, _ = feedForward(x_validateFlat.T, W, b,num_layers,activation)
        predictions_validate = get_predictions(Y_predicted_validate)
        accuracy_val = get_accuracy(predictions_validate, y_validate) *100
        print(f"Iteration {i}:Validation accuracy = {accuracy_val:.2f}")

        Y_actual_train =find(x_trainFlat,y_train)
        train_loss = loss_func(Y_actual_train, Y_predicted,loss_function,weight_decay,W)            
        print("train_loss:",train_loss)
        Y_actual_validate = find(x_validateFlat,y_validate)
        validation_loss = loss_func(Y_actual_validate,Y_predicted_validate,loss_function,weight_decay,W)
        print("validation_loss:",validation_loss)
        wandb.log({"validation_accuracy": accuracy_val, "train_accuracy": accuracy,"train_loss": train_loss, "val_loss":validation_loss})

    return W,b


def neuralNetwork(epochs,batch_size,optimizer,learning_rate,beta,beta1,beta2,epsilon,weight_init,num_layers,hidden_size,activation,momentum,loss_function,weight_decay):
      

      numberOfNeurons =[] 
      for i in range(num_layers):
        temp = hidden_size
        numberOfNeurons.append(temp)
      W=[]
      b=[]
      if weight_init == 'random':
        W,b = Initalize_Wb(num_layers,numberOfNeurons)
      else:
        W,b = xavier_init(num_layers, numberOfNeurons)

      if optimizer == 'GD':
        W,b =gradient_descent(epochs,x_trainFlat,Y_actual, y_train,learning_rate,numberOfNeurons,num_layers,activation,W,b,weight_decay)
        prediction_Test(x_testFlat,W,b,num_layers,activation)
      elif optimizer =='sgd':      
        W,b =stochastic_gradient_descent(x_trainFlat, Y_actual, y_train,epochs,learning_rate, batch_size,W,b,numberOfNeurons,num_layers,activation,loss_function,weight_decay)
        prediction_Test(x_testFlat,W,b,num_layers,activation)
      elif optimizer == 'momentum':
        W,b = momentum_gradient_descent(x_trainFlat, Y_actual, y_train,epochs,learning_rate,batch_size,momentum,W,b,numberOfNeurons,num_layers,activation,loss_function,weight_decay)
        prediction_Test(x_testFlat,W,b,num_layers,activation)
      elif optimizer == 'nestrov':
        W,b = nesterov_gradient_descent(x_trainFlat, Y_actual, y_train,epochs,learning_rate,batch_size,momentum,W,b,numberOfNeurons,num_layers,activation,loss_function,weight_decay)
        prediction_Test(x_testFlat,W,b,num_layers,activation)
      elif optimizer == 'rmsprop':
        W,b = rmsprop(x_trainFlat, Y_actual, y_train,epochs, learning_rate, batch_size,momentum,W,b,numberOfNeurons,num_layers,activation,beta,beta1,beta2,epsilon,loss_function,weight_decay)
        prediction_Test(x_testFlat,W,b,num_layers,activation)
      elif optimizer == 'adam':
        W,b = adam(x_trainFlat, Y_actual, y_train,epochs,learning_rate,batch_size,beta1,beta2,epsilon,W,b,numberOfNeurons,num_layers,activation,loss_function,weight_decay)
        prediction_Test(x_testFlat,W,b,num_layers,activation)
      elif optimizer == 'nadam':
        W,b = nadam(x_trainFlat, Y_actual, y_train,epochs,learning_rate,batch_size,beta1,beta2,epsilon,W,b,numberOfNeurons,num_layers,activation,loss_function,weight_decay)
        prediction_Test(x_testFlat,W,b,num_layers,activation)
      

      
  

sweep_config={
    'method' : 'bayes' ,
    'metric' : { 'name' :'validation_accuracy' , 'goal' : 'maximize' } ,
    'parameters' : {
        'epochs' : { 'values' : [10,15,20] },
        'n_hidden_layers' : {'values' : [1,2,3]},
        'n_hidden_layer_size' : { 'values' : [32,64,128,256]},
        'batch_size' : { 'values' : [32,64,128]},
        'learning_rate' : { 'values' : [0.1,0.01, 0.002]},
        'optimizer' : { 'values' : ["sgd", "mgd","adam","nestrov","rmsprop","nadam"] },
        'activations' : { 'values' : ["sigmoid", "tanh", "ReLU"] },
        'loss_function' : {'values' : ['cross_entropy']},
        'weight_ini' : {'values' : ['random','xavier']},
        'weight_decay' : { 'values' : [0,0.001]},
        'beta': {'values' : [0.9,0.85]},
        'beta1': {'values' : [0.9,0.8]},
        'beta2': {'values' : [0.99,0.9]},
        'momentum': {'values' : [0.9,0.83]},
        'epsilon' : {'values' : [1e-7,1e-9]}

    }
}

def train_neural_network():
  config_default={
      'weight_ini':'random',
      'n_hidden_layers':2,
      'n_hidden_layer_size':64,
      'optimizer':'sgd',
      'learning_rate':0.1,
      'epoch':20,
      'batch_size':64,
      'activations':'ReLU',
      'loss_function' :'cross_entropy',
      'weight_decay':0


  }
  wandb.init(config=config_default)

  c= wandb.config
  name = "op_"+str(c.optimizer)+", ac"+str(c.activations)+", hl"+str(c.n_hidden_layers)+", hls"+str(c.n_hidden_layer_size)+", ep"+str(c.epochs)+", n"+str(c.learning_rate)+", bs"+str(c.batch_size)+"wi"+str(c.weight_ini)
  wandb.init(name=name)

  hn = c.n_hidden_layer_size
  hl = c.n_hidden_layers
  act = c.activations
  opt = c.optimizer
  ep = c.epochs
  bs = c.batch_size
  lr = c.learning_rate
  wi = c.weight_ini
  mm =c.momentum
  b =c.beta
  b1=c.beta1
  b2=c.beta2
  e =c.epsilon
  wd=c.weight_decay
  ls=c.loss_function
  neuralNetwork(ep,bs,opt, lr,b,b1,b2,e,wi,hl,hn,act,mm,ls,wd)
  return
sweep_id = wandb.sweep(sweep_config, project="dl_assignment")
wandb.agent(sweep_id, function=train_neural_network)

