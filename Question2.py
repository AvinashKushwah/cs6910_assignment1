from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np


# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# Define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_trainFlat = x_train.reshape(len(x_train),784).astype('float32')/255
#print(x_trainFlat.shape)
x_testFlat = x_test.reshape(len(x_test),784).astype('float32')/255
print(x_trainFlat.shape[0])
print(x_testFlat.shape)

# Activation Functions

def sigmoid(x):# limit the range of x to avoid overflow
  return 1.0 / (1.0 + np.exp(-x))
 
def sigmoid_diff(x):
  return (1 - sigmoid(x)) * sigmoid(x)

def softmax(Z):
  exp = np.exp(Z - np.max(Z))      #removing numerical instablity
  return exp / exp.sum(axis=0)

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_deriv(x):
    return x > 0



#Initalizing Weights and Biases

def Initalize_Wb():
  #layers = int(input("enter number of Hidden layers:"))
  #layers = int(layers)
  numberOfNeurons =[]               # number of neurons at each Hidden Layer
  W = []                            #store the Weight matrix at each layer
  b = []                            #store the bias vector at each layer
  for i in range(layers):                       
    randomNumber = np.random.randint(64,256)
    numberOfNeurons.append(randomNumber)
    if( i== 0):
      w_temp =np.random.rand(randomNumber,784)-0.5
      b_temp = np.random.rand(randomNumber,1)-0.5
      W.append(w_temp)
      b.append(b_temp)
    else:
      w_temp = np.random.rand(numberOfNeurons[i],numberOfNeurons[i-1])-0.5
      b_temp = np.random.rand(numberOfNeurons[i],1)-0.5
      W.append(w_temp);
      b.append(b_temp)

  print(numberOfNeurons)
  output_w = np.random.rand(10,numberOfNeurons[layers-1])-0.5
  output_b = np.random.rand(10,1)-0.5
  W.append(output_w)
  b.append(output_b)
  # print(len(W))
  # print(len(b))
  # print(W[0].shape)
  # print(W[1].shape)
  # print(W[2].shape)
  # print(b[0].shape)
  # print(b[1].shape)
  # print(b[2].shape)

  return W,b



#Feedforward Neural Network
def feedForward(input_data,W,b):
  
   output =[]
   temp =[]
   A =[]
  #  print("hello",b[0].shape)
  #  print(b[0].reshape(-1,1).shape)
   Y =np.dot(W[0],input_data) + b[0]
   temp.append(Y)
   Y= ReLU(Y)
   A.append(Y)
   for i in range(1,layers):
      Y =np.dot(W[i],Y) + b[i]
      temp.append(Y)
      Y=ReLU(Y)
      A.append(Y)
   temp2=np.dot(W[layers],Y) + b[layers]
   temp.append(temp2)
   output=(softmax(ReLU(temp2)))
   #print("softmax",output[:,1])
   return output,temp,A                            #output = Y_predicted , temp = PreActivation  A = postActivation


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
    # loss = -np.mean(np.sum(Y_actual * np.log(Y_predicted),axis =1))          #crossEntropy loss Caluclation
  return Y_actual


def predict(W, b, x):
    x = x.reshape(784, 1)
    A = x
    for i in range(len(W)-1):
        Z = np.dot(W[i], A) + b[i]
        A = np.maximum(Z, 0) # ReLU activation
    Z = np.dot(W[-1], A) + b[-1]
    y_pred = softmax(Z) # softmax activation
    return y_pred

layers = int(input("enter number of Hidden layers:"))
layers = int(layers)
Y_actual = find_Actuall_Y()
Y_actual = np.array(Y_actual)
Y_actual = Y_actual.T
W,b =Initalize_Wb()
Y_predicted, Intermediate_Activation, A = feedForward(x_trainFlat.T,W,b)

Index = int(input("Enter Index of an Image:"))
image = x_testFlat[Index]
image = np.reshape(image, (784, 1))
probablity = predict(W,b,image)
print(probablity)





