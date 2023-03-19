# cs6910_assignment1
Deep Learning Assignment


Code Execution Intruction for "train.py" File: To execute the "train.py" file, ensure that the following hyperparameter names are used:\
parameters: \
loss: options{"mse", "cross_entropy"}    NOTE: "mean_squared_error" will not work\
optimizer: options{"sgd", "nestrov", "adam", "nadam", "rmsprop"}     NOTE: "nag" will not work\
dataset: options{"mnist", "fashion_mnist"}\
rest all parameter names are same...\

Code Execution Intruction for Question 1: In order to run the code, you will need to modify the project name specified in the following line:\ wandb.init(project="dl_assignment"). Additionally, you will need to provide your API key in order to successfully execute the code.\

Code Execution Intruction for Question 2: In order to execute this code, you will need to provide the following inputs: the number of hidden layers, the number of neurons\ within each hidden layer, and an index of image from the test data set (which ranges from 0 to 9999). The code will then calculate the probability associated with the\ specified test data index.\

Code Execution Intruction for Question 3,4,5,6: In order to run the code, you will need to modify the project name specified in the following line:\ sweep_id = wandb.sweep(sweep_config, project="dl_assignment"). Additionally, you will need to provide your API key in order to successfully execute the code.\


Code Execution Intruction for  Question 7: To generate a confusion matrix, only the best performing hyperparameters that provide the highest accuracy have been used in sweep.\
To execute this code, you simply need to modify the following line of code: run=wandb.init(project="dl_assignment",entity="cs22m024",reinit='true') by replacing the "project\ name" and "entity name" placeholders with your desired project and entity names respectively. Additionally, ensure that you provide your API Key to authenticate and access the\ required resources.


Code Execution Intruction for Question 8: There are two files available, namely "dl_question_8_mse" which generates graphs for the mean square error loss function, and\ "dl_question_8_cross_entropy" which generates graphs for the cross-entropy loss function.Code execution Instruction: To run this,simply modify the "project name" with your own\ project name in the following line of code: sweep_id = wandb.sweep(sweep_config, project="dl_assignment") and provide your API Key.\
