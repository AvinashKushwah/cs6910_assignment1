# cs6910_assignment1
Deep Learning Assignment


Code Execution Intruction for "train.py" File: To execute the "train.py" file, ensure that the following hyperparameter names are used:
Supported parameters:     
loss: options{"mse", "cross_entropy"}    
optimizer: options{"sgd", "nestrov", "adam", "nadam", "rmsprop"}    
dataset: options{"mnist", "fashion_mnist"}

Question 1: 


Question 7: To generate a confusion matrix, only the best performing hyperparameters that provide the highest accuracy have been used in sweep.

Code execution Instruction : To execute this code, you simply need to modify the following line of code: run=wandb.init(project="dl_assignment",entity="cs22m024",reinit='true') by replacing the "project name" and "entity name" placeholders with your desired project and entity names respectively. Additionally, ensure that you provide your API Key to authenticate and access the required resources.


Question 8: There are two files available, namely "dl_question_8_mse" which generates graphs for the mean square error loss function, and "dl_question_8_cross_entropy" which                generates graphs for the cross-entropy loss function.

Code execution Instruction: To run this , file simply modify the "project name" with your own project name in the following line of code: sweep_id = wandb.sweep(sweep_config, project="dl_assignment") and provide your API Key.
