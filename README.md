# Overview 

In this assignment, we tried to predict customer churn using feed-forward neural networks. The dataset used for this purpose consists of 100 features (including 21 categorical features and 79 numeric variables) for 100,000 records of customers of a telecommunications company obtained from Kaggle. After preprocessing the data, we split it into training and test sets with the respective size of 70 and 30 percent of the complete data. We used PyTorch to train the model, experimented with different network architectures, activation functions, and learning rates to identify their impact on the performance of the model, and selected the best-performing model. 


# Data Preprocessing

Having the data preprocessed, we selected the ‘churn’ column as our target variable and all the remaining columns as predictors. Then, we split the data into training and test sets. The size of the training and test sets are 70 and 30 percent, respectively, and the random state is set to 0 for the sake of reproducibility of the splits. 


# Training the Model

We defined the FNN class inheriting from the torch.nn.Module. The base model defined to have a single hidden layer with the ReLU activation function. Since the input size is similar to the number of predictors at 89, we set the initial size of the hidden layer to 22, which is approximately one-fourth of the input size. This value is derived from our exercise in the class where a hidden size of one-sixth of the input size resulted in a solid performance. The number of epochs was also set to 20. Also, we used the Cross-Entropy loss function and the stochastic gradient descent algorithm for the optimizer. The following table represents the performance metrics of the base model:

Metric | Accuracy |	Precision |	Recall |	F1
--- | --- | --- | --- | --- | 
Value |	0.5057 |	0.5 |	0.0001 |	0.0003


# Experiments
We experimented with different activation functions including ReLU, Leaky ReLU, and Sigmoid. Then, we changed the learning rate to see its impact on the model’s performance. We also changed the size of the hidden layer to see how it impacts the performance of the model. Moreover, we tried adding more hidden layers and studied their impact on the performance. It is worth mentioning that all the experiments have been done with both Leaky ReLU and Sigmoid activation functions. The results of the experiments would be presented in the following sections. It is worth noting that all the parameters and hyperparameters, except for the one mentioned in the related part, are kept consistent for the sake of comparison.  

## Activation Function

Changing the activation function to the Leaky ReLU slightly improved the performance. It is worth mentioning that its impact on the model’s recall was more significant. The following table represents the performance metrics of the model with the Leaky ReLU activation function:

Metric | Accuracy |	Precision |	Recall |	F1
--- | --- | --- | --- | --- | 
Value |	0.5140 |	0.5397 |	0.1142 |	0.1886

Then, we changed the activation function to Sigmoid, which significantly improved the recall. However, it slightly decreased the accuracy and precision of the model. The following table represents the performance metrics of the model with the Sigmoid activation function:

Metric | Accuracy |	Precision |	Recall |	F1
--- | --- | --- | --- | --- | 
Value |	0.4943 |	0.4943 |	0.1 |	0.6616


## Learning Rate

Decreasing or increasing the learning rate for the model with Sigmoid activation function significantly lowered its performance. It shows that the model is very sensitive to the learning rate. The following table represents the performance metrics for the model with Sigmoid activation function and different learning rates:

| Learning Rate | Accuracy  | Precision  | Recall | F1     |
|---------------|-----------|------------|--------|--------|
| 0.005         | 0.5057    | 0          | 0      | 0      |
| 0.01          | 0.4943    | 0.4943     | 1      | 0.6616 |
| 0.02          | 0.5057    | 0          | 0      | 0      |


For the Leaky ReLU activation function, decreasing or increasing the learning rate lowered the performance of the model. The change was more significant with increasing the learning rate. The following table represents the performance metrics for the model with Leaky ReLU activation function and different learning rates:

Learning Rate | Accuracy |	Precision |	Recall |	F1
--- | --- | --- | --- | --- | 
0.005 |	0.5049 |	0.4444 |	0.007 |	0.0138
--- | --- | --- | --- | --- | 
0.01 |	0.5140 |	0.5397 |	0.1142 |	0.1886
--- | --- | --- | --- | --- | 
0.02 |	0.5057 |	0 |	0 |	0


## Hidden Layers

Both the size and number of hidden layers were studied for both the Leaky ReLU and Sigmoid activation functions. It is worth mentioning that, when adding a new hidden layers, the size of all hidden layers were kept similar. This isolated change enabled us to see the sole impact of the changing the number of hidden layers in the architecture on the model’s performance. 

### The Size of Hidden Layers

Decreasing the size of the hidden layer for the model with Leaky ReLU activation function significantly increased the model’s performance. Increasing the size of the hidden layer also improved the performance but the impact was less significant. It is worth mentioning that the learning rate was 0.005 in all the experiments. The following table represents the performance of the model with Leaky ReLU activation function and different sizes of the hidden layer:

Hidden Layer Size | Accuracy |	Precision |	Recall |	F1
--- | --- | --- | --- | --- | 
11 |	0.4943 |	0.4943 |	1 |	0.6616
--- | --- | --- | --- | --- | 
22 |	0.5049 |	0.4444 |	0.007 |	0.0138
--- | --- | --- | --- | --- | 
44 |	0.5072 |	0.5009 |	0.7683 |	0.6065

For the model with Sigmoid activation function, decreasing or increasing the size of the hidden layer significantly lowered the performance. The performance decrease was more significant when lowering the size of the hidden layer. It is worth mentioning that the learning rate was 0.01 in all the experiments. The following table represents the performance of the model with Sigmoid activation function and different sizes of the hidden layer:

Hidden Layer Size | Accuracy |	Precision |	Recall |	F1
--- | --- | --- | --- | --- | 
11 |	0.5057 |	0 |	0 |	0
--- | --- | --- | --- | --- | 
22 |	0.4943 |	0.4943 |	1 |	0.6616
--- | --- | --- | --- | --- | 
44 |	0.5056 |	0.5047 |	0.0988 |	0.1652


### The Number of Hidden Layers

Adding a hidden layer to the model with the Leaky ReLU activation function, slightly improved its performance. However, adding one more layer to the decreased the performance of the model significantly. It is worth mentioning that the size of all hidden layers were set to 44 and the learning rate was 0.005 in all the experiments. The following tables represents the performance of the model with Leaky ReLU activation function and different numbers of hidden layers:

Number of Hidden Layers | Accuracy |	Precision |	Recall |	F1
--- | --- | --- | --- | --- | 
1 |	0.5072 |	0.5009 |	0.7683 |	0.6065
--- | --- | --- | --- | --- | 
2 |	0.4943 |	0.4943 |	0.9999 |	0.6615
--- | --- | --- | --- | --- | 
3 |	0.5057 |	0 |	0 |	0

For the model with the Sigmoid activation function, adding a hidden layer significantly improved the model’s performance. However, adding one more hidden layer to the model did not change the performance at all, which shows this case is an example of diminishing returns. It is worth mentioning that the size of all hidden layers were set to 44 and the learning rate was 0.01 in all the experiments. The following tables represents the performance of the model with Sigmoid activation function and different numbers of hidden layers:

Number of Hidden Layers | Accuracy |	Precision |	Recall |	F1
--- | --- | --- | --- | --- | 
1 |	0.5056 |	0.5047 |	0.0988 |	0.1652
--- | --- | --- | --- | --- | 
2 |	0.4943 |	0.4943 |	1 |	0.6616
--- | --- | --- | --- | --- | 
3 |	0.4943 |	0.4943 |	1 |	0.6616


# Best Performing Model

The best-performing model could achieve accuracy, precision, recall, and f1 of 0.4943, 0.4943, 1, and 0.6616, respectively. It is worth mentioning that we could get similar performance metrics with different architectures and hyperparameter. Different model specifications that yielded the mentioned results are listed in the following table:

Number | Activation Function |	Learning Rate |	Hidden Layer Size |	Number of Hidden Layers
--- | --- | --- | --- | --- | 
1 |	Sigmoid |	0.01 |	22 |	1
--- | --- | --- | --- | --- | 
2 |	Leaky ReLU |	0.005 |	11 |	1
--- | --- | --- | --- | --- | 
3 |	Sigmoid |	0.01 |	44 |	2
--- | --- | --- | --- | --- | 
4 |	Sigmoid |	0.01 |	44 |	3

Since neural networks are computationally expensive, selecting the second or first models is recommended for practical use cases due to the similar performance metrics. Due to the lower size of the hidden layers in the second model, it is expected to be more computationally efficient, especially for larger datasets. 
