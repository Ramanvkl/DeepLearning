# Overview 

In this project, we used PyTorch Lightning to build a feed-forward neural network model that predicts both house prices (a regression task) and house categories (a classification task). For this purpose, we used a dataset from Kaggle, including 81 attributes (sale price and 80 predictors) for 1460 houses. After preprocessing the data and based on the ‘HouseStyle’, ‘BldgType’, ‘YearBuilt’, and ‘YearRemodAdd’ features, we added a new feature, ‘HouseCategory’, which served as our target variable for the classification task. Having the data ready for model development, we split the data into train, test, and validation sets with the respective size of 70, 15, and 15 percent of the overall data. We tried different activation functions, optimizers, and loss functions to develop our feed-forward neural network model. Finding the optimal specifications for our model, we utilized PyTorch Lightning’s integration with Optuna for hyperparameter optimization. The final model possesses an average accuracy of 0.942, average precision of 0.947, average recall of 0.965, average F1 of 0.954, and RMSE of 38567 on the test set. It is worth mentioning that Pytorch Lightning’s features, such as callback and trainer, are used throughout the process.


# Data Preprocessing

Since the dataset is relatively small, we dropped the columns that their number of missing values exceeded 10 percent of the length of the data to avoid impacting the distributions of variables. Then, we replaced missing values with the median value of the respective column for numeric variables and mode value of the column for categorical variables. Having the missing values handled, we created a new target variable for our classification task to represent the house categories  based on the ‘HouseStyle’, ‘BldgType’, ‘YearBuilt’, and ‘Year RemodAdd’ Since the data is not large enough, we do not create a category for each combination of the mentioned variables. Instead, we define classes based on the following rules:

1. If a house has a ‘BldgType’ of ‘1Fam’ or a ‘HouseStyle’ of ‘1Story’/’1.5Fin’/’1.5Unf’, it is considered a small house.

2. If the ‘YearBuilt’ for a house is 2000 or newer, the house is considered a new house.

3. If the ‘YearRemodAdd’ is larger than the ‘YearBuilt’ by more than 20, the house is considered remodeled.
   
Finally, the HouseCategory of 0 is assigned to houses that are small and new, 1 is assigned to small, old, remodeled houses, 2 is assigned to small, old, not remodeled houses, 3 is assigned to big and new houses, 4 is assigned to big, old, remodeled houses, and 5 is assigned to big, old, not remodeled houses. It is assumed that houses built after 2000 are not remodeled and are all considered new.

We selected the ‘SalePrice’ and ‘HouseCategory’ features as our target variables and the remaining features, except for ‘Id’, as predictors. Due to the small size of our dataset, we utilized label encoding instead of one-hot encoding to avoid overcomplicating the model while encoding categorical variables. Also, we used the min-max scaler instead of the standard scalar to normalize the data because some variables do not follow the Gaussian distribution. Finally, we split the data into train, test, and validation sets with the respective size of 70, 15, and 15 percent of the overall data. It is worth mentioning that our preprocessed data consists of 73 predictors and 2 target variables, and the target variable for the classification task has 6 different classes due to our definition.


# Multi-task Model Building

We used PyTorch Lightning for training our model. Since the model is a multi-task model, the architecture is shared in base layers while the top layers differ for the two tasks. The shared layer consists of two layers with respective sizes of 36 and 18, and the top layer for both tasks has a size of 10. The output layer size for the regression task is 1, and 6 for the classification task. Also, the input layer size is 73, according to the number of predictors. We experimented with different activation functions, optimizers, and loss functions. We also used PyTorch Lightning’s trainer to train all the models in this project. It is worth noting that we initialized the weights using Xavier Initialization and used the MSE loss function for the regression task and the Cross-Entropy Loss function for the classification task. Also, the overall loss function is the sum of the loss functions for individual tasks.


# Activation Functions

We experimented with ReLU, LeakyReLU, and Sigmoid activation functions as well as their combinations. Among all the models, the model with the LeakyReLU activation function performed the best. The following table represents the performance metrics for all the models:

| Activation Function               | RMSE   | Accuracy | Precision | Recall | F1     |
|-----------------------------------|--------|----------|-----------|--------|--------|
| ReLU                              | 118493 | 0.426    | 0.071     | 0.1667 | 0.0996 |
| LeakyReLU                         | 91473  | 0.2411   | 0.0402    | 0.1667 | 0.0648 |
| Sigmoid                           | 197467 | 0.426    | 0.071     | 0.1667 | 0.0996 |
| LeakyReLU (Shared), Sigmoid (Top) | 197470 | 0.426    | 0.071     | 0.1667 | 0.0996 |
| Sigmoid (Top), LeakyReLU (Shared) | 197385 | 0.426    | 0.071     | 0.1667 | 0.0996 |

According to the results, the LeakyReLU activation function significantly improves the performance of the regression task. On the other hand, ReLU and Sigmoid activation functions perform better than LeakyReLU for the classification task. According to the order of magnitude of the target variables for both tasks, the regression error’s contribution to the loss function is expected to be much more significant than the classification error. Therefore, we prioritized better performance in the regression task over the classification task and chose LeakyReLU as the activation function of all the shared and top layers. It is worth mentioning that this setting remained unchanged for all the next steps for consistency.


## Optimizers

Having the activation functions chosen, we experimented with Adam and RMSProp loss functions to find the best-performing alternative for our use case. The learning rates for both the Adam and RMSProp optimizers were set to 0.001 for comparison. The following table represents the performance metrics for both models:

| Optimizer                         | RMSE   | Accuracy | Precision | Recall | F1     |
|-----------------------------------|--------|----------|-----------|--------|--------|
| Adam                              | 91473  | 0.2411   | 0.0402    | 0.1667 | 0.0648 |
| RMSProp                           | 151095 | 0.426    | 0.071     | 0.1667 | 0.0996 |

Similar to the previous part, the Adam optimizer performs worse for the classification tasks, whereas its performance in the regression task is significantly superior to RMSProp. Therefore, we chose Adam as our optimizer because the regression error’s contribution to the loss function is expected to be much more significant than the classification error. This setting also remained unchanged for all the next steps for consistency.


# Loss Functions

Having the activation functions and optimizer algorithm identified, we experimented with different loss functions to find the superior combination of functions for this use case. We started with changing the loss function for the regression task. We tried both the MSE and L1 loss functions for the regression task. The following table represents the performance metrics for both models:

| Regression Loss Function          | RMSE   | Accuracy | Precision | Recall | F1     |
|-----------------------------------|--------|----------|-----------|--------|--------|
| MSE                               | 91473  | 0.2411   | 0.0402    | 0.1667 | 0.0648 |
| L1                                | 78061  | 0.2411   | 0.0402    | 0.1667 | 0.0648 |

According to the results, the model with the L1 loss function for the regression task performs better than the model with the MSE loss function in the regression task while the performance metrics of the models for the classification task are identical. Therefore, we chose the L1 loss function for the regression task. This setting remained unchanged for all the next steps for consistency. It is worth noting that we did not change the loss function for the classification task since we had a multi-class classification task. 

After finding the optimal loss functions for individual tasks, we experimented with the overall loss function of the model to improve the model’s performance. To do so, we changed the aggregation method we used to define the overall loss function based on individual loss functions of each task. We experimented with arithmetic mean and geometric mean of the two loss functions. Since the model with the loss function of the geometric mean of individual loss functions performed better than the other model, we continued working on this model. Having in mind the order of magnitude of the target variable of the regression task is significantly larger than the classification task, we expected the contribution of the regression error to be much more significant as well. Therefore, we also tried to prioritize minimizing the regression error by multiplying its individual loss function by three. It is worth noting that we have changed the definition of the individual loss function here, not the aggregation method. The following table represents the performance metrics for all the models:

| Aggregation method                       | RMSE   | Accuracy | Precision | Recall | F1     |
|------------------------------------------|--------|----------|-----------|--------|--------|
| Arithmetic mean                          | 78061  | 0.2411   | 0.0402    | 0.1667 | 0.0648 |
| Geometric mean                           | 197358 | 0.849    | 0.7212    | 0.6784 | 0.6954 |
| Geometric mean  (regression prioritized) | 197326 | 0.859    | 0.9       | 0.659  | 0.705  |

According to the results, changing the aggregation method significantly improves the model’s performance in the classification task but lowers its performance in the regression task. Since the improvement in the classification task’s performance (almost 3.5X performance) is more significant than the decrease in the regression task’s performance (almost 2.5X), we prioritized the model’s performance in the classification task this time. Therefore, we changed the aggregation method to geometric mean while prioritizing the individual loss function of the regression task. This setting remained unchanged for all the next steps for consistency.


# Model Evaluation 

We evaluated the model throughout the process to find the optimal choices for activation, optimizer, and loss functions. The performance metrics are mentioned in the tables above. It is worth mentioning that all the mentioned performance metrics are measured on the validation set. Since we aimed to optimize the hyperparameters of the model, we initially trained our different models on the training set and evaluated them on the validation set. The test set will not be used until the end of the process to avoid information leakage. According to the results, our best-performing model possesses an RMSE of 197326, average accuracy of 0.859, average precision of 0.9, average recall of 0.659, and average F1 of 0.705. Average classification scores are derived by calculating the arithmetic mean of classification scores for different categories. It is also with noting that this model uses the LeakyReLU activation function, Adam optimizer, L1 loss for the regression task, Cross-Entropy loss for the classification task, and the geometric mean of the individual loss functions prioritizing the loss function of the regression task for the overall loss. The following table represents this model’s performance metrics:

| RMSE   | Accuracy | Precision | Recall | F1     |
|--------|----------|-----------|--------|--------|
| 197326 | 0.859    | 0.9       | 0.659  | 0.705  |


# Advanced PyTorch Lightning Features

As mentioned earlier, we used PyTorch Lightning’s trainer to train all the models. It enabled us to focus on designing and implementing the model’s and made the training process much easier by providing a high-level interface that simplifies the training process. It also standardized training loop that follows best practices for training deep learning models handling essential training components, such as batching, optimization, gradient accumulation, and distributed training. 

We also used its callback during the hyperparameter optimization step. It simplified the implementation of early stopping by allowing us to extend the behavior of the training process without modifying the core training code. Its integration with Optuna also simplified the hyperparameter optimization process.


# Hyperparameter Tuning

Having our model selected, we leveraged PyTorch Lightning’s integration with Optuna to find the optimal hyperparameters for our model. According to our model’s specifications, we tried to find the optimal learning rate for the optimization algorithm used in the model. To do so, we defined a study in Optuna to find the optimal learning rate and used PyTorch Lightning’s trainer and callback system to simplify the process. 

The following table represents the model’s performance metrics before and after hyperparameter optimization:

| Model                                    | RMSE   | Accuracy | Precision | Recall | F1     |
|------------------------------------------|--------|----------|-----------|--------|--------|
| Before Hyperparameter Optimization       | 197326 | 0.859    | 0.9       | 0.659  | 0.705  |
| After Hyperparameter Optimization        | 38567  | 0.942    | 0.947     | 0.965  | 0.954  |

The optimal learning rate for the Adam optimizer was found to be 0.0374. Compared to the non-optimized model, the performance of the optimized model has been significantly improved. Also, the recall and F1 scores for the optimized model are considerably higher than the non-optimized model. In addition, the accuracy and precision of the model are also improved after the hyperparameter optimization. 

It is worth noting that the performance metrics mentioned above for the classification task are the arithmetic means of performance metrics for individual categories. The following table represents the model’s performance metrics in the classification task for individual classes:

| HouseCategory | Precision | Recall | F1    |
|---------------|-----------|--------|-------|
| 0             | 0.954     | 0.949  | 0.952 |
| 1             | 0.884     | 1      | 0.938 |
| 2             | 0.97      | 0.905  | 0.936 |
| 3             | 0.875     | 1      | 0.933 |
| 4             | 1         | 1      | 1     |
| 5             | 1         | 0.936  | 0.967 |


# Conclusion

In this project, we implemented multi-task learning models with PyTorch and PyTorch Lightning, experimented with different activation functions, optimizers, and loss functions in a multi-task learning context, used advanced features of PyTorch Lightning, such as the trainer and callback system, used PyTorch Lightning's integration with Optuna to tune hyperparameters, and handled real-world data and predicted outcomes for multiple tasks using a shared model architecture. Our final model uses the LeakyReLU activation function, Adam optimizer, L1 loss for the regression task, Cross-Entropy loss for the classification task, and the geometric mean of the individual loss functions prioritizing the loss function of the regression task for the overall loss. The optimal learning rate for this model was found to be 0.0374. This model possesses an RMSE of 38567, average accuracy of 0.942, average precision of 0.947, average recall of 0.965, and average F1 of 0.954, measured on the test set.

