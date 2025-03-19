# INSAID-Fraud-Detection-Model-for-Financial-Transactions

Expectations: Your task is to execute the process for proactive detection of fraud while answering the following questions.

1. Data cleaning including missing values, outliers, and multi-collinearity.
2. Describe your fraud detection model in elaboration.
3. How did you select variables to be included in the model?
4. Demonstrate the performance of the model by using the best set of tools.
5. What are the key factors that predict fraudulent customers?
6. Do these factors make sense? If yes, How? If not, How not?
7. What kind of prevention should be adopted while the company updates its infrastructure?
8. Assuming these actions have been implemented, how would you determine if they work?

Here there is a step-by-step process in which the above objective is achieved

1). Reading data from .csv (Comma Separated Value) file into DataFrame using pandas 

2). Check the numerical and categorical features and find the missing values in DataFrame. 
There are a total of 8 numerical features
There are a total of 3 categorical features
Finding the number of missing values in DataFrame, there is no null value in all the column 

3). Plotting the outlier using the matplotlib library 
Hence the data is large converting column values to logarithmic values for better analysis of the data distribution of column ['oldbalanceOrg', 'oldbalanceDest', 'amount']

4). Analysis of data of Column 'type' in the data frame
Here, the Categorical column type of transaction is important Whereas other Categorical columns like 'nameOrig', 'nameDest' Don't contain any significance toward the model training, analysis, or prediction.
Here is some interesting insight from the data of column 'type' 
Transaction type TRANSFER, CASH_OUT are most prone to get fraud whereas other types of transactions 'PAYMENT', 'DEBIT', and 'CASH_IN' don't have any fraud cases reported

5). Plotting histogram to further analysis of fraud case in TRANSFER, CASH_OUT, Also finding min, max, and mean of amount column that is fraud in TRANSFER, CASH_OUT

6). Handling Multi-Collinearity and finding the correlation coefficient between the columns To do the same
i). First Converting Categorical column to Numerical value data 
ii). For this use One-Hot Encoding which will split one categorical column to the number of unique values present in that respective columns
In this Data frame, there are 3 categorical columns ['type', 'nameOrig', 'nameDest']. The nameOrig and nameDest is just the name of the transmitter and receiver but the type of transaction has major significance towards the model and the amount distribution which is getting fraudulent 

7). Finding Correlation among the Columns Using the Pearson method (Using function pandas corr()), 
Using Numeric values to find the Correlation
Any NaN values are automatically excluded. To ignore any non-numeric values, use the parameter numeric_only = True

By plotting the correlation matrix in grayscale, the color code for black is 0 and the color code for white is 255 in grayscale
 This Correlation matrix looks fairly good, therefore the Correlation between the same column is high, like corr between the amount-amount column is 1 high positive Correlation. This means that we can find the column which is highly correlated positively or negatively to each other or with the target column. On this basis, we can verify/make various assumptions that hold for further analysis of the model.

8). Getting Variance Inflation Factor (VIF) to check for multi-collinearity 
Variance Inflation Factor (VIF) provides a measure of multicollinearity among the independent variables in a multiple regression model.
VIF Interpretation: VIF > 5 or 10 indicates high multicollinearity, and you may consider dropping the variable.
This approach helps you decide which variables to drop based on redundancy.

 As per the analysis of multicollinearity using variance_inflation_factor(VIF), need to remove the below feature to avoid redundancy in the model
 1    oldbalanceOrg  669.074835
 2   oldbalanceDest   68.524775
 Now as per VIF Interpretation the dropped column will be  newbalanceOrig,newbalanceDest 
 Again finding the VIF for the remaining columns to check the multicollinearity
 Here the VIF is less than 5, and hence mod_num_col is the new feature to analyze the model
 
9). Splitting data for the training and testing of the different model
 Now the new data frame, feature for Splittng and training model
 Here is the count of target column value count where 0 is not fraud, 1 is fraud
 Model Training: as observed isFraud column there is a class imbalance which means
 Fraud transactions were much lower than non-fraudulent ones
 
 Firstly will use simple train_test_split (without handling class imbalance)
 secondly will use train_test_split (with handling class imbalance , stratify = y )
 
10). Model Building using Logistic Regression model from sklearn linear_model and model evaluation metrics accuracy_score, classification_report, confusion_matrix

Key insight after the Logistic Regression model with training through train_test_split:
a).  Using simple train_test_split the Accuracy score achieved of 0.9987
b).  Using stratified sampling to handle class imbalance the Accuracy score achieved of 0.9987

11). Feature Scaling As we are aware Logistic Regression is sensitive to the scaling of the input features.

i).  Using StandardScaler for feature scaling 
Key insight after the Logistic Regression model with training through using feature scaling StandardScaler:

ii). Using MinMaxScaler from sklearn.preprocessing
Key insight after the Logistic Regression model with training through using feature scaling MinMaxScaler:

Insightful result from the above model
As seen in the above result here is the interesting insight which is as follows:
a).  Using simple train_test_split the Accuracy score achieved of 0.9987
b).  Using stratified sampling to handle class imbalance the Accuracy score achieved of 0.9987
i).  StandardScaler: Using this feature scaling the Accuracy score: of 0.9346 significantly decreases as compared with the above model(a,b)
ii). MinMaxScaler:   Using this feature scaling the Accuracy score: of 0.9204 significantly decreases as compared with the above model(a,b)

12). Creating Model Selection 'Decision Tree'.

i).   Model Training and Prediction: Two models are trained: one using the Gini impurity criterion and 
      the other using the Entropy criterion and predictions are made using both models.
ii).  Model Evaluation: Both models are evaluated using a confusion matrix, accuracy, and classification report.
iii). Visualization: The decision trees for both models are visualized using plot_tree from sklearn.tree.

Results of the Model 'Decision Tree' using 'Gini impurity criterion' and 'Entropy criterion' are same.
Accuracy score by 'Gini impurity criterion': 0.9989
Accuracy score by 'Entropy criterion': 0.9989

13). Creating Model Selection: Evaluated RandomForestClassifier 
Results: The Random Forest Classifier Model has an Accuracy score of 0.9996 which is better than the above-trained model

14). Creating Model Selection: XGBOOST model for training 
The XGBOOST Model has an Accuracy score of 0.9987 which is not better than the Random forest model

15). Feature Engineering
Created new feature balance_change = oldbalanceOrg - newbalanceOrig.
Derived transaction_speed = amount/step to detect fast withdrawals.
isFlaggedFraud: Business rule-based flag for illegal transactions.

16). Using GridSearchCV for hyperparameter tuning for the trained model 
This is used to find the optimal parameters to increase the model's accuracy and 
performance up to a certain extent under the trade-off with time and costing 
