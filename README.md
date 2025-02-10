# INSAID-Fraud-Detection-Model-for-Financial-Transactions

Expectations: Your task is to execute the process for proactive detection of fraud while answering following questions.

1. Data cleaning including missing values, outliers and multi-collinearity.
2. Describe your fraud detection model in elaboration.
3. How did you select variables to be included in the model?
4. Demonstrate the performance of the model by using best set of tools.
5. What are the key factors that predict fraudulent customer?
6. Do these factors make sense? If yes, How? If not, How not?
7. What kind of prevention should be adopted while company update its infrastructure?
8. Assuming these actions have been implemented, how would you determine if they work?

Here there is a step-by-step process in which the above objective is acheived

1) Reading data from .csv (Comma Separated Value) file into DataFrame using pandas 

2) Check the numerical and categorical features and find the missing values in DataFrame. 
There are a total of 8 numerical features
There are a total of 3 categorical features
Finding the number of missing values in DataFrame, there is no null value in all the column 

3) Plotting the outlier using the matplotlib library 
Hence the data is large converting column values to logarithmic values for better analysis of the data distribution of column ['oldbalanceOrg', 'oldbalanceDest', 'amount']

4) Analysis of data of Column 'type' in the data frame
Here, the Categorical column type of transaction is important Whereas other Categorical columns like 'nameOrig', 'nameDest' Don't contain any significance toward the model training, analysis, or prediction.
Here is some interesting insight from the data of column 'type' 
Transaction type TRANSFER, CASH_OUT are most prone to get fraud whereas other types of transactions 'PAYMENT', 'DEBIT', and 'CASH_IN' doesn't have any fraud cases reported

5). Plotting histogram to further analysis of fraud case in TRANSFER, CASH_OUT, Also finding min, max, and mean of amount column that is fraud in TRANSFER, CASH_OUT
