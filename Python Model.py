import pandas as pd
import numpy as np
import os

Folder_path = r"F:\Python 26-08-2024\INSAID"

file_name = "Fraud.csv"

if os.path.exists(Folder_path):
    print("File Exists")
    print("Reading file into dataframe.")
    file_path = os.path.join(Folder_path, file_name)
    df = pd.read_csv(file_path)    
else:
    print('File doesn\'t exist')
      

Column_Name =df.columns
print("Column Name in the fraud.csv file, Total Column : ",len(Column_Name))
print(Column_Name)
# Column_Name = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
#        'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud',
#        'isFlaggedFraud']

# Identify numerical features
numerical_features = df.select_dtypes(include=['int', 'float']).columns
num_numerical_features = len(numerical_features)

# Identify categorical features
categorical_features = df.select_dtypes(include=['object']).columns
num_categorical_features = len(categorical_features)

print("Number of numerical features:", num_numerical_features)
print("Number of categorical features:", num_categorical_features)

#Find the number of missing value in Pandas
#NaN (Not a Number) and None are treated as missing values in Pandas.

print(df.isna().sum())
# step              0
# type              0
# amount            0
# nameOrig          0
# oldbalanceOrg     0
# newbalanceOrig    0
# nameDest          0
# oldbalanceDest    0
# newbalanceDest    0
# isFraud           0
# isFlaggedFraud    0
# dtype: int64


# Plotting the outlier using the matplotlib library
# Converting each column value to logaritm values
import matplotlib.pyplot as plt

item = ['oldbalanceOrg','oldbalanceDest','amount']

for i in item:
    # Creating plot for column oldbalanceOrg
    # Hist, Bar : plt.bar(df['step'],df[i]), Box plot 
    plt.boxplot(np.log2(df[i]))
    plt.title(i)
    # show plot
    plt.show()

# Here Categorical column type of transaction is important
# Whereas other Categorical column like 'nameOrig', 'nameDest'
# Doesn't contain any significance towards the model training, analysis , prediction

unique_values = df['type'].unique()
print("Unique Value in Column type is ",unique_values)
print("Count of unique value in column ")
print(df['type'].value_counts())

# Unique Value in Column 'type' is  
# ['PAYMENT' 'TRANSFER' 'CASH_OUT' 'DEBIT' 'CASH_IN']
# Count of unique value in column 
# type
# CASH_OUT    2237500
# PAYMENT     2151495
# CASH_IN     1399284
# TRANSFER     532909
# DEBIT         41432

# Here is some interesting insight from the data 
# Count of fraud and non fraud case in the various types of transaction
# List of transaction types
type_of_trans = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']

# Count fraud and non-fraud cases for each transaction type
fraud_counts = df[df['isFraud'] == 1].groupby('type').size().reindex(type_of_trans, fill_value=0)
non_fraud_counts = df[df['isFraud'] == 0].groupby('type').size().reindex(type_of_trans, fill_value=0)

# Create a summary DataFrame
summary_df = pd.DataFrame({'Transaction Type': type_of_trans,'Fraud Cases': fraud_counts.values,
    'Non-Fraud Cases': non_fraud_counts.values})

print("Count of Fraud and Non-Fraud Cases by Transaction Type:")
print(summary_df)
        
# Count of Fraud and Non-Fraud Cases by Transaction Type:
#   Transaction Type  Fraud Cases  Non-Fraud Cases
# 0          PAYMENT            0          2151495
# 1         TRANSFER         4097           528812
# 2         CASH_OUT         4116          2233384
# 3            DEBIT            0            41432
# 4          CASH_IN            0          1399284
# fraud_counts.sum() + non_fraud_counts.sum() = 6362620

# Here Transaction type TRANSFER,CASH_OUT are most prone to get fraud 
# Filter fraud cases for TRANSFER and CASH_OUT
#=============================================================================================
# Plotting hisogram to further analysis of fraud case in TRANSFER,CASH_OUT
# Also getting min, max and mean of amount column that is fraud in TRANSFER,CASH_OUT
# import seaborn as sns

# # Filter fraud cases for TRANSFER and CASH_OUT
# fraud_transfer = df[(df['isFraud'] == 1) & (df['type'] == 'TRANSFER')]
# fraud_cash_out = df[(df['isFraud'] == 1) & (df['type'] == 'CASH_OUT')]

# # Calculate min, max, and mean for TRANSFER fraud cases
# if not fraud_transfer.empty:
#     transfer_min = fraud_transfer['amount'].min()
#     transfer_max = fraud_transfer['amount'].max()
#     transfer_mean = fraud_transfer['amount'].mean()
# else:
#     transfer_min, transfer_max, transfer_mean = None, None, None

# # Calculate min, max, and mean for CASH_OUT fraud cases
# if not fraud_cash_out.empty:
#     cash_out_min = fraud_cash_out['amount'].min()
#     cash_out_max = fraud_cash_out['amount'].max()
#     cash_out_mean = fraud_cash_out['amount'].mean()
# else:
#     cash_out_min, cash_out_max, cash_out_mean = None, None, None

# # Print results
# print("TRANSFER Fraud Cases:")
# print(f"Min Amount: {transfer_min}, Max Amount: {transfer_max}, Mean Amount: {transfer_mean}")

# print("\nCASH_OUT Fraud Cases:")
# print(f"Min Amount: {cash_out_min}, Max Amount: {cash_out_max}, Mean Amount: {cash_out_mean}")

# # Plot histograms separately
# plt.figure(figsize=(12, 6))

# # Histogram for TRANSFER fraud cases
# plt.subplot(1, 2, 1)
# if not fraud_transfer.empty:
#     sns.histplot(fraud_transfer['amount'], bins=50, color='red', kde=True)
#     plt.title('TRANSFER Fraud Cases - Amount Distribution')
#     plt.xlabel('Amount')
#     plt.ylabel('Frequency')
# else:
#     plt.text(0.5, 0.5, 'No TRANSFER Fraud Cases', ha='center', va='center')
#     plt.title('TRANSFER Fraud Cases - Amount Distribution')

# # Histogram for CASH_OUT fraud cases
# plt.subplot(1, 2, 2)
# if not fraud_cash_out.empty:
#     sns.histplot(fraud_cash_out['amount'], bins=50, color='blue', kde=True)
#     plt.title('CASH_OUT Fraud Cases - Amount Distribution')
#     plt.xlabel('Amount')
#     plt.ylabel('Frequency')
# else:
#     plt.text(0.5, 0.5, 'No CASH_OUT Fraud Cases', ha='center', va='center')
#     plt.title('CASH_OUT Fraud Cases - Amount Distribution')

# plt.tight_layout()
# plt.show()

#=============================================================================================

# For finding out Collinearity and Handling Multi-Collinearity but
# First Converting Categorical column to Numerical value data
# For this using One-Hot Encoding which will be split into 5 columns:
# type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, and type_TRANSFER
# And assigning 0 and 1 value into the repective 5 column

from sklearn.preprocessing import OneHotEncoder

# Creating copy of the data frame for safer side
data = df.copy()
categorical_columns = ['type']

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(data[categorical_columns])

one_hot_data = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

data_encoded = pd.concat([data, one_hot_data], axis=1)

data_encoded = data_encoded.drop(categorical_columns, axis=1)
print(f"Encoded Employee data : \n{data_encoded.head()}")

# Getting Modified categorical Column 
mod_cat_col = data_encoded.columns
# ['step', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
#        'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud',
#        'isFlaggedFraud', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
#        'type_PAYMENT', 'type_TRANSFER']


# For finding Correlation among the Columns Using pearson Method (pandas corr() function)
# Any NaN values are automatically excluded. 
# To ignore any non-numeric values, use the parameter numeric_only = True
# Using Numeric value to find the Correlation

mod_num_col = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
               'newbalanceDest', 'isFraud','isFlaggedFraud', 'type_CASH_IN', 
               'type_CASH_OUT', 'type_DEBIT','type_PAYMENT', 'type_TRANSFER']

data_mod_encoded = data_encoded[mod_num_col]
corr_matrix = data_mod_encoded.corr(method='pearson')
print(corr_matrix)

# Sorting or plotting the correlation between the different column
plt.matshow(corr_matrix, cmap=plt.cm.gray)
plt.xlabel(mod_num_col)
plt.ylabel(mod_num_col)
plt.show()

# The color code for black is 0 and the color code for white is 255  in gray scale
# This Correlation matrix looks fairly good, since the Correlation between the same column 
# Like corr between amount-amount column is 1 high postive Correlation.
# Which means that we can find the column which is highly correlated positively or negatively
# to each other or with target column. On this basis we can verify/make various assumption 
# which hold true for further analysis the model.

# Plotting the numerical data to analysis , visualize
data_mod_encoded.hist(bins=50,figsize=(20,15))
plt.show()
# figsize=(20,15) argument will set the dimensions of the generated histogram plot to be 20 units wide and 15 units tall,

# Plotting the log of numerical data to analysis , visualize
# np.log2(data_mod_encoded).hist(bins=50,figsize=(20,15))
# plt.show()


# After analysing 
# If two variables are highly correlated, consider dropping one to avoid redundancy. 
# Create a DataFrame to store the correlation values

corr_df = pd.DataFrame(corr_matrix)

# Initialize a dictionary to store the maximum correlation values and corresponding column names
max_corr_dict = {}

# Iterate through the correlation matrix to find the maximum correlation for each column
for col in corr_df.columns:
    max_corr = corr_df[col].drop(col).max()  # Exclude self-correlation
    max_corr_col = corr_df[col].drop(col).idxmax()
    max_corr_dict[col] = (max_corr_col, max_corr)

# Sort the dictionary by maximum correlation values in descending order
sorted_max_corr = sorted(max_corr_dict.items(), key=lambda x: x[1][1], reverse=True)

# Display the sorted results
print("Sorted Maximum Correlation Values:")
for col, (max_col, max_val) in sorted_max_corr:
    print(f"{col} has maximum correlation with {max_col}: {max_val:.4f}")

# Now finding the max correlation between the different column
# As seeing the pairwise correlation of all columns

# Sorted Maximum Correlation Values:
# oldbalanceOrg has maximum correlation with newbalanceOrig: 0.9988
# newbalanceOrig has maximum correlation with oldbalanceOrg: 0.9988
# oldbalanceDest has maximum correlation with newbalanceDest: 0.9766
# newbalanceDest has maximum correlation with oldbalanceDest: 0.9766
# type_CASH_IN has maximum correlation with newbalanceOrig: 0.5274
# amount has maximum correlation with newbalanceDest: 0.4593
# type_TRANSFER has maximum correlation with amount: 0.3659
# type_CASH_OUT has maximum correlation with newbalanceDest: 0.0935
# isFraud has maximum correlation with amount: 0.0767
# isFlaggedFraud has maximum correlation with isFraud: 0.0441
# type_DEBIT has maximum correlation with oldbalanceDest: 0.0093
# type_PAYMENT has maximum correlation with isFlaggedFraud: -0.0011

# Use a correlation matrix or Variance Inflation Factor (VIF) to assess this.
from statsmodels.stats.outliers_influence import variance_inflation_factor

mod_num_col = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                'newbalanceDest', 'isFraud','isFlaggedFraud', 'type_CASH_IN', 
                'type_CASH_OUT', 'type_DEBIT','type_PAYMENT', 'type_TRANSFER']
X = data_encoded[mod_num_col]
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print(vif_data)

#            feature         VIF
# 0           amount    4.070943
# 1    oldbalanceOrg  669.074835
# 2   newbalanceOrig  688.656291
# 3   oldbalanceDest   68.524775
# 4   newbalanceDest   78.925382
# 5          isFraud    1.225252
# 6   isFlaggedFraud    1.002818
# 7     type_CASH_IN    2.180069
# 8    type_CASH_OUT    1.107247
# 9       type_DEBIT    1.001438
# 10    type_PAYMENT    1.001904
# 11   type_TRANSFER    1.242269

# VIF Interpretation:
# VIF > 5 or 10 indicates high multicollinearity, and you may consider dropping the variable.
# This approach helps you decide which variables to drop based on redundancy.

# As per the analysis of multicollinearity using variance_inflation_factor(VIF)
# need to remove below feature to avoid redundancy in the model

# 1    oldbalanceOrg  669.074835
# 2   newbalanceOrig  688.656291
# 3   oldbalanceDest   68.524775
# 4   newbalanceDest   78.925382


# Now as per VIF Interpretation the dropped column will be  newbalanceOrig,newbalanceDest 
# Again finding the VIF for the remaining colun to ceck the multicollinearity
mod_num_col = ['amount', 'oldbalanceOrg', 'oldbalanceDest',
                'isFraud','isFlaggedFraud', 'type_CASH_IN', 
                'type_CASH_OUT', 'type_DEBIT','type_PAYMENT', 'type_TRANSFER']
X = data_encoded[mod_num_col]
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print(vif_data)
#           feature       VIF
# 0          amount  1.257448
# 1   oldbalanceOrg  1.347950
# 2  oldbalanceDest  1.136286
# 3         isFraud  1.010589
# 4  isFlaggedFraud  1.002057
# 5    type_CASH_IN  1.507540
# 6   type_CASH_OUT  1.088977
# 7      type_DEBIT  1.001412
# 8    type_PAYMENT  1.000456
# 9   type_TRANSFER  1.237795

# Here the VIF is less than 5, and hence mod_num_col is the new feature to analysis the model


# Now the new dataframe, feature for Splittng and training model
New_mod_feature_col = ['amount', 'oldbalanceOrg', 'oldbalanceDest',
                'isFraud','isFlaggedFraud', 'type_CASH_IN', 
                'type_CASH_OUT', 'type_DEBIT','type_PAYMENT', 'type_TRANSFER']

New_df = data_encoded[New_mod_feature_col]
# New_df['isFraud'].value_counts()
# isFraud
# 0    6354407
# 1       8213

# Here is the count of target column value count where 0 is not fraud, 1 is fraud
# Model Training: as observed isFraud column there is an class imbalance which means
# fraud transactions were much lower than non-fraudulent ones

from sklearn.model_selection import train_test_split

X = New_df[New_mod_feature_col]
X = X.drop('isFraud',axis=True)
y = New_df[['isFraud']]


# def Splitting_summary(X_train, X_test, y_train, y_test):
#     Tran_type = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
#     # Total number of fraud cases in X_train and X_test
#     print(f"Total number of fraud cases in X_train: {y_train['isFraud'].sum()}")
#     print(f"Total number of fraud cases in X_test: {y_test['isFraud'].sum()}")
#     # Summary of Fraud cases in X_train and X_test with type of transaction vice
#     print("Summary of Fraud cases in X_train and X_test with type of transaction vice:")
#     # Initialize a dictionary to store the cumulative counts
#     cumulative_counts = {tran_type: 0 for tran_type in Tran_type}
#     # Iterate over each transaction type
#     for tran_type in Tran_type:
#         # Count fraud cases in X_train for the current transaction type
#         train_fraud_count = y_train[X_train[tran_type] == 1]['isFraud'].sum()
        
#         # Count fraud cases in X_test for the current transaction type
#         test_fraud_count = y_test[X_test[tran_type] == 1]['isFraud'].sum()
        
#         # Update the cumulative count
#         cumulative_counts[tran_type] = train_fraud_count + test_fraud_count
        
#         # Print the counts for the current transaction type
#         print(f"{tran_type}:")
#         print(f"  X_train fraud cases: {train_fraud_count}")
#         print(f"  X_test fraud cases: {test_fraud_count}")
#         print(f"  Cumulative fraud cases: {cumulative_counts[tran_type]}")
    
#     # Print the total cumulative fraud cases across all transaction types
#     total_cumulative_fraud = sum(cumulative_counts.values())
#     print(f"Total cumulative fraud cases across all transaction types: {total_cumulative_fraud}")


# Model Builing using Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def LogisticRegression_model(X_train, X_test, y_train, y_test):
    # Initialize and train the Logistic Regression model
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = log_reg.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)

# Using simple  train_test_split sampling  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Splitting_summary(X_train, X_test, y_train, y_test)

# Here is the below observation for the column type_CASH_OUT,type_TRANSFER other doesn't have fraud cases
# type_CASH_OUT:
#   X_train fraud cases: 3305
#   X_test fraud cases: 811
#   Cumulative fraud cases: 4116

# type_TRANSFER:
#   X_train fraud cases: 3288
#   X_test fraud cases: 809
#   Cumulative fraud cases: 4097
  
# LogisticRegression_model(X_train, X_test, y_train, y_test)


# Using stratified sampling to handle class imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Splitting_summary(X_train, X_test, y_train, y_test)

# Here is the below observation for the column type_CASH_OUT,type_TRANSFER other doesn't have fraud cases

# type_CASH_OUT:
#   X_train fraud cases: 3318
#   X_test fraud cases: 798
#   Cumulative fraud cases: 4116

# type_TRANSFER:
#   X_train fraud cases: 3252
#   X_test fraud cases: 845
#   Cumulative fraud cases: 4097
  
# LogisticRegression_model(X_train, X_test, y_train, y_test)


#======================================================================================

#11). Feature Scaling As we are aware Logistic Regression is sensitive to the scaling of the input features.
#i).  Using StandardScaler for the input feature scaling for further analysis

# Using stratified sampling to handle class imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Class Imbalance: If your dataset has imbalanced classes (e.g., very few fraud cases 
# compared to non-fraud cases), consider using techniques like oversampling, undersampling, 
# or adjusting the class_weight parameter in LogisticRegression.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Using StandardScaler for feature scaling with class_weight='balanced' in LogisticRegression model")
log_reg = LogisticRegression(random_state=42, class_weight='balanced')
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)

def Evaluation_matix(y_pred,y_test):
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)
    
Evaluation_matix(y_pred,y_test)

#ii). Using MinMaxScaler from sklearn.preprocessing
# Using stratified sampling to handle class imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Class Imbalance: If your dataset has imbalanced classes (e.g., very few fraud cases 
# compared to non-fraud cases), consider using techniques like oversampling, undersampling, 
# or adjusting the class_weight parameter in LogisticRegression.

from sklearn.preprocessing import MinMaxScaler
# Initialize MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Using MinMaxScaler for feature scaling with class_weight='balanced' in LogisticRegression model")
log_reg = LogisticRegression(random_state=42, class_weight='balanced')
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)
Evaluation_matix(y_pred,y_test)


#============================================
# working in jupyter and not in .py file
# 12). Creating Model Selection: Evaluated Decision Tree

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Assuming X and y are already defined
# Using simple train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the classifier object with Gini criterion
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
# Performing training
clf_gini.fit(X_train, y_train)

# Decision tree with entropy criterion
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
# Performing training
clf_entropy.fit(X_train, y_train)

# Predictions using Gini model
y_pred_gini = clf_gini.predict(X_test)

# Predictions using Entropy model
y_pred_entropy = clf_entropy.predict(X_test)

# Evaluation of the Gini model
print("Gini Model Evaluation:")
Evaluation_matix(y_pred_gini,y_test)

# Evaluation of the Entropy model
print("Entropy Model Evaluation:")
Evaluation_matix(y_pred_entropy,y_test)

# Visualizing the Decision Tree (Gini)
plt.figure(figsize=(12,8))
plot_tree(clf_gini, filled=True, feature_names=X.columns, class_names=True)
plt.title("Decision Tree (Gini)")
plt.show()

# Visualizing the Decision Tree (Entropy)
plt.figure(figsize=(12,8))
plot_tree(clf_entropy, filled=True, feature_names=X.columns, class_names=True)
plt.title("Decision Tree (Entropy)")
plt.show()



# 13). Creating Model Selection: Evaluated RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def RandomForestClassifier_Model(X_train,y_train,X_test):
    # Initialize RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the classifier to the training data
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    return y_pred

y_pred = RandomForestClassifier_Model(X_train,y_train,X_test)
print("Random Forest Classifier Model Evaluation:")
Evaluation_matix(y_pred,y_test)


# 14). Creating Model Selection: XGBOOST model for training 
import xgboost as xgb
from sklearn.metrics import accuracy_score
# Converting Dataset into Dmatrix : XGBoost presents the DMatrix class, which optimizes speed and memory for effective dataset storage. 
# To use the XGBoost API, datasets must be converted to this format. Labels and training features are both accepted by DMatrix. 
# enable_categorical is set to True to encrypt Pandas category columns automatically.

def XGboost_model(X_train, y_train,X_test, y_test):
    xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)
    
    n=50
    params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1,}
    model = xgb.train(params=params,dtrain=xgb_train,num_boost_round=n)
    # The code initializes an XGBoost model with hyperparameters like a binary logistic objective, 
    # a maximum tree depth of 3, and a learning rate of 0.1.
    # It then trains the model using the `xgb_train` dataset for 50 boosting rounds.
    
    y_pred = model.predict(xgb_test)
    y_pred = y_pred.astype(int)
    accuracy = accuracy_score(y_test,y_pred)
    print('Accuracy of the model is:', accuracy)

XGboost_model(X_train, y_train,X_test, y_test)

