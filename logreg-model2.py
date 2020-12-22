# Model 2 Logistic Regression

# Case:
# Measuring if the level of education in the Small Area is influenced
# by single motor car ownership and Internet Access.

# Import pandas to read in the data into the program
import pandas as pds
# Import numpy for maths functions 
import numpy as npy 

# Get training and test data from the dataset
from sklearn.model_selection import train_test_split
# Function to produce a Logistic Regresstion classifier
from sklearn.linear_model import LogisticRegression

# MATLAB library to plot the data
import matplotlib.pyplot as plt

# For Confusion Matrix and Precision
from sklearn import metrics

# For a baseline classifier 
from sklearn.dummy import DummyClassifier

# For hyperparameter selection
from sklearn.model_selection import GridSearchCV


# Add function to classify Education Level
# Target Variable column produced 
# +1 if over 50%, -1 if under 50%
# Input - education level percentage
def assign_tv(edu_level_perc):

    # +1 if over 50%
    if edu_level_perc >= 50.00:
        return 1
    # -1 if under 50%
    elif edu_level_perc <= 50.00:
        return -1
    # invalid number
    else:
        return 0

# Read in the data for the second model 
# Feature 1 = % of Households with Broadband Internet Access
# Feature 2 = % of Households with a Single Motor Car (read in - opposite order)
# Select specific columns in dataset - usecols
# header = 0 as first row has names of columns
dataset_part1 = pds.read_csv('theme_15_small_areas-internet.csv',usecols=['Perc_Households_With_Cars_One_Motor_Car_2011','Perc_Households_With_Internet_Access_Broadband_2011'])

# Check value
print(dataset_part1)

# Target Variable - Education Not Ceased
dataset_part2 = pds.read_csv('theme_10_small_areas-education.csv',usecols=['Perc_Persons_15_And_Over_Edu_Not_Ceased_Total_At_School_University_2011'])
print(dataset_part2)

# Combine into one dataset
dataset_2 = pds.concat([dataset_part1,dataset_part2],axis=1)
print(dataset_2)
# print(dataset_2['Perc_Households_With_Internet_Access_Broadband_2011'])

# Reorder columns : features, target variable
dataset_2 = dataset_2[['Perc_Households_With_Internet_Access_Broadband_2011','Perc_Households_With_Cars_One_Motor_Car_2011','Perc_Persons_15_And_Over_Edu_Not_Ceased_Total_At_School_University_2011']]
print(dataset_2)

# Rename columns
dataset_2 = dataset_2.rename(columns={'Perc_Households_With_Internet_Access_Broadband_2011':'Internet Access Broadband','Perc_Households_With_Cars_One_Motor_Car_2011':'Single Motor Car in Household','Perc_Persons_15_And_Over_Edu_Not_Ceased_Total_At_School_University_2011':'Education Level' })
print(dataset_2)

# Add list to store the target variables 
tv_list = []


# Assign +1 or -1 for the target variable 
for x in range(len(dataset_2)):
    # select each value in Education Level
    target_var = dataset_2['Education Level'][x]
    # print("Column:", target_var)

    # send to function
    # to get +1 or -1
    target_value = assign_tv(target_var)

    # Add the value to a list 
    tv_list.append(target_value)

# Add new column with +1 and -1 markers 
dataset_2['Education Level - target variable'] = tv_list
# Includes new column
print(dataset_2)

# Both features put into one column
features = ['Internet Access Broadband','Single Motor Car in Household']
# From dataset
features_x = dataset_2[features]

# The target variable +1 -1 only
target_variable_y = dataset_2['Education Level - target variable']

# print(features_x)
# print(target_variable_y)

# Training and test data 
# 80% Training data, 20% Test data
training_x, test_x, training_y, test_y = train_test_split(features_x,target_variable_y,test_size=0.2)
# Print to see output 
# print("Train X",training_x)
# print("Test X",test_x)
# print("Train Y",training_y)
# print("Test Y",test_y)

# Logistic Regression model 
# Liblinear penalty so both l1 and l2 can be tested
log_regr = LogisticRegression(solver='liblinear')

# Select optimal C and penalty hyperparameters
c_list = [0.0001,0.01,1,10,100]
penalty_list = ['l1','l2']
# Hyperparamter - create a key-value pair (dict)
hyperparams = dict(C = c_list, penalty = penalty_list)

# Grid Search with cross validation folds = 5
grid_search = GridSearchCV(log_regr, hyperparams,cv=5)
# To get the optimal logistic regression
optimal_model = grid_search.fit(training_x,training_y)
# Print the optimal hyperparameters
optimal_penalty = grid_search.best_estimator_.get_params()['penalty']
optimal_c = grid_search.best_estimator_.get_params()['C']
print("Model 2 - Optimal Penalty: ", optimal_penalty)
print("Model 2 - Optimal C", optimal_c)


# Logistic Regression model 
# Given optimal penalty and optimal C, liblinear solver
logistic_regression = LogisticRegression(penalty=optimal_penalty, C = optimal_c, solver='liblinear')

# Fit with the training data 
logistic_regression.fit(training_x,training_y)

# Generate predictions using test values 
predict_y = logistic_regression.predict(test_x)
print("Model 2 - Predictions: ", predict_y)

# Plot the logistic regression
# figure = plt.figure()

# Different markers for +1 and -1
mark = ["+","o"]
test_x = npy.array(test_x)
for i in range(len(test_x)):
    
    if predict_y[i] == 1:
        m = mark[0]
        l = '+1'
    if predict_y[i] == -1:
        m = mark[1]
        l = '-1'

    # Scatter plot - column 0 (feature 1 = PC in Household) 
    # column 1 (feature 2 = Education Level)
    # Vary the label 
    sctr = plt.scatter(test_x[[i],0], test_x[[i],1], marker =m, color = 'tomato',label=l)

plt.xlabel('Internet Access Broadband')
plt.ylabel('Single Motor Car in Household')
plt.title('Model 2 - Education Level')
# Legend 
plt.legend(handles=[sctr])

plt.show()

# Confusion matrix produced 
confusion_matrix = metrics.confusion_matrix(test_y,predict_y)
# Print to terminal
print("Model 2 - Confusion Matrix: ")
print(confusion_matrix)

# Measure precision and print to terminal
print("Model 2 - Prediction precision: ")
print(metrics.precision_score(test_y,predict_y))

# Add mean absolute error for the model
print("Model 2 - Mean absolute error: ")
# Mean absolute error - actual values, estimates
print(metrics.mean_absolute_error(test_y,predict_y))


# Baseline model - for comparison
# Uniform strategy - random predictions at all times
dummy_classifier = DummyClassifier(strategy='uniform')

# Train the model with the training data
dummy_classifier.fit(training_x,training_y)

# Produce predictions 
dummy_predict_y = dummy_classifier.predict(test_x)

# Different markers for +1 and -1
mark = ["+","o"]
test_x = npy.array(test_x)
for i in range(len(test_x)):
    
    if dummy_predict_y[i] == 1:
        m = mark[0]
        l = '+1'
    if dummy_predict_y[i] == -1:
        m = mark[1]
        l = '-1'

    # Scatter plot - column 0 (feature 1 = PC in Household) 
    # column 1 (feature 2 = Education Level)
    sctr = plt.scatter(test_x[[i],0], test_x[[i],1], marker =m, color = 'firebrick',label=l)

plt.xlabel('Internet Access Broadband')
plt.ylabel('Single Motor Car in Household')
plt.title('Baseline Model 2 - Education Level')
# Legend 
plt.legend(handles=[sctr])

plt.show()

# Confusion matrix produced 
dummy_confusion_matrix = metrics.confusion_matrix(test_y,dummy_predict_y)
# Print to terminal
print("Baseline Model 2 Confusion Matrix: ")
print(dummy_confusion_matrix)

# Measure precision and print to terminal
print("Baseline Model 2 Prediction precision: ")
print(metrics.precision_score(test_y,dummy_predict_y))

# Add mean absolute error for the baseline model
print("Baseline Model 2 Mean absolute error: ")
# Mean absolute error - actual values, estimates
print(metrics.mean_absolute_error(test_y,dummy_predict_y))