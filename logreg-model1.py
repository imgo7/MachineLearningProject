# Model 1 Logistic Regression

# Case:
# Households in the Small Area have Internet Access taking
# into account PC instance and level of education.

# Import pandas to read in the data into the program
import pandas as pds
# Import numpy for maths functions 
import numpy as npy 

# To get training and test data from the dataset
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

# Add function to classify Broadband
# Target Variable column produced 
# +1 if over 50%, -1 if under 50%
# Input - internet level percentage
def assign_tv(internet_level_perc):

    # +1 if over 50%
    if internet_level_perc >= 50.00:
        return 1
    # -1 if under 50%
    elif internet_level_perc <= 50.00:
        return -1
    # invalid number
    else:
        return 0



# Read in the data for the first model 
# Feature 1 = % of Households with a PC
# Target Variable = % of Households with Broadband Internet Access
# Select specific columns in dataset - usecols
# header = 0 as first row has names of columns
dataset_part1 = pds.read_csv('theme_15_small_areas-internet.csv',usecols=['Perc_Households_With_Personal_Computer_Yes_2011','Perc_Households_With_Internet_Access_Broadband_2011'])
# Discard last row, as first row = headers
# dataset_part1.drop(index=0)
# Check value
print(dataset_part1)

# Feature 2 - Education not ceased at School or University
dataset_part2 = pds.read_csv('theme_10_small_Areas-education.csv',usecols=['Perc_Persons_15_And_Over_Edu_Not_Ceased_Total_At_School_University_2011'])
print(dataset_part2)

# Combine into one dataset
# Merge 2 datasets
# Feature 1 = Household PC level to the left
dataset_1 = pds.concat([dataset_part1,dataset_part2],axis=1)
# print(dataset_1)
# print(dataset_1['Perc_Households_With_Internet_Access_Broadband_2011'])

# Reorder columns : features, target variable
dataset_1 = dataset_1[['Perc_Households_With_Personal_Computer_Yes_2011','Perc_Persons_15_And_Over_Edu_Not_Ceased_Total_At_School_University_2011','Perc_Households_With_Internet_Access_Broadband_2011']]
# print(dataset_1)
# print(dataset_1['Perc_Persons_15_And_Over_Edu_Not_Ceased_Total_At_School_University_2011'])

# Rename columns
dataset_1 = dataset_1.rename(columns={'Perc_Households_With_Personal_Computer_Yes_2011':'PC in Household','Perc_Persons_15_And_Over_Edu_Not_Ceased_Total_At_School_University_2011':'Education Level','Perc_Households_With_Internet_Access_Broadband_2011':'Internet Access Broadband' })
print(dataset_1)

# Add list to store the target variables 
tv_list = []


# Assign +1 or -1 for the target variable 
for x in range(18489):
    # select each value in Internet Broadband
    target_var = dataset_1['Internet Access Broadband'][x]
    # print("Column:", target_var)

    # send to function
    # to get +1 or -1
    target_value = assign_tv(target_var)

    # Add the value to a list 
    tv_list.append(target_value)

# Add new column with +1 and -1 markers 
dataset_1['Internet Broadband - target variable'] = tv_list
# Drop last row - empty with NaN
dataset_1 = dataset_1.drop([dataset_1.index[18488]])
# Includes new column and drops last row 
print(dataset_1)

# All features put into one column
features = ['PC in Household','Education Level']
# From dataset
features_x = dataset_1[features]

# Target variable +1 -1 only
target_variable_y = dataset_1['Internet Broadband - target variable']

# print(features_x)
# print(target_variable_y)

# Training and test data 
# 80% Training data, 20% Test data
training_x, test_x, training_y, test_y = train_test_split(features_x,target_variable_y,test_size=0.2)
# Print to see output 
# print("Train X",training_x)
print("Test X",test_x)
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
print("Model 1 - Optimal Penalty: ", optimal_penalty)
print("Model 1 - Optimal C", optimal_c)


# Logistic Regression model 
# Given optimal penalty and optimal C, liblinear solver
# Increase number of iterations to solve liblinear error
logistic_regression = LogisticRegression(penalty=optimal_penalty, C = optimal_c, solver='liblinear',max_iter=500)

# Fit with the training data 
logistic_regression.fit(training_x,training_y)

# Generate predictions using test values 
predict_y = logistic_regression.predict(test_x)
print("Model 1 - Predictions: ", predict_y)

# Plot the logistic regression
# figure = plt.figure()

# Different markers for +1 and -1
mark = ['+','o']
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
    sctr = plt.scatter(test_x[[i],0], test_x[[i],1], marker =m, color = 'turquoise',label=l)

plt.xlabel('PC in Household')
plt.ylabel('Education Level')
plt.title('Model 1 - Internet Broadband Access')
# Legend 
plt.legend(handles=[sctr])

plt.show()

# Confusion matrix produced 
confusion_matrix = metrics.confusion_matrix(test_y,predict_y)
# Print to terminal
print("Model 1 - Confusion Matrix: ")
print(confusion_matrix)

# Measure precision and print to terminal
print("Model 1 - Prediction precision: ")
print(metrics.precision_score(test_y,predict_y))

# Add mean absolute error for the model
print("Model 1 - Mean absolute error: ")
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
mark = ['+','o']
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
    sctr = plt.scatter(test_x[[i],0], test_x[[i],1], marker =m, color = 'midnightblue',label=l)

plt.xlabel('PC in Household')
plt.ylabel('Education Level')
plt.title('Baseline Model 1 - Internet Broadband Access')
# Legend 
plt.legend(handles=[sctr])

plt.show()

# Confusion matrix produced for baseline
dummy_confusion_matrix = metrics.confusion_matrix(test_y,dummy_predict_y)
# Print to terminal
print("Baseline Model 1 Confusion Matrix: ")
print(dummy_confusion_matrix)

# Measure precision for baseline and print to terminal
print("Baseline Model 1 Prediction precision: ")
print(metrics.precision_score(test_y,dummy_predict_y))

# Add mean absolute error for the baseline model
print("Baseline Model 1 Mean absolute error: ")
# Mean absolute error - actual values, estimates
print(metrics.mean_absolute_error(test_y,dummy_predict_y))


    

