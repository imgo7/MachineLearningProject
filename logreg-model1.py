# Model 1 Logistic Regression

# Case:
# Households in the Small Area have Internet Access taking
# into account PC instance and level of education.

# Import pandas to read in the data into the program
import pandas as pds
# Import numpy for maths functions 
import numpy as npy 

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
# Includes new column
print(dataset_1)

    

