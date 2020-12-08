# Model 1 Logistic Regression

# Case:
# Households in the Small Area have Internet Access taking
# into account PC instance and level of education.

# Import pandas to read in the data into the program
import pandas as pds
# Import numpy for maths functions 
import numpy as npy 

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
print(dataset_1)
# print(dataset_1['Perc_Persons_15_And_Over_Edu_Not_Ceased_Total_At_School_University_2011'])

# Rename columns
dataset_1 = dataset_1.rename(columns={'Perc_Households_With_Personal_Computer_Yes_2011':'PC in Household','Perc_Persons_15_And_Over_Edu_Not_Ceased_Total_At_School_University_2011':'Education Level','Perc_Households_With_Internet_Access_Broadband_2011':'Internet Access Broadband' })
print(dataset_1)