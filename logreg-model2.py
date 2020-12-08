# Model 2 Logistic Regression

# Case:
# Measuring if the level of education in the Small Area is influenced
# by single motor car ownership and Internet Access.

# Import pandas to read in the data into the program
import pandas as pds
# Import numpy for maths functions 
import numpy as npy 

# Read in the data for the second model 
# Feature 1 = % of Households with Broadband Internet Access
# Feature 2 = % of Households with a Single Motor Car (read in - opposite order)
# Select specific columns in dataset - usecols
# header = 0 as first row has names of columns
dataset_part1 = pds.read_csv('theme_15_small_areas-internet.csv',usecols=['Perc_Households_With_Cars_One_Motor_Car_2011','Perc_Households_With_Internet_Access_Broadband_2011'])

# Check value
print(dataset_part1)

# Target Variable - Education Not Ceased
dataset_part2 = pds.read_csv('theme_10_small_Areas-education.csv',usecols=['Perc_Persons_15_And_Over_Edu_Not_Ceased_Total_At_School_University_2011'])
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