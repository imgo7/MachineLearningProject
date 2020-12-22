#!/usr/bin/env python3

"""Use car ownership and PC and Internet access to predict highest level of
education achieved on average.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import minmax_scale, PolynomialFeatures

import itertools

__author__ = 'Basil L. Contovounesios'
__email__ = 'contovob@tcd.ie'
__version__ = '2020-12-22'


def lvl_or_nan(pct):
    """Convert `pct` to a float or NaN on failure."""
    try:
        return float(pct)
    except ValueError:
        return np.nan


def weight_avg(row):
    """Return the weighted average of array `row`."""
    return np.average(range(0, len(row)), weights=row)


def powerset(iterable):
    """Return the powerset of `iterable` as an iterator.
    Borrowed from https://docs.python.org/3/library/itertools.html.
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r)
                                         for r in range(len(s) + 1))


# Names of the columns of interest in the education dataset.  Each
# column represents the %-age of the small area's population above the
# age of 15 that has achieved a particular level of education.  The
# columns are ordered from no formal education to postgraduate level,
# with an extra 8th column indicating the %-age with no data.  The
# latter is used only in preprocessing and will later be discarded.
lvl_cols_orig = [
    f'Perc_Persons_Aged_15_Plus_By_Highest_Level_of_Edu_{lvl}_2011'
    for lvl in ('No_Formal_Edu',
                'Primary',
                'Secondary',
                'Technical_Vocational',
                'Non_Degree',
                'Degree_Level',
                'Post_Grad_Level',
                'Not_Stated')]

# Corresponding human readable column summaries.
lvl_cols = ['No Formal Education',
            'Primary',
            'Secondary',
            'Technical Vocational',
            'Non Degree',
            'Degree',
            'Post Grad',
            'Not Stated']

# Names of the columns of interest in the Internet dataset.  Each
# column represents the %-age of the small area's households that have
# access to a particular commodity, namely a car, PC, or the Internet.
# Car ownership is broken down by the number of cars owned, and
# Internet access is categorised as either being broadband or not.
net_cols_orig = [
    'Perc_Households_With_Cars_No_Motor_Car_2011',
    'Perc_Households_With_Cars_One_Motor_Car_2011',
    'Perc_Households_With_Cars_Two_Motor_Cars_2011',
    'Perc_Households_With_Cars_Three_Motor_Cars_2011',
    'Perc_Households_With_Cars_Four_Or_More_Motor_Cars_2011',
    'Perc_Households_With_Personal_Computer_Yes_2011',
    'Perc_Households_With_Internet_Access_Broadband_2011',
    'Perc_Households_With_Internet_Access_Other_Connection_2011'
]

# Corresponding human readabale column summaries.
net_cols = ['0 Cars',
            '1 Car',
            '2 Cars',
            '3 Cars',
            '4+ Cars',
            'PC',
            'Broadband',
            'Other Net']

# Read in the datasets.
df_edu = pd.read_csv('theme_10_small_areas-education.csv',
                     usecols=lvl_cols_orig,
                     converters={col: lvl_or_nan for col in lvl_cols_orig})
df_net = pd.read_csv('theme_15_small_areas-internet.csv',
                     usecols=net_cols_orig)

# Make their columns human-readable.
df_edu.rename(columns=dict(zip(lvl_cols_orig, lvl_cols)), inplace=True)
df_net.rename(columns=dict(zip(net_cols_orig, net_cols)), inplace=True)

# Summarise the datasets.
print(df_edu)
print(df_net)

# Join the datasets (their small areas already line up).
df_big = pd.concat((df_net, df_edu), axis=1)
# Drop rows with no education level information.
df_big.dropna(inplace=True)
df_big = df_big[df_big['Not Stated'] < 100]
df_big.drop(columns='Not Stated', inplace=True)
lvl_cols.pop()

# Print the resulting super-dataset.
print(df_big)

# Combine different levels of education and numbers of cars into a
# single weighted average for each, and replace the multiple old
# columns with each of the new columns.
lvl_avg = df_big[lvl_cols].apply(weight_avg, axis=1, raw=True)
car_avg = df_big[net_cols[:5]].apply(weight_avg, axis=1, raw=True)
df = pd.concat((car_avg, df_big[net_cols[5:]], lvl_avg), axis=1)

# Give the new columns names, and print the resulting dataset.
df.rename(columns={df.columns[0]: 'Avg. Cars',
                   df.columns[-1]: 'Avg. Education'},
          inplace=True)
print(df)

# Scale all of the features into the range [0, 1].
dataset = minmax_scale(df)
xtrain = dataset[:, :-1]
ytrain = dataset[:, -1]

# Set up matplotlib.
plt.rcParams.update({'figure.autolayout': True,
                     'lines.markersize': 2})

# Visualise the dataset for each of the 4 features.
fig, axs = plt.subplots(2, 2)
((ax0, ax1), (ax2, ax3)) = axs
ax0.scatter(xtrain[:, 0], ytrain)
ax0.set(xlabel='# of cars')
ax1.scatter(xtrain[:, 1], ytrain)
ax1.set(xlabel='PC access')
ax2.scatter(xtrain[:, 2], ytrain)
ax2.set(xlabel='Broadband access')
ax3.scatter(xtrain[:, 3], ytrain)
ax3.set(xlabel='Other Net access')
for ax in axs.flat:
    ax.set(ylabel='Max. education level')
plt.show()
plt.close()

# Cross-validate polynomial features and select feature subsets.
feat_scores = []
featsets = list(powerset(range(0, 4)))[1:]
degs = range(1, 6)
for feats in featsets:
    x = xtrain[:, feats]
    for deg in degs:
        xpoly = PolynomialFeatures(deg).fit_transform(x)
        model = KNeighborsRegressor()
        feat_scores.append(cross_val_score(model, x, ytrain))
feat_scores = np.array(feat_scores)

# Cross-validate number of neighbours.
k_scores = []
ks = range(1, 26)
for k in ks:
    # No difference with weights='distance'.
    model = KNeighborsRegressor(k)
    k_scores.append(cross_val_score(model, xtrain, ytrain))
k_scores = np.array(k_scores)

# Plot the cross-validation results.
fig, axs = plt.subplots(1, 2)
fig.suptitle('Feature selection by cross-validation')
ax0, ax1 = axs
ax0.errorbar(range(0, len(feat_scores)), feat_scores.mean(axis=1),
             yerr=feat_scores.std(axis=1), linewidth=2, elinewidth=1,
             ecolor='gray', capsize=4)
ax0.set(xlabel='Feature & polynomial degree combination')
ax1.errorbar(ks, k_scores.mean(axis=1), yerr=k_scores.std(axis=1),
             linewidth=2, elinewidth=1, ecolor='gray', capsize=4)
ax1.set(xlabel='Number of neighbours')
for ax in axs:
    ax.set(ylabel='R^2 score')
    ax.label_outer()
plt.show()
plt.close()

# Train optimal kNN and dummy mean regressors.
knn = KNeighborsRegressor(20)
dummy = DummyRegressor()
knn_pred = cross_val_predict(knn, xtrain, ytrain)
dummy_pred = cross_val_predict(dummy, xtrain, ytrain)
knn_mse = cross_val_score(knn, xtrain, ytrain,
                          scoring='neg_mean_squared_error')
dummy_mse = cross_val_score(dummy, xtrain, ytrain,
                            scoring='neg_mean_squared_error')

# Evaluate the models.
print('MSE scores')
print(f'{repr(knn):36}', knn_mse.mean())
print(f'{repr(dummy):36}', dummy_mse.mean())
bb = xtrain[:, 2]
pred = np.c_[bb, knn_pred, dummy_pred]
pred = pred[bb.argsort()]
plt.figure()
plt.scatter(bb, ytrain, c='gray', label='Training data')
plt.plot(pred[:, 0], pred[:, 1], label='kNN')
plt.plot(pred[:, 0], pred[:, 2], label='Baseline (mean)')
plt.xlabel('Broadband access')
plt.ylabel('Max. education level')
plt.legend()
plt.show()
plt.close()
