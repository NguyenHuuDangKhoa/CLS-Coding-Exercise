import numpy as np
import pandas as pd
from typing import Dict, Callable
from scipy.stats import entropy


def mutual_information_binary(X: pd.Series, Y: pd.Series) -> np.float64:
    """
    A function to calculate the mutual information between two discrete variables,
    assuming the two variables X and Y are binary or can be categorized into two groups.
    This approach using normalized histograms to estimate the joint and marginal distributions.
    param X: first variable
    param Y: second variable
    return: mutual information between X and Y
    """
    joint_prob = np.histogram2d(X, Y, bins=2)[0] / len(X)
    prob_X = np.histogram(X, bins=2)[0] / len(X)
    prob_Y = np.histogram(Y, bins=2)[0] / len(Y)

    mi = 0
    for i in range(2):
        for j in range(2):
            if joint_prob[i][j] > 0:
                mi += joint_prob[i][j] * np.log(joint_prob[i][j] / (prob_X[i] * prob_Y[j]))

    return mi


def mutual_information_multiple_discrete(X: pd.Series, Y: pd.Series) -> np.float64:
    """
    A function to calculate the mutual information between two discrete variables,
    assuming the two variables X and Y have multiple discrete values.
    This approach using normalized histograms to estimate the joint and marginal distributions.
    The sum of the joint histogram may not neccessarily be 1.
    param X: first variable
    param Y: second variable
    return: mutual information between X and Y
    """
    joint_prob, x_edges, y_edges = np.histogram2d(X, Y, bins=(len(set(X)), len(set(Y))), density=True)
    prob_X = np.histogram(X, bins=len(set(X)), density=True)[0]
    prob_Y = np.histogram(Y, bins=len(set(Y)), density=True)[0]

    mi = 0
    for i in range(len(set(X))):
        for j in range(len(set(Y))):
            if joint_prob[i][j] > 0:
                mi += joint_prob[i][j] * np.log(joint_prob[i][j] / (prob_X[i] * prob_Y[j]))

    return mi


def mutual_info_with_entropy(X: pd.Series, Y: pd.Series) -> np.float64:
    """
    A function to calculate the mutual information between two discrete variables.
    This approach first calculate the entropies of each individual variable and their join entropy.
    The mutual information is then obtained using the relation: MI(X, Y) = H(X) + H(Y) - H(X, Y)
    The sum of the joint histogram may not neccessarily be 1.
    param X: first variable
    param Y: second variable
    return: mutual information between X and Y
    """
    # Calculate the individual entropies
    h_x = entropy(np.histogram(X, bins=len(set(X)))[0] / len(X))
    h_y = entropy(np.histogram(Y, bins=len(set(Y)))[0] / len(Y))

    # Calculate the joint histogram and joint entropy
    c_xy = np.histogram2d(X, Y, bins=(len(set(X)), len(set(Y))))[0]
    h_xy = entropy(c_xy.reshape(-1))

    # Compute the mutual information
    mutual_info = h_x + h_y - h_xy
    return mutual_info


def compute_mi(data: pd.DataFrame, mi_func: Callable) -> Dict:
    """
    This function compute the mutual information of each pair of variables in the dataset.
    The results are stored in an dictionary.
    param data: an input dataframe containing variables' values 
    param mi_func: a function to calculate the mutual information
    return: a dictionary containing mutual information of all variable pairs.
    """
    results = {}
    vars = data.columns
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            var1 = vars[i]
            var2 = vars[j]
            try:
                mi = mi_func(data[var1], data[var2])
            except ValueError:
                mi = mi_func(np.array(data[var1]).reshape(-1, 1), data[var2], discrete_features=True)
            results[(var1, var2)] = mi
    
    # Display results
    for var, mi_value in results.items():
        print(f"Mutual Information between {var[0]} and {var[1]}: {mi_value}")

    return results