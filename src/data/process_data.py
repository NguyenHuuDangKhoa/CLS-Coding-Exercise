import pandas as pd
import structlog
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logger = structlog.getLogger(__name__)

def remove_na(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function check and remove all missing values from the dataset
    param data: a Pandas dataframe
    return: a Pandas dataframe
    """
    logger.info(f"Number of missing values: \n{data.isna().sum()}\n")
    return data.dropna()

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function check and remove all duplicated values from the dataset
    param data: a Pandas dataframe
    return: a Pandas dataframe
    """
    num_duplicated_rows = data.duplicated().sum()
    logger.info(f"Number of duplicated rows: {num_duplicated_rows}")
    return data.drop_duplicates()

def remove_outliers_iqr_custom(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function use IQR technique to remove some outliers from the dataset
    param data: a Pandas dataframe
    return: a Pandas dataframe
    """
    for column in data.columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data 

def normalize(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function normalize the data set with Min-Max Scaling
    param data: a Pandas dataframe
    return: a Pandas dataframe
    """
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)