import pandas as pd 
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def simple_imputer(train):
# For these categorical features, missing values indicate absence of the feature (not missing at random),
# as described in the data description file. We fill them with "NA" to represent "Not Applicable".
     
    na_features = ['Alley', 'PoolQC', 'MiscFeature', 'MasVnrType', 'Fence', 'FireplaceQu',
                   'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',
                   'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'BsmtCond', 'BsmtQual']
    
    for col in na_features:
        train[col] = train[col].fillna("NA")

    # True missing categorical feature → use most frequent value(there is no house without electicity)
    train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

    # Numeric features → use median
    train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())
    train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].median())

    # GarageYrBlt → missing means no garage
    train["GarageYrBlt"] = train["GarageYrBlt"].fillna(0)
    return train
def apply_log1p(df, columns, threshold=0.75):
    """
    Applies log1p transformation to skewed numerical columns.

    Parameters:
        df (: The input dataframe.
        columns : List of numerical columns to check for skewness.
        threshold : Skewness threshold to consider transformation.

    Returns:
        list: Names of columns that were transformed.
    """
    # Compute skewness of each column (ignore NaNs)
    skewness = df[columns].apply(lambda x: skew(x.dropna()))
    
    # Filter columns with skewness > threshold
    skewed = skewness[skewness > threshold]
    
    # Apply log1p transformation to those columns
    for col in skewed.index:
        df[f"log_{col}"] = df[col].apply(np.log1p)
    
    return list(skewed.index)
def outliers_handling(df, columns):
    """
    Removes outliers from the given DataFrame using the IQR method.
    """
    mask = pd.Series([True] * df.shape[0], index=df.index)

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask &= df[col].between(lower, upper)
    
    df = df[mask]
    return df
def onhotencoder(df):
    """
    Applies OneHotEncoding to all categorical columns in the DataFrame.

    This function automatically detects columns with object dtype

    Parameters:
        df : The input DataFrame containing both numerical and categorical data.

    Returns:
        pd.DataFrame: A DataFrame with categorical columns replaced by their encoded versions.
    """
    
    cat_data = df.select_dtypes(include="object").columns.tolist()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    encoded = encoder.fit_transform(df[cat_data])

    #  Create a DataFrame with the encoded features and same index
    encoded_df=pd.DataFrame(encoded,
                            columns=encoder.get_feature_names_out(cat_data)
                            ,index=df.index)

    #  Concatenate the original DataFrame with encoded features
    df_combined = pd.concat([df, encoded_df], axis=1)

    #  Remove original categorical columns
    df_combined.drop(columns=cat_data, inplace=True)

    return df_combined
def processor(df,log_cols=None,outliers_cols=None):
    """
        Full preprocessing pipeline for a DataFrame.
        """

    df=simple_imputer(df)
    df=onhotencoder(df)

    if log_cols:
        skewed=apply_log1p(df,log_cols)
    if outliers_cols:
        df=outliers_handling(df,outliers_cols)
    return df

# Fit and transform training data, return the scaler object as well
def fit_transform(data):
    scaller=StandardScaler()
    return scaller.fit_transform(data),scaller
# Apply scaling to training and validation without data leakage
def scaler(X_train,X_val):
    X_train,X_train_scaller=fit_transform(X_train)
    x_val=X_train_scaller.transform(X_val)
    return X_train,X_val
def feature_selection(X,y,n_splits=5,scoring="r2",step=2,random_state=42):
    """
      Selects the best subset of features using RFECV based on a scoring metric.


    Returns:
        list: Selected feature names.
    """
    estimator=LinearRegression()
    cv=KFold(n_splits=n_splits,shuffle=True,random_state=42)
    rfecv_=RFECV(estimator=estimator,scoring=scoring,step=step)
    rfecv_.fit(X,y)
    # Get names of selected features
    selected_feature=X.columns[rfecv_.support_].tolist()
    return selected_feature






        



