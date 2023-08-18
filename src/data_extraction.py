

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import *

Log(AppConst.DATA_EXTRACTION)
AppPath()


def data_preparation():
    

    Log().log.info("start extract_data_training")

    data_path = os.path.join(AppPath.RAW)
    # data_url = dvc.api.get_url(path=data_path)
    data_df = pd.read_csv(data_path, nrows=5000)

    data_df.loc[data_df.SeniorCitizen==0,'SeniorCitizen'] = "No"   #convert 0 to No in all data instances
    data_df.loc[data_df.SeniorCitizen==1,'SeniorCitizen'] = "Yes"  #convert 1 to Yes in all data instances

    data_df['TotalCharges'] = pd.to_numeric(data_df['TotalCharges'],errors='coerce')
    #Fill the missing values with with the median value
    data_df['TotalCharges'] = data_df['TotalCharges'].fillna(data_df['TotalCharges'].median())


    data_df.drop(["customerID"],axis=1,inplace = True)

    # Encode categorical features

    #Defining the map function
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0})

    ## Encoding target feature
    data_df['Churn'] = data_df[['Churn']].apply(binary_map)

    # Encoding gender category
    data_df['gender'] = data_df['gender'].map({'Male':1, 'Female':0})

    #Encoding other binary category
    binary_list = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    data_df[binary_list] = data_df[binary_list].apply(binary_map)

    #Encoding the other categoric features with more than two categories
    data_df = pd.get_dummies(data_df, drop_first=True)
    #feature scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    data_df['tenure'] = sc.fit_transform(data_df[['tenure']])
    data_df['MonthlyCharges'] = sc.fit_transform(data_df[['MonthlyCharges']])
    data_df['TotalCharges'] = sc.fit_transform(data_df[['TotalCharges']])

    X = data_df.drop('Churn', axis=1)
    y = data_df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

    Log().log.info("----- Feature schema -----")
    Log().log.info(X_train.info())

    Log().log.info("----- Example features -----")
    Log().log.info(X_train.head())

    X_train.to_csv(AppPath.TRAIN_X, index = False)
    X_test.to_csv(AppPath.TEST_X, index = False)
    y_train.to_csv(AppPath.TRAIN_Y, index = False)
    y_test.to_csv(AppPath.TEST_Y, index= False)

if __name__ == "__main__":
    data_preparation()