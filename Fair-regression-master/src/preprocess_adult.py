import pandas as pd
import numpy as np
import os
from CONSTANTS import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main():
    header_list = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Adult/adult.data'), sep=',\s', engine='python', names=header_list)
    df.loc[(df['salary'] == '<=50K'), 'salary'] = 0
    df.loc[(df['salary'] == '>50K'), 'salary'] = 1
    df = df.replace(to_replace='?', value=np.nan)
    df = df.dropna()

    # convert categorical variables into dummy/indicator variables
    workclass = pd.get_dummies(df['workclass'], prefix="workclass", dtype=bool)
    education = pd.get_dummies(df['education'], prefix="education", dtype=bool)
    marital_status = pd.get_dummies(df['marital-status'], prefix="mstatus", dtype=bool)
    occupation = pd.get_dummies(df['occupation'], prefix="occupation", dtype=bool)
    relationship = pd.get_dummies(df['relationship'], prefix="relationship", dtype=bool)
    race = pd.get_dummies(df['race'], prefix="race", dtype=bool)
    sex = pd.get_dummies(df['sex'], prefix="sex", dtype=bool)
    native_country = pd.get_dummies(df['native-country'], prefix="ncountry", dtype=bool)

    # drop the categorical features and append the corresponding indicator variables
    df.drop(['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'], axis=1, inplace=True)
    df = pd.concat([df, workclass, education, marital_status, occupation, relationship, race, sex, native_country], axis=1)

    # does the dataframe have any missing value
    assert df.isnull().values.any() == False

    d = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
    df = df.astype(d)

    # dump into csv
    if not os.path.isdir(os.path.join(PROCESSED_DATA_DIR, 'Adult')):
        os.mkdir(os.path.join(PROCESSED_DATA_DIR, 'Adult'))
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'Adult/adult_processed.csv'), index=False)


if __name__ == '__main__':
    main()
