import pandas as pd
import os
from CONSTANTS import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main():
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'COMPAS/compas-scores-two-years-violent.csv'))

    # only consider black and white defendants who were assigned COMPAS risk scores within 30 days of their arrest and were not arrested for an ordinary
    # traffic crime
    idx = (df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30) & (df['c_charge_degree'] != "O") & (df['v_score_text'] != 'N/A')
    df = df[idx]
    i = (df['race'] == 'African-American') | (df['race'] == 'Caucasian')
    df = df[i]

    # only consider race, age, sex, number of prior convictions, COMPAS violent crime risk score and its category
    # 'is_violent_recid' is the target
    df = df[['race', 'age', 'sex', 'priors_count', 'v_decile_score', 'v_score_text', 'is_violent_recid']]
    race_uniq = df['race'].unique()

    # convert categorical variables into dummy/indicator variables
    race = pd.get_dummies(df['race'])
    sex = pd.get_dummies(df['sex'])
    v_decile_score = pd.get_dummies(df['v_decile_score'], prefix="score")
    v_score_text = pd.get_dummies(df['v_score_text'])

    # drop the categorical features and append the corresponding indicator variables
    df.drop(['race', 'sex', 'v_decile_score', 'v_score_text'], axis=1, inplace=True)
    df = pd.concat([df, race, sex, v_decile_score, v_score_text], axis=1)

    # does the dataframe have any missing value
    assert df.isnull().values.any() == False

    # dump into csv
    if not os.path.isdir(os.path.join(PROCESSED_DATA_DIR, 'COMPAS')):
        os.mkdir(os.path.join(PROCESSED_DATA_DIR, 'COMPAS'))
    df.to_csv(os.path.join(PROCESSED_DATA_DIR,'COMPAS/compas_processed.csv'), index=False)

    # create csvs for separate models setting
    for sensitive_attr_val in race_uniq:
        split_df = df[df[sensitive_attr_val] == 1]
        split_df = split_df.drop(sensitive_attr_val, axis=1)
        split_df.to_csv(os.path.join(PROCESSED_DATA_DIR,'COMPAS/compas_' + sensitive_attr_val + '.csv'), index=False)


if __name__ == '__main__':
    main()
