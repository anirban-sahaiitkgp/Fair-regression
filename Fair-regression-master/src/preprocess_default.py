import pandas as pd
import numpy as np
import os
from CONSTANTS import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main():
    df = pd.read_excel(os.path.join(RAW_DATA_DIR, 'Default/default of credit card clients.xls'), header=1)

    df.drop(['ID'], axis=1, inplace=True)

    # does the dataframe have any missing value
    assert df.isnull().values.any() == False

    # dump into csv
    if not os.path.isdir(os.path.join(PROCESSED_DATA_DIR, 'Default')):
        os.mkdir(os.path.join(PROCESSED_DATA_DIR, 'Default'))
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'Default/default_processed.csv'), index=False)


if __name__ == '__main__':
    main()
