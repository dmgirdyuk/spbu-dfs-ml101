import warnings

import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_mushrooms_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = df_train.columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        count_encoder = ce.CountEncoder()
        df_train[columns] = count_encoder.fit_transform(df_train)
        df_test[columns] = count_encoder.transform(df_test)

    standard_scaler = StandardScaler()
    df_train[columns] = standard_scaler.fit_transform(df_train)
    df_test[columns] = standard_scaler.transform(df_test)

    return df_train, df_test
