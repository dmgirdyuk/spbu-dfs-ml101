import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_wdbc_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    standard_scaler = StandardScaler()
    df_train[df_train.columns] = standard_scaler.fit_transform(df_train)
    df_test[df_train.columns] = standard_scaler.transform(df_test)

    return df_train, df_test
