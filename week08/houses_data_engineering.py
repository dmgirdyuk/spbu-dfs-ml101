import warnings

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def prepare_houses_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train, df_test = _preprocess_data(df_train=df_train, df_test=df_test)
    df_train, df_test, y_train, y_test = _fill_nans(
        df_train=df_train, df_test=df_test, y_train=y_train, y_test=y_test
    )
    df_train, df_test = _drop_useless_features(df_train=df_train, df_test=df_test)
    df_train, df_test, y_train, y_test = _drop_outliers(
        df_train=df_train, df_test=df_test, y_train=y_train, y_test=y_test
    )
    df_train, df_test = _add_features(df_train=df_train, df_test=df_test)
    df_train, df_test = _normalize_data(df_train=df_train, df_test=df_test)

    return df_train, df_test, y_train, y_test


def _preprocess_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in (df_train, df_test):
        df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})

        # names beginning with numbers are awkward to work with
        df.rename(
            columns={
                "1stFlrSF": "FirstFlrSF",
                "2ndFlrSF": "SecondFlrSF",
                "3SsnPorch": "Threeseasonporch",
            },
            inplace=True,
        )

        # удаляем скореллированные признаки
        df.drop(
            columns=["GarageYrBlt", "TotRmsAbvGrd", "FirstFlrSF", "GarageCars"],
            inplace=True,
        )

    return df_train, df_test


_NA_CAT_COLS = [
    "GarageType",
    "GarageFinish",
    "BsmtFinType2",
    "BsmtExposure",
    "BsmtFinType1",
    "GarageCond",
    "GarageQual",
    "BsmtCond",
    "BsmtQual",
    "FireplaceQu",
    "KitchenQual",
    "HeatingQC",
    "ExterQual",
    "ExterCond",
]

_NA_GROUP_FILL_NUM_COLS = ["LotFrontage", "GarageArea"]
_NA_GROUP_FILL_CAT_COLS = [
    "MasVnrType",
    "MSZoning",
    "Exterior1st",
    "Exterior2nd",
    "SaleType",
    "Electrical",
    "Functional",
]


def _fill_nans(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert not y_train.isna().sum() and not y_test.isna().sum()
    assert (
        not df_train["Neighborhood"].isna().sum()
        and not df_test["Neighborhood"].isna().sum()
    )

    # выбрасываем признаки, процент пропущенных значений у которых больше 80
    nan_info_train = (df_train.isnull().mean() * 100).reset_index()
    nan_info_train.columns = ["column_name", "percentage"]
    nan80_cols = list(nan_info_train[nan_info_train.percentage > 80]["column_name"])
    df_train.drop(columns=nan80_cols, inplace=True)
    df_test.drop(columns=nan80_cols, inplace=True)

    # создаем новую категорию для подмножества категориальных признаков
    df_train[_NA_CAT_COLS] = df_train[_NA_CAT_COLS].fillna("NA")
    df_test[_NA_CAT_COLS] = df_test[_NA_CAT_COLS].fillna("NA")

    # заполняем медианой с учетом группы
    for col in _NA_GROUP_FILL_NUM_COLS:
        na_group_fill_num_mapping = (
            df_train.groupby("Neighborhood")[col].median().to_dict()
        )
        for df in (df_train, df_test):
            for neighb, fill_value in na_group_fill_num_mapping.items():
                mask = df["Neighborhood"] == neighb
                df.loc[mask, col] = df.loc[mask, col].fillna(fill_value)

    # заполняем модой с учетом группы
    for col in _NA_GROUP_FILL_CAT_COLS:
        fill_value_gen = df_train[col].dropna().mode()[0]
        for neighb, df_group in df_train.groupby("Neighborhood"):
            mode_output = df_group[col].dropna().mode()
            fill_value = mode_output[0] if len(mode_output) else fill_value_gen
            for df in (df_train, df_test):
                mask = df["Neighborhood"] == neighb
                df.loc[mask, col] = df.loc[mask, col].fillna(fill_value)

    # заполняем медианой и модой все оставшиеся столбцы с пропущенными значениями
    num_cols = df_train.select_dtypes(exclude=["object"]).columns
    cat_cols = df_train.select_dtypes(include=["object"]).columns
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    num_imputer.fit(df_train[num_cols])
    cat_imputer.fit(df_train[cat_cols])
    for df in (df_train, df_test):
        df[num_cols] = num_imputer.transform(df[num_cols])
        df[cat_cols] = cat_imputer.transform(df[cat_cols])

    return df_train, df_test, y_train, y_test


def _drop_useless_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # отбрасываем категориальные столбцы, у которых 95+% повторяющихся значений
    almost_constant_cat_cols = _get_almost_constant_columns(
        df_train.select_dtypes(include=["object"])
    )
    df_train.drop(columns=almost_constant_cat_cols, inplace=True)
    df_test.drop(columns=almost_constant_cat_cols, inplace=True)

    # удаляем столбцы в интервальной шкале с незначительной дисперсией
    df_train_num = df_train.select_dtypes(exclude=["object"])
    var_thd = VarianceThreshold(threshold=0.1)
    var_thd.fit(df_train_num)
    low_var_cols = df_train_num.columns[~var_thd.get_support()]  # noqa
    df_train.drop(columns=low_var_cols, inplace=True)
    df_test.drop(columns=low_var_cols, inplace=True)

    return df_train, df_test


def _get_almost_constant_columns(df: pd.DataFrame, dropna: bool = True) -> list[str]:
    cols = []
    for col in df:
        counts = df[col].dropna().value_counts() if dropna else df[col].value_counts()
        most_popular_value_count = counts.iloc[0]
        if (most_popular_value_count / len(df)) * 100 > 95:
            cols.append(col)

    return cols


def _drop_outliers(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for col, upper_bound in (
        ("LotFrontage", 200),
        ("LotArea", 100000),
        ("BsmtFinSF1", 4000),
        ("TotalBsmtSF", 5000),
        ("GrLivArea", 4000),
    ):
        drop_index = df_train[df_train[col] > upper_bound].index
        df_train = df_train.drop(drop_index, axis=0)
        y_train = y_train.drop(drop_index, axis=0)

        drop_index = df_test[df_test[col] > upper_bound].index
        df_test = df_test.drop(drop_index, axis=0)
        y_test = y_test.drop(drop_index, axis=0)

    return df_train, df_test, y_train, y_test


_ORDINAL_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
_FINTYPE_MAP = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NA": 0}
_EXPOSE_MAP = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0}

_ORD_COLs = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "GarageQual",
    "GarageCond",
    "FireplaceQu",
]
_FIN_COLS = ["BsmtFinType1", "BsmtFinType2"]


def _transform_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in (df_train, df_test):
        # столбец с числовым признаком, который на самом деле можно представить как
        # категориальный
        df["MSSubClass"] = df["MSSubClass"].apply(str)

        for col in _ORD_COLs:
            df[col] = df[col].map(_ORDINAL_MAP)

        for col in _FIN_COLS:
            df[col] = df[col].map(_FINTYPE_MAP)

        df["BsmtExposure"] = df["BsmtExposure"].map(_EXPOSE_MAP)

    return df_train, df_test


_COLS_TO_BIN = [
    "MasVnrArea",
    "TotalBsmtFin",
    "TotalBsmtSF",
    "SecondFlrSF",
    "WoodDeckSF",
    "TotalPorch",
]


def _add_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_train = _add_new_features(df_train)
        df_test = _add_new_features(df_test)

    df_train, df_test = _add_ohe_features(df_train=df_train, df_test=df_test)

    return df_train, df_test


def _add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalLot"] = df["LotFrontage"] + df["LotArea"]
    df["TotalBsmtFin"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]
    df["TotalSF"] = df["TotalBsmtSF"] + df["SecondFlrSF"]
    df["TotalBath"] = df["FullBath"] + df["HalfBath"]
    df["TotalPorch"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["ScreenPorch"]
    df["LivLotRatio"] = df["GrLivArea"] / df["LotArea"]

    for col in _COLS_TO_BIN:
        df.loc[:, f"{col}_bin"] = df[col].apply(lambda x: 1 if x > 0 else 0)

    return df


def _add_ohe_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_cols = df_train.select_dtypes(exclude=["object"]).columns
    cat_cols = df_train.select_dtypes(include=["object"]).columns

    onehot_encoder = OneHotEncoder(
        sparse_output=False, min_frequency=0.3, handle_unknown="ignore"
    ).fit(df_train[cat_cols])
    df_train_ohe = pd.DataFrame(
        onehot_encoder.transform(df_train[cat_cols]),
        columns=onehot_encoder.get_feature_names_out(),
        index=df_train.index,
    )
    df_test_ohe = pd.DataFrame(
        onehot_encoder.transform(df_test[cat_cols]),
        columns=onehot_encoder.get_feature_names_out(),
        index=df_test.index,
    )
    df_train = pd.concat([df_train[num_cols], df_train_ohe], axis=1)
    df_test = pd.concat([df_test[num_cols], df_test_ohe], axis=1)

    return df_train, df_test


def _normalize_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = RobustScaler()
    df_train[df_train.columns] = scaler.fit_transform(df_train)
    df_test[df_test.columns] = scaler.transform(df_test)

    return df_train, df_test
