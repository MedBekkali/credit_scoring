from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Chemin racine du projet : parent de src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def load_raw():
    app_train = pd.read_csv(DATA_DIR / "application_train.csv")
    app_test  = pd.read_csv(DATA_DIR / "application_test.csv")
    bureau    = pd.read_csv(DATA_DIR / "bureau.csv")
    prev      = pd.read_csv(DATA_DIR / "previous_application.csv")
    pos       = pd.read_csv(DATA_DIR / "POS_CASH_balance.csv")
    cc        = pd.read_csv(DATA_DIR / "credit_card_balance.csv")
    inst      = pd.read_csv(DATA_DIR / "installments_payments.csv")
    bureaubal = pd.read_csv(DATA_DIR / "bureau_balance.csv")
    return app_train, app_test, bureau, prev, pos, cc, inst, bureaubal


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    num_cols = bureau.select_dtypes(include=[np.number]).columns
    num_cols = num_cols.difference(["SK_ID_CURR", "SK_ID_BUREAU"])

    cat_cols = bureau.select_dtypes(include=["object"]).columns

    bureau_num_agg = bureau.groupby("SK_ID_CURR")[num_cols].agg(["mean", "max", "min", "sum"])
    bureau_num_agg.columns = ["BUREAU_" + "_".join(col).upper() for col in bureau_num_agg.columns]

    bureau_cat = pd.get_dummies(
        bureau[["SK_ID_CURR"] + list(cat_cols)],
        columns=cat_cols
    )
    bureau_cat_agg = bureau_cat.groupby("SK_ID_CURR").mean()
    bureau_cat_agg.columns = ["BUREAU_" + col.upper() for col in bureau_cat_agg.columns]

    bureau_agg = bureau_num_agg.join(bureau_cat_agg, how="left")
    return bureau_agg

def merge_all(
    app_train: pd.DataFrame,
    app_test: pd.DataFrame,
    bureau_agg: pd.DataFrame = None,
    prev_agg: pd.DataFrame = None,
    pos_agg: pd.DataFrame = None,
    cc_agg: pd.DataFrame = None,
    inst_agg: pd.DataFrame = None,
    bureaubal_agg: pd.DataFrame = None,
):
    """Merge all aggregated tables into application train/test on SK_ID_CURR."""
    for df_agg in [bureau_agg, prev_agg, pos_agg, cc_agg, inst_agg, bureaubal_agg]:
        if df_agg is None:
            continue
        app_train = app_train.merge(df_agg, on="SK_ID_CURR", how="left")
        app_test  = app_test.merge(df_agg, on="SK_ID_CURR", how="left")
    return app_train, app_test




def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # don't include TARGET if present
    if "TARGET" in num_cols:
        num_cols.remove("TARGET")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor

def build_datasets():
    app_train, app_test, bureau, prev, pos, cc, inst, bureaubal = load_raw()

    bureau_agg    = aggregate_bureau(bureau)
    # prev_agg      = aggregate_previous_app(prev)
    # pos_agg       = aggregate_pos(pos, prev)
    # cc_agg        = aggregate_credit_card(cc, prev)
    # inst_agg      = aggregate_installments(inst, prev)
    # bureaubal_agg = aggregate_bureau_balance(bureaubal, bureau)

    # pour l’instant, on ne merge que bureau_agg
    train_df, test_df = merge_all(
        app_train,
        app_test,
        bureau_agg,
        prev_agg=None,
        pos_agg=None,
        cc_agg=None,
        inst_agg=None,
        bureaubal_agg=None
    )

    return train_df, test_df


