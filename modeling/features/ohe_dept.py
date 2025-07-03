import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import set_config

set_config(transform_output="pandas")


# OHE the Department
def ohe_dept(df, test=False, transformer=None):
    if test:
        return transformer.transform(df)

    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformer = ColumnTransformer(
            [
                (
                    "ohe",
                    ohe,
                    ["DepartmentDescription"],
                )
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        df_ohe_dept = transformer.fit_transform(df)
        return df_ohe_dept, transformer


def ohe_dept_groupby(ohe_dept_df):
    # Multiply all ohe department by the ScanCount
    ohe_features = [
        feature
        for feature in ohe_dept_df.columns.to_list()
        if "DepartmentDescription_" in feature
    ]

    new_df = pd.DataFrame(ohe_dept_df["VisitNumber"])
    for feature in ohe_features:
        new_df[feature] = ohe_dept_df[feature] * ohe_dept_df["ScanCount"]

    # Groupby OHE dept by VisitNumber, totaling the scancount per department
    new_df_groupby = new_df.groupby("VisitNumber").sum()
    new_df_groupby

    return new_df_groupby
