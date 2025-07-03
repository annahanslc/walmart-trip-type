def groupby_visitnumber(df):
    # Group by VisitNumber to get features and target (triptype)
    df = df.groupby("VisitNumber").agg(
        triptype=("TripType", "first"),
        weekday=("Weekday", "first"),
        num_unique_upc=("Upc", "nunique"),
        avg_scancount=("ScanCount", "mean"),
        total_scancount=("ScanCount", "sum"),
        num_unique_dept=("DepartmentDescription", "nunique"),
        num_unique_fileline=("FinelineNumber", "nunique"),
    )
    df.reset_index(inplace=True)
    return df


def groupby_visitnumber_kaggle(df):
    # Group by VisitNumber to get features and target (triptype)
    df = df.groupby("VisitNumber").agg(
        weekday=("Weekday", "first"),
        num_unique_upc=("Upc", "nunique"),
        avg_scancount=("ScanCount", "mean"),
        total_scancount=("ScanCount", "sum"),
        num_unique_dept=("DepartmentDescription", "nunique"),
        num_unique_fileline=("FinelineNumber", "nunique"),
    )
    df.reset_index(inplace=True)
    return df
