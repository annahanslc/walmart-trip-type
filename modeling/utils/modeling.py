import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_model(X_train_proc, y_train, X_test_proc, y_test):
    ####### Logistic Regression ######
    print("Running Logistic Regression...")
    # Train model
    model_lr = LogisticRegression(solver="lbfgs", max_iter=1000)

    # Cross validate train
    y_proba = cross_val_predict(
        model_lr, X_train_proc, y_train, cv=3, method="predict_proba"
    )
    lr_train_loss = log_loss(y_train, y_proba)

    # Predict on test
    model_lr.fit(X_train_proc, y_train)
    y_proba = model_lr.predict_proba(X_test_proc)
    lr_test_loss = log_loss(y_test, y_proba)
    lr_test_loss

    ####### Random Forest ######
    print("Running Random Forest...")
    # Train model
    model_rf = RandomForestClassifier(max_depth=10, n_estimators=50)

    # Cross validate train
    y_proba = cross_val_predict(
        model_rf, X_train_proc, y_train, cv=3, method="predict_proba"
    )
    rf_train_loss = log_loss(y_train, y_proba)

    # Predict on test
    model_rf.fit(X_train_proc, y_train)
    y_proba = model_rf.predict_proba(X_test_proc)
    rf_test_loss = log_loss(y_test, y_proba)
    rf_test_loss

    ####### XGBoost ######
    print("Running XGBoost...")
    # Train model
    model_xgb = XGBClassifier(
        objective="multi:softprob", num_class=38, eval_metric="mlogloss"
    )

    # Cross validate train
    y_proba = cross_val_predict(
        model_xgb, X_train_proc, y_train, cv=3, method="predict_proba"
    )
    xgb_train_loss = log_loss(y_train, y_proba)

    # Predict on test
    model_xgb.fit(X_train_proc, y_train)
    y_proba = model_xgb.predict_proba(X_test_proc)
    xgb_test_loss = log_loss(y_test, y_proba)
    xgb_test_loss

    # lr_results = ["LogReg", lr_train_loss, lr_test_loss]
    # rf_results = ["RandomForest", rf_train_loss, rf_test_loss]
    # xgb_results = ["XGB", xgb_train_loss, xgb_test_loss]

    results = [
        {"Model": "LogReg", "Train Loss": lr_train_loss, "Test Loss": lr_test_loss},
        {
            "Model": "RForest",
            "Train Loss": rf_train_loss,
            "Test Loss": rf_test_loss,
        },
        {"Model": "XGBoost", "Train Loss": xgb_train_loss, "Test Loss": xgb_test_loss},
    ]
    results_df = pd.DataFrame(results)

    return results_df
