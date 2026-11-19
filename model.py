"""
model.py

Train, validate, and test the RBF-SVM model without data leakage.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from metrics import classification_metrics_summary


def train_validate_test_svm(
    X,
    y,
    C,
    gamma,
    class_names,
    class_weight=None,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
):
    """
    Train SVM using a stratified train-validation-test split.

    The validation set is used during parameter tuning, so the test set remains
    unbiased for final evaluation. StandardScaler is fitted only on the training
    set, then reused to transform validation and test sets. This avoids data
    leakage from validation/test data into the training process.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight)
    model.fit(X_train_scaled, y_train)

    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)

    validation_metrics = classification_metrics_summary(
        y_val,
        y_val_pred,
        class_names,
    )
    test_metrics = classification_metrics_summary(
        y_test,
        y_test_pred,
        class_names,
    )

    return {
        "model": model,
        "scaler": scaler,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "y_val": y_val,
        "y_val_pred": y_val_pred,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }
