from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from clustering import FEATURES


def split_data(df):
    X = df[FEATURES]
    y = df["risk_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_models():
    models = {
        "Logistic Regression": Pipeline([
            ("sampler", RandomOverSampler(random_state=42)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ]),

        "Random Forest": Pipeline([
            ("sampler", RandomOverSampler(random_state=42)),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=42))
        ]),

        "SVM": Pipeline([
            ("sampler", RandomOverSampler(random_state=42)),
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]),

        "KNeighbors": Pipeline([
            ("sampler", RandomOverSampler(random_state=42)),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
    }

    return models


def train_models(models, X_train, y_train):
    trained_models = {}

    for name in models:
        clf = models[name]
        clf.fit(X_train, y_train)
        trained_models[name] = clf

    return trained_models


def tune_logistic_regression(X_train, y_train):
    param_grid_lr = {
        "model__C": [0.01, 0.1, 1, 10, 100]
    }

    lr_search = GridSearchCV(
        Pipeline([
            ("sampler", RandomOverSampler(random_state=42)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ]),
        param_grid_lr,
        cv=5,
        scoring="f1_weighted"
    )

    lr_search.fit(X_train, y_train)

    return lr_search