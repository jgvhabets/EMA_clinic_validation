
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline




def fit_cv_regr(X_train, y_train):
    
    # Pipeline: scaling + ElasticNet
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=10000, random_state=42))
    ])

    # Hyperparameter grid
    param_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    grid_rg = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_rg.fit(X_train, y_train)

    best_model = grid_rg.best_estimator_

    return best_model

