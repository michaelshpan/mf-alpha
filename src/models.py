from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV

@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_grid: Dict[str, Any] | None
    cv_folds: int = 5


def get_model_specs(cfg) -> Dict[str, ModelSpec]:
    specs = {}

    specs["ols"] = ModelSpec(
        name="ols", estimator=LinearRegression(), param_grid=None, cv_folds=cfg["elastic_net"]["cv_folds"]
    )

    specs["enet"] = ModelSpec(
        name="enet",
        estimator=ElasticNet(max_iter=10000),
        param_grid={
            "l1_ratio": cfg["elastic_net"]["l1_ratio_grid"],
            "alpha": cfg["elastic_net"]["alpha_grid"],
        },
        cv_folds=cfg["elastic_net"]["cv_folds"],
    )

    specs["rf"] = ModelSpec(
        name="rf",
        estimator=RandomForestRegressor(
            n_estimators=cfg["random_forest"]["n_estimators"],
            max_features=cfg["random_forest"]["max_features"],
            n_jobs=-1,
            random_state=cfg["experiment"]["seed"],
        ),
        param_grid=None,
        cv_folds=cfg["random_forest"]["cv_folds"],
    )

    specs["xgb"] = ModelSpec(
        name="xgb",
        estimator=XGBRegressor(
            n_estimators=cfg["xgb"]["n_estimators"],
            max_depth=cfg["xgb"]["max_depth"],
            learning_rate=cfg["xgb"]["learning_rate"],
            subsample=cfg["xgb"]["subsample"],
            colsample_bytree=cfg["xgb"]["colsample_bytree"],
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=cfg["experiment"]["seed"],
            verbosity=0,
        ),
        param_grid=None,
        cv_folds=cfg["xgb"]["cv_folds"],
    )
    return specs


def fit_and_predict(spec: ModelSpec, X_train, y_train, X_test, seed: int, return_model: bool = False):
    if spec.param_grid:
        cv = KFold(n_splits=spec.cv_folds, shuffle=True, random_state=seed)
        gs = GridSearchCV(spec.estimator, spec.param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_train, y_train)
        best_est = gs.best_estimator_
    else:
        best_est = spec.estimator
        best_est.fit(X_train, y_train)
    y_hat = best_est.predict(X_test)
    
    if return_model:
        return y_hat, best_est
    return y_hat
