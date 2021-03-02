from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error

from lightgbm import LGBMRegressor, LGBMClassifier


class LGBMEarlyStopping(BaseEstimator):
    def __init__(
        self,
        early_stopping_rounds=5,
        test_size=0.1,
        eval_metric="mae",
        **estimator_params
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = test_size
        self.eval_metric = eval_metric = "mae"
        if self.estimator is not None:
            self.set_params(**estimator_params)

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def get_params(self, **params):
        return self.estimator.get_params()

    def fit(self, X, y):
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size
        )
        self.estimator.fit(
            x_train,
            y_train,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric=self.eval_metric,
            eval_set=[(x_val, y_val)],
        )
        return self

    def predict(self, X):
        return self.estimator.predict(X)


class LGBMRegressorEarlyStopping(LGBMEarlyStopping):
    def __init__(self, *args, **kwargs):
        self.estimator = LGBMRegressor()
        super(LGBMRegressorEarlyStopping, self).__init__(*args, **kwargs)


class LGBMClassifierEarlyStopping(LGBMEarlyStopping):
    def __init__(self, *args, **kwargs):
        self.estimator = LGBMClassifier()
        super(LGBMClassifierEarlyStopping, self).__init__(*args, **kwargs)


mse = make_scorer(
    score_func=lambda y, y_pred: mean_squared_error(y, y_pred, squared=False),
    greater_is_better=False,
)
