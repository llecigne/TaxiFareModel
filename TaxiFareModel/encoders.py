# from datetime import datetime, timezone
import pandas as pd
# from pandas._libs.tslibs import Hour
# from pandas.core.base import DataError
from sklearn.base import BaseEstimator, TransformerMixin

from TaxiFareModel.utils import haversine_vectorized

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""

    def __init__(self, column_name, timezone_name='America/New_York'):
        self.column_name = column_name
        self.timezone_name = timezone_name
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X.index = pd.to_datetime(X[self.column_name])
        X.index.tz_convert(self.timezone_name)
        X['dow'] = X.index.weekday
        X['hour'] = X.index.hour
        X['month'] = X.index.month
        X['year'] = X.index.year
        return X[['dow', 'hour', 'month', 'year']].reset_index(drop=True)

class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['distance'] = haversine_vectorized(X)
        return X[['distance']]