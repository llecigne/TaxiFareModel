# from datetime import datetime, timezone
import numpy as np
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

class CyclicalTransformer(BaseEstimator, TransformerMixin):
    """Compute the cos/sin angle of a cyclical numeric feature bounds within ordinal range."""
    
    def __init__(self, column, range=None):
        self.column = column
        self.range = range
    
    def fit(self, X, y=None):
        if not self.range:
            self.range = range(
                X[self.column].min(), 
                X[self.column].max()+1,
            )
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        angle_step = 2*np.pi/((self.range.stop-self.range.start) / self.range.step)
        angle_index = (X[self.column]-self.range.start) / self.range.step
        X[f'{self.column}_sin'] = np.sin(angle_index * angle_step)
        X[f'{self.column}_cos'] = np.cos(angle_index * angle_step)
        return X[[f'{self.column}_sin', f'{self.column}_cos']]


class DistanceToCenterTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, center=None):
        self.center = center
    
    def fit(self, X, y=None):
        if not self.center:
            self.center = (
                (X['pickup_latitude'].mean()+X['dropoff_latitude'].mean())/2,
                (X['pickup_longitude'].mean()+X['dropoff_longitude'].mean())/2,
            )
        return self
    
    def transform(self, X, y=None):
        # fare barycentre to center
        X['distance_to_center'] = haversine_vectorized(
            pd.DataFrame({
                'lat': (X['pickup_latitude']+X['dropoff_latitude'])/2,
                'lng': (X['pickup_longitude']+X['dropoff_longitude'])/2,
                'center_lat': self.center[0],
                'center_lng': self.center[1],
            }),
            start_lat='lat',
            start_lon='lng',
            end_lat='center_lat',
            end_lon='center_lng',
        )
        return X[['distance_to_center']]
