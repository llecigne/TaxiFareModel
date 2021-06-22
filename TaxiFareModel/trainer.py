from TaxiFareModel.data import clean_data
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_time = make_pipeline(
            TimeFeaturesEncoder(column_name='pickup_datetime'),
            StandardScaler()
        )
        pipe_distance = make_pipeline(
            DistanceTransformer(), 
            RobustScaler()
        )

        dist_cols = [
            'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude'
        ]
        time_cols = ['pickup_datetime']

        feat_eng_bloc = ColumnTransformer([
            ('time', pipe_time, time_cols),
            ('distance', pipe_distance, dist_cols)
        ])

        # workflow
        self.pipeline = Pipeline([
            ('feat_eng_bloc', feat_eng_bloc),
            ('regressor', RandomForestRegressor())
        ])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        return self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        assert self.pipeline is not None
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from TaxiFareModel.data import get_data, clean_data
    data = clean_data(get_data())
    target = 'fare_amount'
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    trainer = Trainer(X_train, y_train)
    history = trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f'Model RMSE is {rmse:.2f}$')
