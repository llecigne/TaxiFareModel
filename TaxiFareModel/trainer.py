import joblib
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from TaxiFareModel.encoders import CyclicalTransformer, TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse, rmse_scorer

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[FR] [Bordeaux] [llecigne] Taxi Fare Model 3"  

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
            ColumnTransformer([
                ('cyclical-dow', CyclicalTransformer('dow', range(7)), ['dow']),
                ('cyclical-hour', CyclicalTransformer('hour', range(24)), ['hour']),
                ('cyclical-month', CyclicalTransformer('month', range(1, 13)), ['month']),
                ('standard-year', StandardScaler(), ['year']),
            ]),
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

        self.pipeline = Pipeline([
            ('feat_eng', feat_eng_bloc),
            ('regressor', RandomForestRegressor())
        ])
        self._mlflow_log_model()

    def preprocessing_pipeline(self):
        '''return the pipeline used to preprocessd data'''
        if not self.pipeline:
            self.set_pipeline()
        return self.pipeline['feat_eng']

    def preprocess(self):
        '''return a preprocessed dataframe'''
        return self.preprocessing_pipeline().fit_transform(self.X, self.y)

    def run(self):
        """set and train the pipeline"""
        if not self.pipeline:
            self.set_pipeline()
        return self.pipeline.fit(self.X, self.y)

    def cross_validate(self):
        """cross validate the pipeline"""
        if not self.pipeline:
            self.set_pipeline()
        cv = cross_validate(
            self.pipeline, 
            X, y, 
            return_train_score=True, 
            scoring=rmse_scorer(),
            n_jobs=-1,
        )
        import ipdb; ipdb.set_trace()
        self.mlflow_log_metric('cv_test_rmse_mean', cv['test_score'].mean())
        self.mlflow_log_metric('cv_test_rmse_std', cv['test_score'].std())
        self.mlflow_log_metric('cv_train_rmse_mean', cv['train_score'].mean())
        self.mlflow_log_metric('cv_train_rmse_std', cv['train_score'].std())
        return cv

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        assert self.pipeline is not None
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_test, y_pred)
        self.mlflow_log_metric('rmse', rmse)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        assert self.pipeline is not None
        joblib.dump(self.pipeline, 'model.joblib')

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def _mlflow_log_model(self):
        assert self.pipeline is not None
        model = self.pipeline['regressor']
        self.mlflow_log_param('model',  model.__class__())
        for k, v in model.get_params().items():
            self.mlflow_log_param(k, v)
            
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from TaxiFareModel.data import get_data, clean_data
    data = clean_data(get_data())
    target = 'fare_amount'
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    trainer = Trainer(X_train, y_train)
    cv = trainer.cross_validate()
    history = trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f'Model RMSE is {rmse:.2f}$')
    trainer.save_model()
