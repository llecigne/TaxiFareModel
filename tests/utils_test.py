import pandas as pd

from sklearn.linear_model import LinearRegression

from TaxiFareModel.utils import compute_rmse, rmse_scorer

def test_compute_rmse():
    assert compute_rmse(
        pd.Series([1, 2]), 
        pd.Series([2, 4]), 
    ) == 2.5**0.5
