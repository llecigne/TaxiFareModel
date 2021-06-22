import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal

from TaxiFareModel.encoders import CyclicalTransformer

def test_cyclical_encoder_should_supply_cossin_coord():
    df = pd.DataFrame({
        'day': [0, 1, 2, 5, 6],
    })
    enc = CyclicalTransformer(column='day', range=range(7))
    df_enc = enc.fit_transform(df)
    assert 'day_cos' in df_enc.columns
    assert 'day_sin' in df_enc.columns

def test_cyclical_encoder_should_supply_cossin_coord():
    df = pd.DataFrame({
        'day': [0, 1, 2, 5, 6],
    })
    enc = CyclicalTransformer(column='day', range=range(7))
    df_enc = enc.fit_transform(df)
    assert_array_equal(
        df_enc.day_sin,
        np.sin(df.day / 7 * 2 * np.pi),
    )
    assert_array_equal(
        df_enc.day_cos,
        np.cos(df.day / 7 * 2 * np.pi),
    )
