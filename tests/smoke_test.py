import pandas as pd
import numpy as np
import pytest
import random
from sklearn.linear_model import LinearRegression
from src.autora.experimentalist.autora_experimentalist_example import sample

@pytest.fixture
def seed():
    """
    Ensures that the results are the same each time the tests are run.
    """
    random.seed(180)
    np.random.seed(180)
    return

def test_experimentalist_smoke_test(seed):
    """
    Tests that the experimentalist can be called with basic inputs
    and returns a DataFrame of the correct shape. The seed fixture
    ensures any random processes within the sampler are deterministic.
    """
    # 1. Define a simple design space (pool of conditions)
    conditions = pd.DataFrame({"x1": np.linspace(0, 10, 100)})

    # 2. Define a mock theorist model
    model = LinearRegression()

    # 3. Define some dummy experimental history
    reference_conditions = pd.DataFrame({"x1": [1, 5, 9]})

    # 4. Call the sampler to get one new sample
    new_conditions = sample(
        conditions=conditions,
        models=[model],
        reference_conditions=reference_conditions,
        num_samples=1
    )

    # 5. Assertions
    assert isinstance(new_conditions, pd.DataFrame), "Output should be a pandas DataFrame."
    assert new_conditions.shape[0] == 1, "The number of returned samples is incorrect."
    assert "x1" in new_conditions.columns, "The output DataFrame is missing the 'x1' column."

