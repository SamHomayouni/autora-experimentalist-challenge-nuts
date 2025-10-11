import pandas as pd
import numpy as np
import pytest
import random
from src.autora.experimentalist.autora_experimentalist_example import sample

@pytest.fixture
def seed():
    """
    Ensures that the results are the same each time the tests are run.
    """
    random.seed(180)
    np.random.seed(180)
    return

def test_experimentalist_novelty_logic(seed):
    """
    Tests that when no models are provided, the experimentalist
    chooses the point farthest away from existing data (the novelty strategy).
    The seed fixture is included for consistency, although this specific test's
    logic should be deterministic.
    """
    # 1. Define a design space where one point (10.0) is clearly the most novel
    conditions = pd.DataFrame({"x1": np.linspace(0, 10, 11)})

    # 2. Define experimental history clustered at the low end
    reference_conditions = pd.DataFrame({"x1": [0.0, 0.5, 1.0]})

    # 3. Call the sampler with NO models, forcing it to use a non-model-based strategy.
    #    Given the setup, this should default to novelty.
    new_conditions = sample(
        conditions=conditions,
        models=[],  # No models provided
        reference_conditions=reference_conditions,
        num_samples=1
    )

    # 4. Assertion
    # The most novel point is 10.0, as it is farthest from the [0, 0.5, 1.0] cluster.
    assert new_conditions["x1"].iloc[0] == 10.0, \
        "When no models are provided, the sampler should select the most novel point."

