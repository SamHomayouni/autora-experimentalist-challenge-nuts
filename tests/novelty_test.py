import pandas as pd
import numpy as np
import pytest
import random
from sklearn.linear_model import LinearRegression  # Import a dummy model
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
    Tests that the experimentalist chooses the most novel point
    when model-based signals are uninformative.
    """
    # 1. Define a design space where one point (10.0) is clearly the most novel
    conditions = pd.DataFrame({"x1": np.linspace(0, 10, 11)})

    # 2. Define experimental history clustered at the low end
    reference_conditions = pd.DataFrame({"x1": [0.0, 0.5, 1.0]})

    # 3. Call the sampler with a DUMMY model.
    # This prevents the sampler from defaulting to the random_sample branch.
    # Since a dummy model provides no variance, falsification/uncertainty
    # scores will be ~0, making the novelty strategy the winner.
    new_conditions = sample(
        conditions=conditions,
        models=[LinearRegression()],  # Provide a dummy model
        reference_conditions=reference_conditions,
        num_samples=1,
        random_state=180 # Pass the seed for deterministic behavior in sampler
    )

    # 4. Assertion
    # The most novel point is 10.0, as it is farthest from the [0, 0.5, 1.0] cluster.
    assert np.isclose(new_conditions["x1"].iloc[0], 10.0), \
        "The sampler should select the most novel point when model signals are uninformative."

