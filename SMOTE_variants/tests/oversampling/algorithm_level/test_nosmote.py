"""
This module tests the NoSMOTE object
"""

from sm_variants import NoSMOTE
from sm_variants.datasets import load_illustration_2_class


def test_nosmote():
    """
    Testing the NoSMOTE technique
    """

    obj = NoSMOTE()

    assert len(NoSMOTE.parameter_combinations()) == 1

    dataset = load_illustration_2_class()

    assert len(obj.sampling_algorithm(dataset["data"], dataset["target"])[0]) > 0
