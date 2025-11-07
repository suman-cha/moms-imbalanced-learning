"""
Testing the CondensedNN.
"""

from sm_variants.noise_removal import CondensedNearestNeighbors

from sm_variants.datasets import load_all_min_noise, load_separable


def test_specific():
    """
    Oversampler specific testing
    """

    obj = CondensedNearestNeighbors()

    dataset = load_all_min_noise()

    X_samp, _ = obj.remove_noise(dataset["data"], dataset["target"])

    assert len(X_samp) > 0

    dataset = load_separable()

    X_samp, _ = obj.remove_noise(dataset["data"], dataset["target"])

    assert len(X_samp) > 0
