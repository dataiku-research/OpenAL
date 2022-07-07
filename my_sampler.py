# This is where you should define how your sampler is working
# You can find a reminder of the 2 functions needed inside the sampler class below

import numpy as np
from cardinal.typeutils import RandomStateType



# TODO : implement your custom sampler 

class MyCustomSamplerClass():
    """Abstract Base Class handling query samplers relying on a total order.
    Query sampling methods often scores all the samples and then pick samples
    using these scores. This base class handles the selection system, only
    a scoring method is then required.
    Args:
        batch_size: Numbers of samples to select.
        strategy: Describes how to select the samples based on scores.
        random_state: Random seeding
    """
    def __init__(self, batch_size: int, strategy: str = 'top',
                random_state: RandomStateType = None):
        
        # do some stuff

        return


    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit the model on labeled samples.
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        
        # do some stuff

        return self


    def select_samples(self, X: np.array) -> np.array:
        """Selects the samples from unlabeled data using the internal scoring.
        Args:
            X: Pool of unlabeled samples of shape (n_samples, n_features).
            strategy: Strategy to use to select queries. 
        Returns:
            Indices of the selected samples of shape (batch_size).
        """

        index = []
        # do some stuff

        return index





'''
class MyCustomSamplerClass():
    """Abstract Base Class handling query samplers relying on a total order.
    Query sampling methods often scores all the samples and then pick samples
    using these scores. This base class handles the selection system, only
    a scoring method is then required.
    Args:
        batch_size: Numbers of samples to select.
        strategy: Describes how to select the samples based on scores.
        random_state: Random seeding
    """
    def __init__(self, batch_size: int, strategy: str = 'top',
                random_state: RandomStateType = None):

        # do some stuff
        
        return


    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit the model on labeled samples.
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        
        # do some stuff

        return self


    def select_samples(self, X: np.array) -> np.array:
        """Selects the samples from unlabeled data using the internal scoring.
        Args:
            X: Pool of unlabeled samples of shape (n_samples, n_features).
            strategy: Strategy to use to select queries. 
        Returns:
            Indices of the selected samples of shape (batch_size).
        """

        # do some stuff

        return index
'''