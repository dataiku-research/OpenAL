"""
This is where you should define how your sampler is processing
You can find a reminder of the 2 functions 'fit' and 'select_samples' needed inside the sampler class below
"""

import numpy as np
from cardinal.typeutils import RandomStateType
from cardinal.uncertainty import margin_score



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
    def __init__(self, batch_size: int, classifier, iteration: int,
                 strategy: str = 'top', random_state: RandomStateType = None):
        
        self.batch_size = batch_size
        self.classifier = classifier
        self.iteration = iteration
        self.random_state = random_state


    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit the model on labeled samples.
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """

        # Nothing to do, the classifier is already fitted on the selected data!

        return self


    def select_samples(self, X: np.array) -> np.array:
        """Selects the samples from unlabeled data using the internal scoring.
        Args:
            X: Pool of unlabeled samples of shape (n_samples, n_features).
            strategy: Strategy to use to select queries. 
        Returns:
            Indices of the selected samples of shape (batch_size).
        """

        selection_proba = margin_score(self.classifier, X) ** ((self.iteration + 1) * 2)
        selection_proba /= selection_proba.sum()
        np.random.seed(seed=self.random_state)
        index = np.random.choice(X.shape[0], size=self.batch_size, replace=False, p=selection_proba)

        return index

