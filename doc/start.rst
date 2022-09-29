====================================
Getting started
====================================


How to use the benchmark and evaluate your own sampler



Installation
============

The first step is to **fork** or clone the benchmark GitHub repository.

.. code-block:: bash

    $ git clone https://github.com/dataiku-research/benchmark_tabular_active_learning.git


Then you have to install all the required packages used inside the benchmark.

*Please run this command when you are located at the root of the benchmark folder*

.. code-block:: bash

    $ pip install -r requirement.txt


Implement your custom sampler
=============================

Sampler architecture
--------------------

Your sampler should be defined inside ``my_sampler.py`` file inside the ``MyCustomSamplerClass`` class.

The sampler you want to evaluate in this benchmark must follow this architecture, implementing the fit and select samples method :

 .. code-block:: bash

    class MyCustomSamplerClass():
        """
        Args:
            batch_size: Numbers of samples to select.
            random_state: Random seeding
        """
        def __init__(self, batch_size: int, random_state: RandomStateType = None):

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



Sampler input parameters
------------------------

Your custom sampler parameters should be defined inside ``main.py`` file or ``main.ipynb`` file inside the ``get_my_sampler`` function. 
Everything is already imported so that you just have to manage your sampler input parameters defined inside this function. 
Your can either add your custom sampler parameters and remove the dynamic input parameters already implemented.

This function will be used later in the benchmark in order to instanciate your sampler with it's custom parameters, and additional dynamic parameters if needed.

.. code-block:: bash

    def get_my_sampler(params : dict) : 
        """
        Function used to instanciate your sampler with it's parameters

        Parameters:
            params : parameters that will be passed to generated your sampler with automatically generated ’batch_size’, ’classifier’, 'iteration' and ’random_state’ according to the selected dataset, current AL iteration and the current seed used
            You can remove these parameters from the initialisation parameters below if they are not used inside your custom sampler
        """

        # TODO : add your custom sampler parameters and remove the default ones if not useful
        sampler = MyCustomSamplerClass(
            # remove some of these parameters if not useful
            batch_size = params['batch_size'],
            classifier = params['clf'],
            iteration = params['iteration'],    # AL iteration
            random_state = params['seed'],      # Important for reproducibility purpose (Use it as much as possible)
            
            # add you custom parameters here

        )

        return sampler

About available dynamic parameters :

- ``batch_size`` refers to the sampling batch size of the sampler. It could be automatically generated according to the selected dataset.
- ``clf`` refers to the estimator of the sampler. It could be automatically generated according to the selected dataset.
- ``iteration`` refers to the current AL iteration.
- ``seed`` refers to the current seed used. As it is really important for reproducibility purpose, you should use this parameter inside your sampler as much as possible.



Run the benchmark
=================

After you properly defined your custom sampler as shown below, there are 2 possible ways to run the benchmark, depending on the file in which you choosed to define your sampler input parameters.

If you choosed to define your input parameters inside the ``main.ipynb`` file, you can run the benchmark running the notebook cells.

On the other hand, if you choosed to define your input parameters inside the ``main.py`` file, you can run the benchmark typing the command below from your terminal (inside the root of the benchmark folder).

.. code-block:: bash

    python main.py -datasets_ids [list of datasets ids you want to run]

    # Example :
    # python main.py -datasets_ids 1461 cifar10

**Note:** If you want to run all the benchmark datasets in a row, you can leave datasets_ids argument empty

.. code-block:: bash

    python main.py -datasets_ids

Save your results
=================

After you ran the benchmark for the dataset, a window will automatically pop-up and you will have the possibility to merge your sampler results inside benchmark results.

If you accept to share your results to the AL community, you just need to **create a Git Pull Request in the main repository** so that your experiments could be verified and merged into the main repository. 

Then your results would be available to everyone
