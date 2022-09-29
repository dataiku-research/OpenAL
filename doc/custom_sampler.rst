===============================
Sampler with dynamic parameters
===============================


How to define custom dynamic parameters for your sampler ?

In this AL benchmark, there is a possibility for you to define **custom dynamic parameters** that will be used to **instanciate your sampler before each time it is called**.



How to define custom dynamic parameters ?
=========================================

There are already a few dynamic parameters that have been implemented in the benchmark. You will find them inside the ``get_my_sampler`` sampler generator function as defaut parameters for ``MyCustomSamplerClass``.

If you need to implement new ones, you will have to implement them by yourself in the ``main_run.py`` script.


Where do we register our custom dynamic parameters ?
====================================================

Update dynamic params dictionary
--------------------------------

In order for the benchmark to take your dynamic parameters into account, you must define them in the dynamic parameters dictionary ``DYNAMIC_PARAMS`` (inside the ``run_benchmark`` function from the ``main_run.py`` script)

.. code-block:: bash

    DYNAMIC_PARAMS = dict(name_dynamic_parameter = value_dynamic_parameter)


Reminder
--------

Don't forget to register these dynamic parameters inside the ``get_my_sampler`` sampler generator function.

.. code-block:: bash

    def get_my_sampler(params : dict) : 

        sampler = MyCustomSamplerClass(
            name_dynamic_parameter = params['name_dynamic_parameter']
            ... # other static and/or dynamic parameters
        )
