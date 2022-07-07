import argparse

from main_run import run_benchmark
from my_sampler import MyCustomSamplerClass


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


parser = argparse.ArgumentParser()
# parser.add_argument('dataset_id', type=int, help='Dataset to process')
parser.add_argument('dataset_id', help='Dataset to process')
args = parser.parse_args()
dataset_id = args.dataset_id
del args

new_sampler_generator = get_my_sampler

#TODO you might need to remove the previous result folder (with the same dataset id) before doing another run of the benchmark
# ! rm -r results_1461

sampler_name = 'my_custom_sampler' #TODO : change the name of your sampler if you want to

run_benchmark(dataset_id, new_sampler_generator=new_sampler_generator, sampler_name=sampler_name)