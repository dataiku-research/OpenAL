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
        params['batch_size'],
        params['clf'],
        params['iteration'],    # AL iteration
        random_state = params['seed'],      # Important for reproducibility purpose (Use it as much as possible)
        
        # add you custom parameters here

    )
    return sampler


# python main.py -datasets_ids 1461 1471 1502 1590 40922 41138 42395 43439 43551 42803 41162 cifar10 cifar10_simclr mnist

parser = argparse.ArgumentParser()
parser.add_argument('dataset_ids', type=str, nargs='*', help='List of datasets to process', default=[])
args = parser.parse_args()

# list of strings in ['1461', '1471', '1502', '1590', '40922', '41138', '42395', '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr', 'mnist']
# Leave it empty in order to run all available datasets by default
args = parser.parse_args()
datasets_ids = args.dataset_ids
del args

sampler_name = 'my_custom_sampler' #TODO : change the name of your sampler if you want to


run_benchmark(new_sampler_generator=get_my_sampler, datasets_ids=datasets_ids, sampler_name=sampler_name)