import argparse
import shutil

from main_run import run_benchmark
from my_sampler import MyCustomSamplerClass

from cardinal.uncertainty import MarginSampler

def get_my_sampler(params : dict) : 
    """
    Function used to instanciate your sampler with it's parameters

    Parameters:
        params : parameters that will be passed to generated your sampler with automatically generated ’batch_size’, ’classifier’, 'iteration' and ’random_state’ according to the selected dataset, current AL iteration and the current seed used
        You can remove these parameters from the initialisation parameters below if they are not used inside your custom sampler
    """

    # TODO : add your custom sampler parameters and remove the default ones if not useful
    # sampler = MyCustomSamplerClass(
    #     # remove some of these parameters if not useful
    #     batch_size = params['batch_size'],
    #     classifier = params['clf'],
    #     iteration = params['iteration'],    # AL iteration
    #     random_state = params['seed'],      # Important for reproducibility purpose (Use it as much as possible)
        
    #     # add you custom parameters here

    # )
    sampler = MarginSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True)
    return sampler


parser = argparse.ArgumentParser()
# parser.add_argument('dataset_id', type=int, help='Dataset to process')
parser.add_argument('-datasets_ids','--list', nargs='*', help='<Required> List of datasets to process', default=[], required=False)
args = parser.parse_args()

# list of strings in ['1461', '1471', '1502', '1590', '40922', '41138', '42395', '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr', 'mnist']
# Leave it empty in order to run all available datasets by default
_, datasets_ids = parser.parse_args()._get_kwargs()[0]
# print(datasets_ids)
del args

#TODO you might need to remove the previous result folder (with the same dataset id) before doing another run of the benchmark
# ! rm -r user_results/results_1461
# shutil.rmtree('user_results/results_1461/', ignore_errors=True)

new_sampler_generator = get_my_sampler
sampler_name = 'my_custom_sampler' #TODO : change the name of your sampler if you want to


run_benchmark(new_sampler_generator=new_sampler_generator, datasets_ids=datasets_ids, sampler_name=sampler_name)