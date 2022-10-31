from bench.experiment import load_experiment, load_initial_conditions, run
from bench.utils import REFERENCE_METHODS as methods


if __name__ == '__main__':
    dataset_ids = ['1461', '1471', '1502', '1590', '40922', '41138', '42395', '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr', 'mnist']

    for dataset_id in dataset_ids:
        initial_conditions = load_initial_conditions(dataset_id)
        experimental_parameters = load_experiment(dataset_id, initial_conditions)
        run(experimental_parameters, methods)
