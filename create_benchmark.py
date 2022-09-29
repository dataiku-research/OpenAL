from bench.experiment import create_experiment, create_initial_conditions


for dataset_id in [#'1471', '1502', '1590', '40922', '41138',
                   #'42395', '43439', '43551', '42803', '41162',
                   'cifar10', 'cifar10_simclr', 'mnist']:

    ic = create_initial_conditions(dataset_id, dataset_id, 0.001, 10)
    ep = create_experiment(dataset_id, dataset_id, 0.001, 10)
