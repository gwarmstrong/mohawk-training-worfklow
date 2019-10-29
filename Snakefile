import os
# from clippers.train_tasks import TrainTask
# import luigi

trial_dir = os.curdir
trial_name = 'runs'
# log1 = os.path.join(trial_dir, trial_name, 'lr_0.0001')
# log2 = os.path.join(trial_dir, trial_name, 'lr_0.01')
log1 = os.path.join(trial_name, 'lr_0.0001')
log2 = os.path.join(trial_name, 'lr_0.01')

localrules: all

rule all:
    input:
        log1, log2

rule train_task:
    input:
        genome_ids = 'params/sample_genome_ids.tsv',
    params:
        model_name = 'ConvNetAvg',
        lr = lambda wildcards: float(wildcards.lr),
    output:
        directory('{log_dir}lr_{lr}')
    run:
        from mohawk.trainer import train_helper as mohawk_cli
        train_args = {'lr': 0.001, #params.lr,
                      'epochs': 10,
                      'log_dir': str(output),
                      'data_dir': '.',
                      'train_ratio': 0.8,
                      'seed': 0,
                      'gpu': False,
                      'batch_size': 64,
                      'append_time': False,
                      'additional_hyper_parameters': None
                      }
        print("log_dir: {}".format(train_args['log_dir']))
        model_name = params.model_name
        genome_ids = input.genome_ids
        mohawk_cli(model_name, genome_ids, **train_args)
