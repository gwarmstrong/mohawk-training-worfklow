import os
from itertools import product

trial_dir = os.curdir
trial_name = 'runs'
# log1 = os.path.join(trial_dir, trial_name, 'lr_0.0001')
# log2 = os.path.join(trial_dir, trial_name, 'lr_0.01')
log1 = os.path.join(trial_name, 'lr_0.0001')
log2 = os.path.join(trial_name, 'lr_0.01')

RED, N_GENOME, GROUP = glob_wildcards('params/min-red_{red}__n_{n}__group_{group}__genome-ids.txt')

mod_logs = expand('runs/min-red_{red}__n_{n}__group_{group}__lr_{{lr}}__arch_{{arch}}',
                  zip, red=RED, n=N_GENOME, group=GROUP)

architectures = ['ConvNetAvg', 'ConvNetAvg5', 'ConvNetAvg6']
lrs = [0.01, 0.001, 0.0001, 0.00001]

all_logs = [log_.format(lr=lr, arch=arch) for log_, arch, lr in product(mod_logs, architectures, lrs)]

localrules: all

rule all:
    input:
        all_logs

rule train_task:
    input:
        genome_ids = 'params/min-red_{red}__n_{n}__group_{group}__genome-ids.txt',
    params:
        model_name = lambda wildcards: wildcards.arch,
        lr = lambda wildcards: float(wildcards.lr),
    output:
        directory('{log_dir}min-red_{red}__n_{n}__group_{group}__lr_{lr}__arch_{arch}')
    run:
        from mohawk.trainer import train_helper as mohawk_cli
        train_args = {'lr': wildcards.lr,
                      'epochs': 10,
                      'log_dir': str(output),
                      'data_dir': 'data',
                      'train_ratio': 0.8,
                      'seed': 0,
                      'gpu': False,
                      'batch_size': 64,
                      'append_time': False,
                      'additional_hyper_parameters': {'min-red': wildcards.red,
                                                      'n-classes': wildcards.n
                                                      }
                      }
        print("log_dir: {}".format(train_args['log_dir']))
        model_name = params.model_name
        genome_ids = input.genome_ids
        mohawk_cli(model_name, genome_ids, **train_args)
