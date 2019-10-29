import os
import luigi
from mohawk.trainer import train_helper as mohawk_cli


class TrainTask(luigi.Task):
    model_name = luigi.Parameter()
    genome_ids = luigi.Parameter()
    learning_rate = luigi.Parameter(default=0.0001)
    n_epochs = luigi.Parameter(default=100)
    log_dir = luigi.Parameter(default='runs')
    data_dir = luigi.Parameter(default='.')
    train_ratio = luigi.Parameter(default=0.8)
    seed = luigi.Parameter(default=0)
    gpu = luigi.Parameter(default=False)
    batch_size = luigi.Parameter(default=64)
    append_time = luigi.Parameter(default=False)
    hparams = luigi.Parameter(default=None)

    def run(self):
        train_args = {'lr': self.learning_rate,
                      'epochs': self.n_epochs,
                      'log_dir': self.log_dir,
                      'data_dir': self.data_dir,
                      'train_ratio': self.train_ratio,
                      'seed': self.seed,
                      'gpu': self.gpu,
                      'batch_size': self.batch_size,
                      'append_time': False,
                      'additional_hyper_parameters': self.hparams
                      }
        model_name = self.model_name
        genome_ids = self.genome_ids
        mohawk_cli(model_name, genome_ids, **train_args)

    def output(self):
        return luigi.LocalTarget(self.log_dir)


class TrainGridTask(luigi.Task):

    trial_dir = luigi.Parameter(default=os.curdir)
    trial_name = luigi.Parameter(default='runs')

    def run(self):
        with self.output().open('w') as fp:
            fp.write('done')

    def requires(self):
        genome_ids = '../mohawk/mohawk/tests/data/sample_genome_ids.tsv'
        model_name = 'ConvNetAvg'
        return [TrainTask(model_name=model_name,
                          genome_ids=genome_ids,
                          learning_rate=0.0001,
                          log_dir=os.path.join(str(self.trial_dir),
                                               str(self.trial_name),
                                               'lr_0.0001'),
                          n_epochs=10),
                TrainTask(model_name=model_name,
                          genome_ids=genome_ids,
                          learning_rate=0.01,
                          log_dir=os.path.join(str(self.trial_dir),
                                               str(self.trial_name),
                                               'lr_0.01'),
                          n_epochs=10)
                ]

    def output(self):
        return luigi.LocalTarget(os.path.join(str(self.trial_dir),
                                              str(self.trial_name), '.done'))

