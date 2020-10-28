import tensorflow as tf

import config
import models
from input_data import AudioWrapper
from helper import Trainer, Evaluator


def train(args):
    is_training = True

    session = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
    dataset = AudioWrapper(args, 'train', is_training, session)
    wavs, labels = dataset.get_input_and_output_op()

    model = models.__dict__[args.arch](args)
    model.build(wavs=wavs, labels=labels, is_training=is_training)

    trainer = Trainer(model, session, args, dataset)
    trainer.train()

def evaluate(args):
    is_training = False

    session = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
    dataset = AudioWrapper(args, args.dataset_name, is_training, session)
    wavs, labels = dataset.get_input_and_output_op()

    model = models.__dict__[args.arch](args)
    model.build(wavs=wavs, labels=labels, is_training=is_training)

    evaluator = Evaluator(model, session, args, dataset)
    evaluator.evaluate()

if __name__ == "__main__":
    args = config.arg_config()
    if args.mod == 'train':
        train(args)
    else:
        evaluate(args)
