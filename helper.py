import tensorflow as tf
import tensorflow.contrib.slim as slim

import os
import time


def getTimeString():
    timeStamp = int(time.time())
    timeArray = time.localtime(timeStamp)
    timeString = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

    return timeString


class Base:
    def __init__(self, model, session, args, dataset):
        self.model = model
        self.session = session
        self.args = args
        self.dataset = dataset

        self.init_base()

    def init_base(self):
        self.label_names = self.dataset.prepare_words_list
        assert len(self.label_names) == self.args.num_classes
        self._saver = None

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.compat.v1.train.Saver(
                var_list=self.model.model_variables, max_to_keep=self.args.max_to_keep)
        return self._saver
    
    def display_model(self):
        list_v = slim.get_variables_to_restore()
        sum_params = 0
        for v in list_v:
            params = 1
            for dim in v.shape:
                params *= dim
            sum_params += params
            print(v)
        print('total parameter number: {}'.format(sum_params))

    def routine_restore_and_initialize(self):
        self.session.run(tf.compat.v1.global_variables_initializer())
        self.session.run(
            tf.compat.v1.local_variables_initializer())  # for metrics
        if self.args.checkpoint_path:
            self.saver.restore(self.session, self.args.checkpoint_path)
            print('restore from {}'.format(self.args.checkpoint_path))
        else:
            print('from init model')


class Trainer(Base):
    def __init__(self, model, session, args, dataset):
        super().__init__(model, session, args, dataset)

        self.display_model()
        self.init_trainer()
        self.routine_restore_and_initialize()

    def init_trainer(self):
        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder)

        self.global_step_from_checkpoint = self.get_global_step_from_checkpoint(
            self.args.checkpoint_path)
        self.global_step = tf.Variable(
            self.global_step_from_checkpoint, name="global_step", trainable=False)

        self.boundaries = self.args.boundaries
        self.learning_rate_placeholder = tf.compat.v1.train.piecewise_constant(
            self.global_step, self.boundaries, self.args.lr_list
        )

        self.optimizer = self.build_optimizer(self.args.optimizer,
                                              learning_rate=self.learning_rate_placeholder,
                                              momentum=self.args.momentum)
        self.train_op = self.build_train_op(total_loss=self.model.total_loss,
                                            optimizer=self.optimizer,
                                            global_step=self.global_step)

    def get_global_step_from_checkpoint(self, checkpoint_path):
        """It is assumed that `checkpoint_path` is path to checkpoint file, not path to directory
        with checkpoint files.
        In case checkpoint path is not defined, 0 is returned."""
        if checkpoint_path is None or checkpoint_path == "":
            return 0
        else:
            if "-" in checkpoint_path:
                return int(checkpoint_path.split("-")[-1])
            else:
                return 0

    def build_optimizer(self, optimizer, learning_rate, momentum=None):
        kwargs = {
            "learning_rate": learning_rate
        }

        if optimizer == "gd":
            opt = tf.train.GradientDescentOptimizer(**kwargs)
            print("Use GradientDescentOptimizer")
        elif optimizer == "adam":
            opt = tf.compat.v1.train.AdamOptimizer(**kwargs)
            print("Use AdamOptimizer")
        elif optimizer == "mom":
            if momentum:
                kwargs["momentum"] = momentum
            opt = tf.train.MomentumOptimizer(**kwargs)
            print("Use MomentumOptimizer")
        elif optimizer == "rmsprop":
            opt = tf.train.RMSPropOptimizer(**kwargs)
            print("Use RMSPropOptimizer")
        else:
            print("Unknown optimizer: {}".format(optimizer))
            raise NotImplementedError
        return opt

    def build_train_op(self, total_loss, optimizer, global_step):
        variables_to_train = tf.compat.v1.trainable_variables()

        if variables_to_train:
            train_op = slim.learning.create_train_op(
                total_loss,
                optimizer,
                global_step=global_step,
                variables_to_train=variables_to_train,
            )
        else:
            print("Empty variables_to_train")
            train_op = tf.no_op()

        return train_op

    def train(self):
        print("Training started from {} step(s)".format(
            self.global_step_from_checkpoint))

        for global_step in range(self.global_step_from_checkpoint, self.args.steps):
            # Session.Run!
            fetch_vals = self.session.run(
                {
                    'train_op': self.train_op,
                    'acc': self.model.acc,
                    'total_loss': self.model.total_loss,
                    'model_loss': self.model.model_loss,
                }
            )

            # Logging
            if (global_step+1) % self.args.step_logging == 0:
                print("{}  [{:5d}/{}], Acc: {:.4f}, total loss: {:.6f}, model loss: {:.6f}".format(
                    getTimeString(), global_step + 1,
                    self.args.steps, fetch_vals['acc'] * 100,
                    fetch_vals['total_loss'], fetch_vals['model_loss']
                ))

            # Save
            if (global_step+1) % self.args.step_save_checkpoint == 0:
                self.saver.save(self.session, os.path.join(self.args.save_folder, self.args.arch),
                                global_step=global_step+1)

        print("Training finished!")


class Evaluator(Base):
    def __init__(self, model, session, args, dataset):
        super().__init__(model, session, args, dataset)

        self.display_model()
        self.routine_restore_and_initialize()

    def evaluate(self):
        print("Evaluating started")

        total_acc = 0
        for global_step in range((self.dataset.num_samples-1) // self.args.batch_size + 1):
            # Session.Run!
            fetch_vals = self.session.run(
                {
                    'logits': self.model.logits,
                    'acc': self.model.acc,
                }
            )
            total_acc += fetch_vals['acc'] * len(fetch_vals['logits'])
        total_acc /= self.dataset.num_samples

        # Logging
        print("checkpoint: {}, dataset: {}, Acc: {:.4f}\n".format(
            self.args.checkpoint_path, self.args.dataset_name, 100 * total_acc))

        print("Evaluating finished!")
