import tensorflow as tf
import tensorflow.contrib.slim as slim

from audio_nets import tenet, tc_resnet


_available_nets = [
    "TENet6Model",
    "TENet12Model",
    "TENet6NarrowModel",
    "TENet12NarrowModel",
    "TCResNet8Model",
    "TCResNet14Model",
]


def _log_mel_spectrogram(audio, **kwargs):
    # only accept single channels
    audio = tf.squeeze(audio, -1)
    stfts = tf.contrib.signal.stft(audio,
                                   frame_length=kwargs["window_size_samples"],
                                   frame_step=kwargs["window_stride_samples"])
    spectrograms = tf.math.real(stfts * tf.math.conj(stfts))

    num_spectrogram_bins = spectrograms.shape[-1].value
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        kwargs["num_mel_bins"],
        num_spectrogram_bins,
        kwargs["sample_rate"],
        kwargs["lower_edge_hertz"],
        kwargs["upper_edge_hertz"],
    )

    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        spectrograms.shape[:-
                           1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    )

    log_offset = 1e-6
    log_mel_spectrograms = tf.math.log(mel_spectrograms + log_offset)

    return log_mel_spectrograms


def preprocess_mfcc(inputs, **kwargs):
    log_mel_spectrograms = _log_mel_spectrogram(inputs,
                                                **kwargs)

    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)
    mfccs = mfccs[..., :kwargs["num_mfccs"]]
    mfccs = tf.expand_dims(mfccs, axis=-1)

    return mfccs


class AudioNetModel:
    def __init__(self, args):
        self.args = args
        self.weight_decay = self.args.weight_decay
        self.dropout_keep_prob = 1 - self.args.dropout
        self.width_multiplier = self.args.width_multiplier
        self.scope_name = self.args.scope_name

    def build(self, wavs, labels, is_training):
        self.audio = preprocess_mfcc(wavs, **vars(self.args))
        self.labels = labels
        self.is_training = is_training

        self.logits = self.build_inference(self.audio, is_training=is_training)
        self.model_variables, self.model_l2_variables = self.get_variables(
            self.scope_name
        )
        self.total_loss, self.model_loss = self.build_loss(
            self.logits, self.labels
        )
        self.acc = self.build_acc(self.logits, self.labels)

    def get_variables(self, scope=''):
        def exclude_batch_norm(name):
            return ("batch_normalization" not in name) and ("BatchNorm" not in name)

        def include_model_scope(name, scope):
            if scope == '':
                return True
            return name.split(':')[0].split('/')[0].strip() == scope

        model_variables = [v for v in slim.get_variables_to_restore(
        ) if include_model_scope(v.name, scope)]
        model_l2_variables = [v for v in tf.compat.v1.trainable_variables(
        ) if exclude_batch_norm(v.name) and include_model_scope(v.name, scope)]

        return model_variables, model_l2_variables

    def build_loss(self, logits, labels):
        model_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits,
        )

        l2_loss = self.args.weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32))
             for v in self.model_l2_variables]
        ) if self.model_l2_variables != [] else 0

        total_loss = model_loss + l2_loss
        return total_loss, model_loss

    def build_acc(self, logits, labels):
        self.preds = tf.math.argmax(logits, -1)
        acc = tf.reduce_mean(tf.cast(tf.equal(labels, self.preds), tf.float32))
        return acc

    def build_inference(self, inputs, is_training=True):
        raise NotImplementedError


class TCResNet8Model(AudioNetModel):
    def __init__(self, args):
        super().__init__(args)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(tc_resnet.TCResNet_arg_scope(
            is_training=is_training,
            weight_decay=self.weight_decay,
            keep_prob=self.dropout_keep_prob)
        ):
            logits = tc_resnet.TCResNet8(
                inputs,
                self.args.num_classes,
                width_multiplier=self.width_multiplier,
                scope=self.scope_name
            )

            return logits


class TCResNet14Model(AudioNetModel):
    def __init__(self, args):
        super().__init__(args)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(tc_resnet.TCResNet_arg_scope(
            is_training=is_training,
            weight_decay=self.weight_decay,
            keep_prob=self.dropout_keep_prob)
        ):
            logits = tc_resnet.TCResNet14(
                inputs,
                self.args.num_classes,
                width_multiplier=self.width_multiplier,
                scope=self.scope_name
            )

            return logits


class TENet6Model(AudioNetModel):
    def __init__(self, args):
        super().__init__(args)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(tenet.TENet_arg_scope(
            is_training=is_training,
            weight_decay=self.weight_decay,
            keep_prob=self.dropout_keep_prob)
        ):
            logits = tenet.TENet6(
                inputs,
                self.args.num_classes,
                kernel_list=self.args.kernel_list,
                scope=self.scope_name,
            )

        return logits


class TENet12Model(AudioNetModel):
    def __init__(self, args):
        super().__init__(args)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(tenet.TENet_arg_scope(
            is_training=is_training,
            weight_decay=self.weight_decay,
            keep_prob=self.dropout_keep_prob)
        ):
            logits = tenet.TENet12(
                inputs,
                self.args.num_classes,
                kernel_list=self.args.kernel_list,
                scope=self.scope_name,
            )

        return logits


class TENet6NarrowModel(AudioNetModel):
    def __init__(self, args):
        super().__init__(args)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(tenet.TENet_arg_scope(
            is_training=is_training,
            weight_decay=self.weight_decay,
            keep_prob=self.dropout_keep_prob)
        ):
            logits = tenet.TENet6Narrow(
                inputs,
                self.args.num_classes,
                kernel_list=self.args.kernel_list,
                scope=self.scope_name,
            )

        return logits


class TENet12NarrowModel(AudioNetModel):
    def __init__(self, args):
        super().__init__(args)

    def build_inference(self, inputs, is_training):
        with slim.arg_scope(tenet.TENet_arg_scope(
            is_training=is_training,
            weight_decay=self.weight_decay,
            keep_prob=self.dropout_keep_prob)
        ):
            logits = tenet.TENet12Narrow(
                inputs,
                self.args.num_classes,
                kernel_list=self.args.kernel_list,
                scope=self.scope_name,
            )

        return logits
