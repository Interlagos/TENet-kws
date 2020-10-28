import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

import pandas as pd
import os
import time

SILENCE_LABEL = '_silence_'
UNKNOWN_WORD_LABEL = '_unknown_'
BACKGROUND_DIR = '_background_noise_'


def _gen_random_from_zero(maxval, dtype=tf.float32):
    return tf.random.uniform([], maxval=maxval, dtype=dtype)


def _gen_empty_audio(desired_samples):
    return tf.zeros([desired_samples, 1], dtype=tf.float32)


def _mix_background(
        audio,
        desired_samples,
        background_data,
        is_silent,
        is_training,
        background_frequency,
        background_max_volume,
        **kwargs
):
    foreground_wav = tf.cond(
        is_silent,
        true_fn=lambda: _gen_empty_audio(desired_samples),
        false_fn=lambda: tf.identity(audio)
    )

    # sampling background
    random_background_data_idx = _gen_random_from_zero(
        len(background_data),
        dtype=tf.int32
    )
    background_wav = tf.case({
        tf.equal(background_data_idx, random_background_data_idx):
            lambda tensor=wav: tensor
        for background_data_idx, wav in enumerate(background_data)
    }, exclusive=True)
    background_wav = tf.image.random_crop(background_wav, [desired_samples, 1])

    if is_training:
        background_volume = tf.cond(
            tf.less(_gen_random_from_zero(1.0), background_frequency),
            true_fn=lambda: _gen_random_from_zero(background_max_volume),
            false_fn=lambda: 0.0,
        )
    else:
        background_volume = 0.0

    background_wav = tf.multiply(background_wav, background_volume)
    background_added = tf.add(background_wav, foreground_wav)
    augmented_audio = tf.clip_by_value(background_added, -1.0, 1.0)

    return augmented_audio


def _shift_audio(audio, desired_samples, shift_ratio):
    time_shift = int(desired_samples * shift_ratio)
    time_shift_amount = tf.random.uniform(
        [],
        minval=-time_shift,
        maxval=time_shift,
        dtype=tf.int32
    )

    time_shift_abs = tf.abs(time_shift_amount)

    def _pos_padding():
        return [[time_shift_amount, 0], [0, 0]]

    def _pos_offset():
        return [0, 0]

    def _neg_padding():
        return [[0, time_shift_abs], [0, 0]]

    def _neg_offset():
        return [time_shift_abs, 0]

    padded_audio = tf.pad(
        audio,
        tf.cond(tf.greater_equal(time_shift_amount, 0),
                true_fn=_pos_padding,
                false_fn=_neg_padding),
        mode="CONSTANT",
    )

    sliced_audio = tf.slice(
        padded_audio,
        tf.cond(tf.greater_equal(time_shift_amount, 0),
                true_fn=_pos_offset,
                false_fn=_neg_offset),
        [desired_samples, 1],
    )

    return sliced_audio


def _load_wav_file(filename, desired_samples=-1):
    wav_decoder = contrib_audio.decode_wav(
        tf.read_file(filename),
        desired_channels=1,
        desired_samples=desired_samples,
    )

    return wav_decoder.audio


def anchored_slice_or_pad(
        filename,
        desired_samples,
        sample_rate,
        **kwargs,
):
    is_silent = tf.equal(tf.strings.length(filename), 0)

    audio = tf.cond(
        is_silent,
        true_fn=lambda: _gen_empty_audio(desired_samples),
        false_fn=lambda: _load_wav_file(filename, desired_samples)
    )

    if "background_data" in kwargs:
        audio = _mix_background(audio, desired_samples,
                                is_silent=is_silent, **kwargs)

    return audio


def anchored_slice_or_pad_with_shift(
        filename,
        desired_samples,
        sample_rate,
        **kwargs
):
    is_silent = tf.equal(tf.strings.length(filename), 0)

    audio = tf.cond(
        is_silent,
        true_fn=lambda: _gen_empty_audio(desired_samples),
        false_fn=lambda: _load_wav_file(filename, desired_samples)
    )
    audio = _shift_audio(audio, desired_samples, shift_ratio=0.1)

    if "background_data" in kwargs:
        audio = _mix_background(audio, desired_samples,
                                is_silent=is_silent, **kwargs)

    return audio


class AudioWrapper:
    def __init__(self, args, mod, is_training, session):
        assert mod in ['train', 'valid', 'test']
        self.args = args
        self.mod = mod
        self.is_training = is_training
        self.session = session

        self.prepare_placeholders()
        self.prepare_dataset()
        self.init_iterator()

    def prepare_placeholders(self):
        # prepare args
        self.dataset_path = self.args.dataset_path
        self.split_file = os.path.join(self.args.split_dir, self.mod+'.txt')
        self.desired_samples = self.args.desired_samples
        self.sample_rate = self.args.sample_rate
        self.num_classes = self.args.num_classes

        # prepare words list
        self.prepare_words_list = [SILENCE_LABEL,
                                   UNKNOWN_WORD_LABEL] + self.args.wanted_words
        self.word_to_index = {word: i for i,
                              word in enumerate(self.prepare_words_list)}

        # prepare background data
        self.background_max_volume = tf.constant(
            self.args.background_max_volume)
        self.background_frequency = tf.constant(self.args.background_frequency)
        self.background_data = self.prepare_background_data()

        # prepare filename and label
        self.filenames, self.labels = self.get_filenames_labels()
        self.data = (self.filenames, self.labels)
        self.num_samples = len(self.filenames)

        # prepare placeholder
        self.filenames_placeholder = tf.compat.v1.placeholder(
            tf.string, self.num_samples)
        self.labels_placeholder = tf.compat.v1.placeholder(
            tf.int64, self.num_samples)
        self.placeholders = (self.filenames_placeholder,
                             self.labels_placeholder)

    def prepare_background_data(self):
        background_files = []
        background_path = os.path.join(self.dataset_path, BACKGROUND_DIR)
        filenames = os.listdir(background_path)
        for name in filenames:
            if not name.endswith('wav'):
                continue
            background_files.append(
                _load_wav_file(os.path.join(background_path, name))
            )

        return background_files

    def prepare_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders)
        dataset = dataset.shuffle(self.num_samples)
        dataset = dataset.map(self._parse_function,
                              num_parallel_calls=self.args.workers)
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        dataset = dataset.batch(self.args.batch_size)
        if self.is_training:
            dataset = dataset.shuffle(
                buffer_size=self.args.buffer_size, reshuffle_each_iteration=True).repeat(-1)

        self.dataset = dataset
        self.iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        self.next_elem = self.iterator.get_next()

    def init_iterator(self):
        self.session.run(self.iterator.initializer,
                         feed_dict={placeholder: variable for placeholder, variable in zip(self.placeholders, self.data)})

    def get_filenames_labels(self):
        df = pd.read_csv(self.split_file, header=None, names=['file'])
        df['label'] = df['file'].apply(lambda x: x.split(
            '/')[0]).apply(lambda x: x if x in self.prepare_words_list else UNKNOWN_WORD_LABEL)
        df['index_label'] = df['label'].apply(lambda x: self.word_to_index[x])
        df['file'] = df['file'].apply(
            lambda x: os.path.join(self.dataset_path, x))
        df.loc[df['label'].str.startswith(SILENCE_LABEL), 'file'] = ''

        return list(df['file']), list(df['index_label'])

    def _parse_function(self, filename, label):
        augmented_audio = self.augment_audio(
            filename,
            self.desired_samples,
            self.sample_rate,
            background_data=self.background_data,
            is_training=self.is_training,
            background_frequency=self.background_frequency,
            background_max_volume=self.background_max_volume,
        )

        return augmented_audio, label

    def augment_audio(self, filename, desired_samples, sample_rate, **kwargs):
        if self.is_training:
            return anchored_slice_or_pad_with_shift(filename, desired_samples, sample_rate, **kwargs)
        return anchored_slice_or_pad(filename, desired_samples, sample_rate, **kwargs)

    def get_input_and_output_op(self):
        return self.next_elem
