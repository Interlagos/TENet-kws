import tensorflow as tf
import tensorflow.contrib.slim as slim

import os

import config
import models
from input_data import AudioWrapper


def routine_restore_and_initialize(saver, args, session):
    session.run(tf.compat.v1.global_variables_initializer())
    session.run(
        tf.compat.v1.local_variables_initializer())  # for metrics
    if args.checkpoint_path:
        saver.restore(session, args.checkpoint_path)
        print('restore from {}'.format(args.checkpoint_path))
    else:
        print('from init model')


def cpoy_parms(list_m, list_t, session):
    op_list = []
    for v, t in zip(list_m, list_t):
        op_list.append(t.assign(v))
        pass
    _ = session.run(op_list)


def get_padding_list(kernel_list, is_narrow):
    zero_list = []
    n_channel = 48 if is_narrow else 96
    for kernel in kernel_list:
        pad_dim = (9 - kernel) // 2
        if pad_dim <= 0:
            continue
        zero_list.append(tf.zeros([pad_dim] + [1, n_channel, 1]))

    return zero_list


def pad_weights(m_weights, kernel_list, is_narrow):
    outs = []
    kernel_num = len(kernel_list)
    zero_list = get_padding_list(kernel_list, is_narrow)
    for i in range(len(m_weights)//kernel_num):
        for j in range(kernel_num-1):
            v = m_weights[i*kernel_num + j]
            outs.append(tf.concat([zero_list[j], v, zero_list[j]], 0))
        # append 9*1 kernel
        outs.append(m_weights[i*kernel_num + kernel_num-1])

    return outs


def fuse_weights_betas(m_weights, m_betas, kernel_list, is_narrow):
    m_weights_ = pad_weights(m_weights, kernel_list, is_narrow)
    kernel_num = len(kernel_list)

    output_weights = []
    output_betas = []
    for idx in range(len(m_weights_)//kernel_num):
        weights = []
        betas = []
        for j in range(kernel_num):
            weight = m_weights_[idx*kernel_num+j]
            beta = m_betas[(idx*kernel_num+j)*3]
            mean = m_betas[(idx*kernel_num+j)*3+1]
            vari = m_betas[(idx*kernel_num+j)*3+2]
            
            n_channel = 48 if is_narrow else 96
            weights.append(tf.reshape(
                (tf.squeeze(weight) / tf.math.sqrt(vari)), [-1, 1, n_channel, 1]))
            betas.append(beta - mean / tf.math.sqrt(vari))

        output_weights.append(tf.add_n(weights))
        output_betas.append(tf.add_n(betas))

    return output_weights, output_betas


def fusion_tenet(list_v, saver_m, kernel_list, args, session):
    list_model_m = [v for v in list_v if v.name.startswith('MTENet')]
    list_model_t = [v for v in list_v if v.name.startswith('TENet')]

    is_narrow = list_model_m[0].name.split('/')[0].endswith('Narrow')

    m_weights = [v for v in list_model_m if v.name.split(
        '/')[-1].startswith('depthwise_weights')]
    m_betas = [v for v in list_model_m if v.name.split(
        '/')[-2].startswith('BatchNorm') and v.name.split('/')[-3].startswith('depthwise')]
    t_weights = [v for v in list_model_t if v.name.split(
        '/')[-1].startswith('depthwise_weights')]
    t_betas = [v for v in list_model_t if v.name.split(
        '/')[-1].startswith('beta') and v.name.split('/')[-3].startswith('depthwise')]

    m_others = [v for v in list_model_m if v not in m_weights + m_betas]
    t_others = [v for v in list_model_t if v not in t_weights +
                t_betas and not v.name.split('/')[-3].startswith('depthwise')]

    output_weights, output_betas = fuse_weights_betas(
        m_weights, m_betas, kernel_list, is_narrow)
    routine_restore_and_initialize(saver_m, args, session)
    cpoy_parms(m_others, t_others, session)
    cpoy_parms(output_weights, t_weights, session)
    cpoy_parms(output_betas, t_betas, session)


def main(args):
    is_training = False
    session = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
    dataset = AudioWrapper(args, args.dataset_name, is_training, session)
    wavs, labels = dataset.get_input_and_output_op()

    kernel_list = args.kernel_list
    model_m = models.__dict__[args.arch](args)
    model_m.build(wavs=wavs, labels=labels, is_training=is_training)

    args.kernel_list = None
    model_t = models.__dict__[args.arch](args)
    model_t.build(wavs=wavs, labels=labels, is_training=is_training)

    list_v = slim.get_variables_to_restore()

    saver_m = tf.compat.v1.train.Saver(var_list=model_m.model_variables)
    saver_t = tf.compat.v1.train.Saver(var_list=model_t.model_variables)

    fusion_tenet(list_v, saver_m, kernel_list, args, session)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    saver_t.save(session, os.path.join(args.save_folder, args.arch),
                 global_step=30000)
    
    print('Fusion finished!')


if __name__ == "__main__":
    args = config.arg_config()
    main(args)
