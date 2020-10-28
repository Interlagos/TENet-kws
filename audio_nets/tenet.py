import tensorflow as tf

slim = tf.contrib.slim


def TENet_arg_scope(is_training, weight_decay=0.00004, keep_prob=0.8):
    batch_norm_params = {
        "is_training": is_training,
        "decay": 0.99,
        "activation_fn": None,
    }

    with slim.arg_scope([slim.conv2d, slim.separable_convolution2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=None,
                        normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.dropout],
                                keep_prob=keep_prob,
                                is_training=is_training) as scope:
                return scope


def tenet(inputs, num_classes, n_channels, n_strides, n_ratios, n_layers, kernel_list, scope):
    L = inputs.shape[1]
    C = inputs.shape[2]

    assert len(n_channels) == len(n_strides) + 1

    with tf.compat.v1.variable_scope(scope):
        inputs = tf.reshape(inputs, [-1, L, 1, C])  # [N, L, 1, C]
        first_conv_kernel = [3, 1]
        conv_kernel = [9, 1]

        net = slim.conv2d(
            inputs, num_outputs=n_channels[0], kernel_size=first_conv_kernel, stride=1, scope="conv0")

        n_channels = n_channels[1:]

        for i, n in enumerate(n_channels):
            with tf.compat.v1.variable_scope(f"block{i}"):
                expand_n = int(n * n_ratios[i])
                for j, channel in enumerate(range(n_layers[i])):
                    stride = n_strides[i] if j == 0 else 1
                    if stride != 1 or net.shape[-1] != n:
                        layer_in = slim.conv2d(
                            net, num_outputs=n, activation_fn=None, kernel_size=1, stride=stride, scope=f"down")
                    else:
                        layer_in = net

                    net = slim.conv2d(net,
                                      expand_n,
                                      kernel_size=[1, 1],
                                      scope=f"pointwise_conv{j}_0")
                    if kernel_list:
                        list_net = []
                        for k, kernel_size in enumerate(kernel_list):
                            list_net.append(slim.separable_convolution2d(net,
                                                                         num_outputs=None,
                                                                         activation_fn=None,
                                                                         stride=stride,
                                                                         depth_multiplier=1,
                                                                         kernel_size=[
                                                                             kernel_size, 1],
                                                                         scope=f"depthwise_conv{j}_{k}"))
                        net = tf.add_n(list_net)
                        net = tf.nn.relu(net)
                    else:
                        net = slim.separable_convolution2d(net,
                                                           num_outputs=None,
                                                           stride=stride,
                                                           depth_multiplier=1,
                                                           kernel_size=conv_kernel,
                                                           scope=f"depthwise_conv{j}")
                    net = slim.conv2d(net, n, activation_fn=None, kernel_size=[
                                      1, 1], scope=f"pointwise_conv{j}_1")
                    net += layer_in

        net = slim.avg_pool2d(
            net, kernel_size=net.shape[1:3], stride=1, scope="avg_pool")
        net = slim.dropout(net)

        logits = slim.conv2d(
            net, num_classes, 1, activation_fn=None, normalizer_fn=None, scope="fc")
        logits = tf.reshape(
            logits, shape=(-1, logits.shape[3]), name="squeeze_logit")

    return logits


def TENet12(inputs, num_classes, kernel_list, scope):
    n_channels = [32] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [4] * 3

    if scope == '':
        if kernel_list:
            scope = "MTENet12"
        else:
            scope = "TENet12"

    return tenet(inputs, num_classes, n_channels, n_strides, n_ratios, n_layers, kernel_list, scope)


def TENet6(inputs, num_classes, kernel_list, scope):
    n_channels = [32] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [2] * 3

    if scope == '':
        if kernel_list:
            scope = "MTENet6"
        else:
            scope = "TENet6"

    return tenet(inputs, num_classes, n_channels, n_strides, n_ratios, n_layers, kernel_list, scope)


def TENet12Narrow(inputs, num_classes, kernel_list, scope):
    n_channels = [16] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [4] * 3

    if scope == '':
        if kernel_list:
            scope = "MTENet12Narrow"
        else:
            scope = "TENet12Narrow"

    return tenet(inputs, num_classes, n_channels, n_strides, n_ratios, n_layers, kernel_list, scope)


def TENet6Narrow(inputs, num_classes, kernel_list, scope):
    n_channels = [16] * 4
    n_strides = [2] * 3
    n_ratios = [3] * 3
    n_layers = [2] * 3

    if scope == '':
        if kernel_list:
            scope = "MTENet6Narrow"
        else:
            scope = "TENet6Narrow"

    return tenet(inputs, num_classes, n_channels, n_strides, n_ratios, n_layers, kernel_list, scope)
