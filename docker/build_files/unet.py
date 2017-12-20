import numpy as np
import tensorflow as tf

import layers


# Network

def parameter_efficient(in_channels=1, out_channels=2, start_filters=64, input_side_length=256, depth=4, res_blocks=2, filter_size=3, sparse_labels=True, batch_size=1, activation="cReLU", batch_norm=True):

    activation = str.lower(activation)
    if activation not in ["relu", "crelu"]:
        raise ValueError("activation must be \"ReLU\" or \"cReLU\".")

    pool_size = 2

    # Define inputs and helper functions #

    with tf.variable_scope('inputs'):
        inputs = tf.placeholder(tf.float32, shape=(batch_size, input_side_length, input_side_length, in_channels), name='inputs')
        if sparse_labels:
            ground_truth = tf.placeholder(tf.int32, shape=(batch_size, input_side_length, input_side_length), name='labels')
        else:
            ground_truth = tf.placeholder(tf.float32, shape=(batch_size, input_side_length, input_side_length, out_channels), name='labels')
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        training = tf.placeholder(tf.bool, shape=[], name="training")

        network_input = tf.transpose(inputs, perm=[0, 3, 1, 2])

    # [conv -> conv -> max pool -> drop out] + parameter updates
    def step_down(name, input_, filter_size=3, res_blocks=2, keep_prob=1., training=False):

        with tf.variable_scope(name):
            
            with tf.variable_scope("res_block_0"):
                conv_out, tiled_input = layers.res_block(input_, filter_size, channel_multiplier=2, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")
            
            for i in xrange(1, res_blocks):
                with tf.variable_scope("res_block_" + str(i)):
                    conv_out = layers.res_block(conv_out, filter_size, channel_multiplier=1, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")
            
            conv_out = conv_out + tiled_input

            pool_out = layers.max_pool(conv_out, pool_size, data_format="NCHW")
            
            bottom_out = layers.dropout(pool_out, keep_prob)
            side_out = layers.dropout(conv_out, keep_prob)

        return bottom_out, side_out

    # parameter updates + [upconv and concat -> drop out -> conv -> conv]
    def step_up(name, bottom_input, side_input, filter_size=3, res_blocks=2, keep_prob=1., training=False):

        with tf.variable_scope(name):
            added_input = layers.upconv_add_block(bottom_input, side_input, data_format="NCHW")

            conv_out = added_input
            for i in xrange(res_blocks):
                with tf.variable_scope("res_block_" + str(i)):
                    conv_out = layers.res_block(conv_out, filter_size, channel_multiplier=1, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")
            
            result = layers.dropout(conv_out, keep_prob)

        return result

    # Build the network #

    with tf.variable_scope('contracting'):

        outputs = []

        with tf.variable_scope("step_0"):

            # Conv 1
            in_filters = in_channels
            out_filters = start_filters

            stddev = np.sqrt(2. / (filter_size**2 * in_filters))
            w = layers.weight_variable([filter_size, filter_size, in_filters, out_filters], stddev=stddev, name="weights")

            out_ = tf.nn.conv2d(network_input, w, [1, 1, 1, 1], padding="SAME", data_format="NCHW")
            out_ = out_ + layers.bias_variable([out_filters, 1, 1], name='biases')

            # Batch Norm 1
            if batch_norm:
                out_ = tf.layers.batch_normalization(out_, axis=1, momentum=0.999, center=True, scale=True, training=training, trainable=True, name="batch_norm", fused=True)

            in_filters = out_filters

            # concatenated ReLU
            if activation == "crelu":
                out_ = tf.concat([out_, -out_], axis=1)
                in_filters = 2 * in_filters
            out_ = tf.nn.relu(out_)

            # Conv 2
            stddev = np.sqrt(2. / (filter_size**2 * in_filters))
            w = layers.weight_variable([filter_size, filter_size, in_filters, out_filters], stddev=stddev, name="weights")

            out_ = tf.nn.conv2d(out_, w, [1, 1, 1, 1], padding="SAME", data_format="NCHW")
            out_ = out_ + layers.bias_variable([out_filters, 1, 1], name='biases')

            # Res Block 1
            conv_out = layers.res_block(out_, filter_size, channel_multiplier=1, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")

            pool_out = layers.max_pool(conv_out, pool_size, data_format="NCHW")
            
            bottom_out = layers.dropout(pool_out, keep_prob)
            side_out = layers.dropout(conv_out, keep_prob)

            outputs.append(side_out)

        # Build contracting path
        for i in xrange(1, depth):
            bottom_out, side_out = step_down('step_' + str(i), bottom_out, filter_size=filter_size, res_blocks=res_blocks, keep_prob=keep_prob, training=training)
            outputs.append(side_out)

    # Bottom [conv -> conv]
    with tf.variable_scope('step_' + str(depth)):

        with tf.variable_scope("res_block_0"):
            conv_out, tiled_input = layers.res_block(bottom_out, filter_size, channel_multiplier=2, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")
        for i in xrange(1, res_blocks):
            with tf.variable_scope("res_block_" + str(i)):
                conv_out = layers.res_block(conv_out, filter_size, channel_multiplier=1, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")
        
        conv_out = conv_out + tiled_input
        current_tensor = layers.dropout(conv_out, keep_prob)

    with tf.variable_scope('expanding'):

        # Set initial parameter
        outputs.reverse()

        # Build expanding path
        for i in xrange(depth):
            current_tensor = step_up('step_' + str(depth + i + 1), current_tensor, outputs[i], filter_size=filter_size, res_blocks=res_blocks, keep_prob=keep_prob, training=training)
 
    # Last layer is a 1x1 convolution to get the predictions
    # We don't want an activation function for this one (softmax will be applied later), so we're doing it manually
    in_filters = current_tensor.shape.as_list()[1]
    stddev = np.sqrt(2. / in_filters)

    with tf.variable_scope('classification'):

        w = layers.weight_variable([1, 1, in_filters, out_channels], stddev, name='weights')
        b = layers.bias_variable([out_channels, 1, 1], name='biases')

        conv = tf.nn.conv2d(current_tensor, w, strides=[1, 1, 1, 1], padding="SAME", data_format="NCHW", name='conv')
        logits = conv + b

        logits = tf.transpose(logits, perm=[0, 2, 3, 1])

    return inputs, logits, ground_truth, keep_prob, training


def dsc(in_channels=1, out_channels=2, input_side_length=256, depth=512, filter_depth=256, filter_width=3, sparse_labels=True, batch_size=None):

    with tf.name_scope('inputs'):

        shape = [batch_size, input_side_length, input_side_length, in_channels]
        inputs = tf.placeholder(tf.float32, shape=shape, name='inputs')

        if sparse_labels:
            ground_truth = tf.placeholder(tf.int32, shape=(batch_size, input_side_length, input_side_length), name='labels')
        else:
            shape = [batch_size, input_side_length, input_side_length, out_channels]
            ground_truth = tf.placeholder(tf.float32, shape=shape, name='labels')
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    network_input = tf.transpose(inputs, perm=[0, 3, 1, 2])

    layer_out = layers.densely_stacked_column(network_input, depth, filter_depth, filter_width, output_depth=filter_depth, data_format="NCHW")

    weights = layers.weight_variable([filter_width, filter_width, filter_depth, out_channels], stddev=np.sqrt(2. / (filter_width * filter_width * filter_depth)))
    logits = tf.nn.conv2d(layer_out, weights, strides=[1, 1, 1, 1], padding="SAME", data_format="NCHW", name="conv")
    logits += layers.bias_variable([2, 1, 1])

    logits = tf.transpose(logits, perm=[0, 2, 3, 1])

    return inputs, logits, ground_truth, keep_prob


def unet(in_channels=1, out_channels=2, start_filters=64, input_side_length=572, depth=4, convolutions=2, filter_size=3, sparse_labels=True, batch_size=1, padded_convolutions=False):

    if not padded_convolutions:
        raise NotImplementedError("padded_convolutions=False has not yet been implemented!")

    pool_size = 2

    padding = "SAME" if padded_convolutions else "VALID"

    # Test whether input_side_length fits the depth, number of convolutions per step and filter_size
    output_side_length = input_side_length if padded_convolutions else get_output_side_length(input_side_length, depth, convolutions, filter_size, pool_size)

    # Define inputs and helper functions #
    with tf.variable_scope('inputs'):
        inputs = tf.placeholder(tf.float32, shape=(batch_size, input_side_length, input_side_length, in_channels), name='inputs')
        if sparse_labels:
            ground_truth = tf.placeholder(tf.int32, shape=(batch_size, output_side_length, output_side_length), name='labels')
        else:
            ground_truth = tf.placeholder(tf.float32, shape=(batch_size, output_side_length, output_side_length, out_channels), name='labels')
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

        network_input = tf.transpose(inputs, perm=[0, 3, 1, 2])

    # [conv -> conv -> max pool -> drop out] + parameter updates
    def step_down(name, _input):

        with tf.variable_scope(name):
            conv_out = layers.conv_block(_input, filter_size, channel_multiplier=2, convolutions=convolutions, padding=padding, data_format="NCHW")
            pool_out = layers.max_pool(conv_out, pool_size, data_format="NCHW")
            result = layers.dropout(pool_out, keep_prob)

        return result, conv_out

    # parameter updates + [upconv and concat -> drop out -> conv -> conv]
    def step_up(name, bottom_input, side_input):

        with tf.variable_scope(name):
            concat_out = layers.upconv_concat_block(bottom_input, side_input, data_format="NCHW")
            drop_out = layers.dropout(concat_out, keep_prob)
            result = layers.conv_block(drop_out, filter_size, channel_multiplier=0.5, convolutions=convolutions, padding=padding, data_format="NCHW")

        return result

    # Build the network #

    with tf.variable_scope('contracting'):

        # Set initial parameters
        outputs = []

        # Build contracting path
        with tf.variable_scope("step_0"):
            conv_out = layers.conv_block(network_input, filter_size, out_filters=start_filters, convolutions=convolutions, padding=padding, data_format="NCHW")
            pool_out = layers.max_pool(conv_out, pool_size, data_format="NCHW")
            current_tensor = layers.dropout(pool_out, keep_prob)
            outputs.append(conv_out)

        for i in xrange(1, depth):
            current_tensor, conv_out = step_down("step_" + str(i), current_tensor)
            outputs.append(conv_out)

    # Bottom [conv -> conv]
    with tf.variable_scope("step_" + str(depth)):
        current_tensor = layers.conv_block(current_tensor, filter_size, channel_multiplier=2, convolutions=convolutions, padding=padding, data_format="NCHW")

    with tf.variable_scope("expanding"):

        # Set initial parameter
        outputs.reverse()

        # Build expanding path
        for i in xrange(depth):
            current_tensor = step_up("step_" + str(depth + i + 1), current_tensor, outputs[i])

    # Last layer is a 1x1 convolution to get the predictions
    # We don't want an activation function for this one (softmax will be applied later), so we're doing it manually
    in_filters = current_tensor.shape.as_list()[1]
    stddev = np.sqrt(2. / in_filters)

    with tf.variable_scope("classification"):

        weight = layers.weight_variable([1, 1, in_filters, out_channels], stddev, name="weights")
        bias = layers.bias_variable([out_channels, 1, 1], name="biases")

        conv = tf.nn.conv2d(current_tensor, weight, strides=[1, 1, 1, 1], padding="VALID", name="conv", data_format="NCHW")
        logits = conv + bias

        logits = tf.transpose(logits, perm=[0, 2, 3, 1])

    return inputs, logits, ground_truth, keep_prob


def get_output_side_length(side_length, depth, convolutions, filter_size, pool_size):

    for i in xrange(depth - 1):

        for j in xrange(convolutions):
            side_length -= (filter_size - 1)
            if side_length < 0:
                raise ValueError("Input side length too small. Side length < 0 in contracting path after {} max pooling layers plus {} convolution.".format(i, j + 1))

        if (side_length % pool_size) != 0:
            raise ValueError("problem with input side length. Side length not divisible by pool size {}. Side length is {} before max pooling layer {}.".format(pool_size, side_length, i + 1))
        else:
            side_length /= pool_size

    for j in xrange(convolutions):
        side_length -= (filter_size - 1)
        if side_length < 0:
            raise ValueError("Input side length too small. Side length < 0 at bottom layer after {} convolution.".format(j + 1))

    for i in xrange(depth - 1):
        side_length *= pool_size
        side_length -= convolutions * (filter_size - 1)

    return side_length
