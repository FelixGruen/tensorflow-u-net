import numpy as np
import tensorflow as tf


# Variables

def weight_variable(shape, stddev=0.1, name=None):
    return tf.Variable(initial_value=tf.truncated_normal(shape, stddev=stddev),
                       name=name,
                       dtype=tf.float32)


def bias_variable(shape, name=None):
    return tf.Variable(initial_value=tf.constant(0.1, shape=shape),
                       name=name,
                       dtype=tf.float32)


# Layers and Operations

def densely_stacked_column(input, column_depth, filter_depth, filter_width, output_depth=None, data_format="NHWC"):

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    if output_depth is None:
        output_depth = column_depth

    activation_maps = [input]

    concat_axis = 1 if data_format == "NCHW" else 3

    for i in xrange(column_depth / 2):

        in_ = tf.concat(activation_maps[max(0, i + 1 - filter_depth / 2):i + 1], axis=concat_axis)

        shape = in_.shape.as_list()
        depth = shape[1] if data_format == "NCHW" else shape[-1]

        filter_shape = [filter_width, filter_width, depth, 1]
        filter = weight_variable(filter_shape, stddev=np.sqrt(2. / (filter_width * filter_width * depth)))

        out_ = tf.nn.conv2d(in_, filter, [1, 1, 1, 1], padding="SAME", data_format=data_format)
        out_ += bias_variable([1])

        out_ = tf.concat([out_, -out_], axis=concat_axis)
        out_ = tf.nn.relu(out_)
        activation_maps.append(out_)

    return tf.concat(activation_maps[-(output_depth / 2):], axis=concat_axis)


def dendrite_layer(input, positions, weights, variances=[1., 1., 2.], width=11, data_format="NHWC"):

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    if data_format == "NHWC":
        raise NotImplementedError("data_format \"NHWC\" is not yet implemented!")

    p_h = positions[0]  # [dendrites]
    p_w = positions[1]  # [dendrites]
    p_d = positions[2]  # [dendrites]

    p_d = tf.expand_dims(p_d, axis=1)  # [dendrites, 1]

    shape = input.shape.as_list()

    height_dim = tf.expand_dims(tf.range(-(width / 2), (width + 1) / 2, dtype=tf.float32), axis=1)  # [width, 1]
    width_dim = tf.expand_dims(tf.range(-(width / 2), (width + 1) / 2, dtype=tf.float32), axis=1)  # [width, 1]
    depth_dim = tf.range(shape[1], dtype=tf.float32)  # [depth]

    height_dim -= p_h  # [width, dendrites]
    width_dim -= p_w  # [width, dendrites]
    depth_dim -= p_d  # [dendrites, depth]

    height_dim = tf.exp(tf.square(height_dim) / (-2. * variances[0])) / np.sqrt(2. * np.pi * variances[0])  # [width, dendrites]
    width_dim = tf.exp(tf.square(width_dim) / (-2. * variances[1])) / np.sqrt(2. * np.pi * variances[1])  # [width, dendrites]
    depth_dim = tf.exp(tf.square(depth_dim) / (-2. * variances[2])) / np.sqrt(2. * np.pi * variances[2])  # [dendrites, depth]

    shape = height_dim.shape.as_list()
    height_dim = tf.reshape(height_dim, [shape[0], 1, 1, shape[1]])  # [width, 1, 1, dendrites]

    shape = width_dim.shape.as_list()
    width_dim = tf.reshape(width_dim, [1, shape[0], 1, shape[1]])  # [1, width, 1, dendrites]

    filter = (height_dim * width_dim) * weights  # [width, width, out_depth, dendrites]

    shape = filter.shape.as_list()
    filter = tf.reshape(filter, [-1, shape[-1]])  # [width * width * out_depth, dendrites]
    filter = tf.matmul(filter, depth_dim)  # [width * width * out_depth, depth]
    filter = tf.reshape(filter, [shape[0], shape[1], shape[2], depth_dim.shape.as_list()[-1]])  # [width, width, out_depth, depth]
    filter = tf.transpose(filter, [0, 1, 3, 2])  # [width, width, depth, out_depth]

    with tf.name_scope("apply_filter"):

        output = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME", data_format=data_format)

    return output


def conv_den(x, dendrites, out_channels, variances=[1., 1., 2.], width=11, data_format="NHWC"):

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    if data_format == "NHWC":
        raise NotImplementedError("data_format \"NHWC\" is not yet implemented!")

    shape = x.shape.as_list()
    depth = shape[1]

    positions_height = tf.Variable(initial_value=tf.truncated_normal([dendrites], stddev=(width - 3) / 4.), name="dendrite_height", dtype=tf.float32)
    positions_width = tf.Variable(initial_value=tf.truncated_normal([dendrites], stddev=(width - 3) / 4.), name="dendrite_width", dtype=tf.float32)
    positions_depth = tf.Variable(initial_value=tf.abs(tf.truncated_normal([dendrites], stddev=(depth - 2) / 2.)) + 0.5, name="dendrite_depth", dtype=tf.float32)

    positions = [positions_height, positions_width, positions_depth]

    weights = weight_variable([out_channels, dendrites], name="weights")
    bias = bias_variable([out_channels, 1, 1], name="biases")

    output = dendrite_layer(x, positions, weights, variances=[1., 1., 2.], width=11, data_format=data_format)

    return tf.nn.relu(output + bias, name="relu")


def conv2d(x, w, b, padding="VALID"):

    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=padding, name='conv')
    return tf.nn.relu(conv + b, name='relu')


def relu_bn(x, name='relu_bn'):
    moments = tf.nn.moments(x, axis=[0])
    return tf.nn.batch_normalization(x, moments[0], moments[1], name=name)


def conv2d_res(x, w, b):
    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    return relu_bn(conv + b, name='relu_bn')


def upconv2d_res(x, w, b, stride, output_shape):
    upconv = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding='VALID', name='upconv')
    return relu_bn(upconv + b, name='relu')


def simple_upconv2d(x, w, stride, output_shape, data_format="NHWC"):

    if data_format == "NCHW":
        return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, 1, stride, stride], padding='VALID', data_format=data_format, name='upconv')

    return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding='VALID', data_format=data_format, name='upconv')


def upconv2d(x, w, b, stride, output_shape, data_format="NHWC"):

    batch_size = x.shape.as_list()[0]
    output_shape = [batch_size] + output_shape
    deconv_shape = tf.stack(output_shape)

    upconv = simple_upconv2d(x, w, stride, deconv_shape, data_format=data_format)
    return tf.nn.relu(upconv + b, name='relu')


def max_pool(x, n, data_format="NHWC"):

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    if data_format == "NCHW":
        return tf.nn.max_pool(x, ksize=[1, 1, n, n], strides=[1, 1, n, n], padding='VALID', data_format=data_format, name='max_pool')
    
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID', data_format=data_format, name='max_pool')


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob, name='dropout')


# Blocks

def crop_and_concat(x1, x2):
    """Crops a centered patch the size of x2 from x1 and concatenates it with x2.

    Arguments:
    x1: The patch is extracted from this image
    x2: The patch is concatenated to this image
    """

    shape_x1 = tf.shape(x1)
    shape_x2 = tf.shape(x2)

    # offset from the upper left corner (coordinate [0, 0])
    offsets = [0, (shape_x1[1] - shape_x2[1]) / 2, (shape_x1[2] - shape_x2[2]) / 2, 0]

    size = [-1, shape_x2[1], shape_x2[2], -1]

    x1_crop = tf.slice(x1, offsets, size, name='shortcut')

    return tf.concat([x1_crop, x2], 3, name='concat')


def conv_block(x, filter_size, out_filters=None, channel_multiplier=1, convolutions=2, padding="VALID", data_format="NHWC"):

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    if data_format == "NHWC":
        raise NotImplementedError("data_format \"NHWC\" is not yet implemented!")

    shape = x.shape.as_list()
    in_filters = shape[1]
    if out_filters is None:
        out_filters = channel_multiplier * in_filters
        if type(out_filters) in [float, np.float32, np.float64]:
            out_filters = int(round(out_filters))

    stddev = np.sqrt(2. / (filter_size**2 * in_filters))

    with tf.name_scope('conv1'):
        w = weight_variable([filter_size, filter_size, in_filters, out_filters], stddev, name='weights')
        b = bias_variable([out_filters, 1, 1], name='biases')

        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=padding, data_format=data_format, name='conv')
        current_tensor = tf.nn.relu(conv + b, name='relu')

    stddev = np.sqrt(2. / (filter_size**2 * out_filters))

    for i in xrange(convolutions - 1):
        with tf.name_scope('conv' + str(i + 2)):
            w = weight_variable([filter_size, filter_size, out_filters, out_filters], stddev, name='weights')
            b = bias_variable([out_filters, 1, 1], name='biases')

            conv = tf.nn.conv2d(current_tensor, w, strides=[1, 1, 1, 1], padding=padding, data_format=data_format, name='conv')
            current_tensor = tf.nn.relu(conv + b, name='relu')

    return current_tensor

def res_block(x, filter_size, channel_multiplier=1, depthwise_multiplier=2, convolutions=2, training=False, activation="relu", batch_norm=True, data_format="NHWC"):

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    if data_format == "NHWC":
        raise NotImplementedError("data_format \"NHWC\" is not yet implemented!")

    activation = str.lower(activation)
    if activation not in ["relu", "crelu"]:
        raise ValueError("activation must be \"ReLU\" or \"cReLU\".")

    shape = x.shape.as_list()
    in_filters = shape[1]
    out_filters = channel_multiplier * in_filters
    current_tensor = x

    for i in xrange(convolutions):
        with tf.variable_scope("conv_" + str(i + 1)):

            out_ = current_tensor
            if batch_norm:
                out_ = tf.layers.batch_normalization(out_, axis=1, momentum=0.999, center=True, scale=True, training=training, trainable=True, name="batch_norm", fused=True)

            if activation == "crelu":
                in_filters = 2 * in_filters
                out_ = tf.concat([out_, -out_], axis=1)
            out_ = tf.nn.relu(out_)

            stddev = np.sqrt(2. / in_filters)
            w_pointwise = weight_variable([1, 1, in_filters, out_filters / depthwise_multiplier], stddev, name='weights_pointwise')

            stddev = np.sqrt(2. / (filter_size**2))
            w_depth = weight_variable([filter_size, filter_size, out_filters / depthwise_multiplier, depthwise_multiplier], stddev, name='weights_depthwise')
            
            b = bias_variable([out_filters, 1, 1], name='biases')

            out_ = tf.nn.conv2d(out_, w_pointwise, [1, 1, 1, 1], padding="VALID", data_format=data_format, name="pointwise_conv")
            out_ = tf.nn.depthwise_conv2d_native(out_, w_depth, [1, 1, 1, 1], padding="SAME", data_format=data_format, name="depthwise_conv")
            out_ = out_ + b

            in_filters = out_filters
            current_tensor = out_

    if channel_multiplier != 1:
        x = tf.tile(x, [1, channel_multiplier, 1, 1])
        return x + current_tensor, x

    return x + current_tensor


def upconv_crop_concat_block(x_bottom, filter_size, in_filters, output_shape, x_side):
    """Returns a block that does the upconvolution of x_bottom and concats it with
    a centered crop from x_side.

    Argumentes:
    x_bottom: The input to the upconvolution.
    filter_size: The filter size for the upconvolution.
    x_side: the input to be center cropped to the size of x_bottom and concatenated.
    """

    out_filters = output_shape[-1]

    stddev = np.sqrt(2. / (filter_size**2 * in_filters))
    stride = filter_size

    with tf.name_scope('upconv'):
        w1 = weight_variable([filter_size, filter_size, out_filters, in_filters], stddev, name='weights')
        b1 = bias_variable([out_filters], name='biases')
        y_bottom = upconv2d(x_bottom, w1, b1, stride, output_shape)

    y_total = crop_and_concat(x_side, y_bottom)

    return y_total


def upconv_concat_block(x_bottom, x_side, data_format="NHWC"):
    """Returns a block that does the upconvolution of x_bottom and concats it with x_side.

    Argumentes:
    x_bottom: The input to the upconvolution.
    filter_size: The filter size for the upconvolution.
    x_side: the input to be concatenated.
    """

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    if data_format == "NHWC":
        raise NotImplementedError("data_format \"NHWC\" is not yet implemented!")

    bottom_shape = x_bottom.shape.as_list()
    side_shape = x_side.shape.as_list()

    in_filters = bottom_shape[1]
    out_filters = side_shape[1]
    filter_size = side_shape[2] / bottom_shape[2]

    stddev = np.sqrt(2. / in_filters)
    stride = filter_size

    with tf.name_scope('upconv'):
        w = weight_variable([filter_size, filter_size, out_filters, in_filters], stddev, name='weights')
        b = bias_variable([out_filters, 1, 1], name='biases')

        y_bottom = simple_upconv2d(x_bottom, w, stride, side_shape, data_format=data_format)
        y_bottom += b

    y_total = tf.concat([x_side, y_bottom], 1, name='concat')

    return y_total


def upconv_add_block(x_bottom, x_side, data_format="NHWC"):

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    if data_format == "NHWC":
        raise NotImplementedError("data_format \"NHWC\" is not yet implemented!")

    bottom_shape = x_bottom.shape.as_list()
    side_shape = x_side.shape.as_list()

    in_filters = bottom_shape[1]
    out_filters = side_shape[1]
    filter_size = side_shape[2] / bottom_shape[2]

    stddev = np.sqrt(2. / in_filters)
    stride = filter_size

    with tf.variable_scope('upconv'):
        w = weight_variable([filter_size, filter_size, out_filters, in_filters], stddev, name='weights')
        b = bias_variable([out_filters, 1, 1], name='biases')
        y_bottom = simple_upconv2d(x_bottom, w, stride, side_shape, data_format=data_format)
        y_bottom += b

    y_total = x_side + y_bottom

    return y_total


# Operations

def training(loss_op, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=0.1e-08, update_ops=False):

    with tf.variable_scope('train'):

        global_step = tf.Variable(0, name='global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)

        if update_ops:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss_op, global_step=global_step, name='minimize_loss')
        else:
            train_op = optimizer.minimize(loss_op, global_step=global_step, name='minimize_loss')

    return train_op, global_step