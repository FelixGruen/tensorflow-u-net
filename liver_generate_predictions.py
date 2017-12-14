import os.path
import time
import argparse
import fnmatch
import string
import datetime

import numpy as np
import tensorflow as tf
import nibabel
import scipy.ndimage

import liver_preprocessing as preprocessing
import measurements
import unet

# Command Line Arguments

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m', help='the model file.')
parser.add_argument('--data_directory', '-d', help='the directory which contains the validation data set.')
parser.add_argument('--name', '-n', help='the name of the experiment', default=None)
parser.add_argument('--out_postfix', '-o', help='this postfix will be added to all files.', default="_prediction")
parser.add_argument('--no_batch_norm', help='set if you want to load a model without batch normalization', action='store_true')
parser.add_argument('--no_crelu', help='set if you want to load a model with standard ReLUs', action='store_true')
parser.add_argument('--unet', help='set if you want to load the standard U-Net model', action='store_true')

args = parser.parse_args()

# Training Parameters

print "Loading model file"

model = args.model

data_dir = args.data_directory
data_dir = os.path.join(data_dir, '')

batch_size = 1
in_channels = 5
edge_radius = (in_channels - 1) / 2

tf.reset_default_graph()
graph = tf.get_default_graph()

batch_norm = not args.no_batch_norm
activation_function = "ReLU" if args.no_crelu else "cReLU"

print "Setting up the architecture with {} and batch normalization {} ...".format(activation_function, "enabled" if batch_norm else "disabled")

if not args.unet:
    tf_inputs, tf_logits, _, tf_keep_prob, tf_training = unet.parameter_efficient(in_channels=in_channels, out_channels=2, start_filters=90, input_side_length=256, sparse_labels=True, batch_size=batch_size, activation=activation_function, batch_norm=batch_norm)
else:
    tf_inputs, tf_logits, _, tf_keep_prob = unet.unet(in_channels=in_channels, out_channels=2, start_filters=64, input_side_length=256, sparse_labels=True, batch_size=batch_size, padded_convolutions=True)

tf_prediction = tf.to_int32(tf.argmax(tf_logits, 3, name='prediction'))

np_inputs = np.zeros([batch_size, 256, 256, in_channels], dtype=np.float32)

saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)

print ""
print model

print "Loading pre-processing pipeline"
validation_pipeline = preprocessing.test(data_dir, save_name=args.name)

with tf.Session() as sess:

    print "Loading model {}".format(model)

    saver.restore(sess, model)

    print "Starting volume generation"

    slice_counter = edge_radius
    name = ""
    out_volume = None
    header = None

    for inputs, parameters in validation_pipeline:

        # Check if start of new volume
        if parameters["file_names"][1] != name:

            # If new volume is not the first, save the volume that came before
            if out_volume is not None:

                if slice_counter != out_volume.shape[-1] - edge_radius:
                    raise RuntimeError("slice_counter: {}, volume.shape: {}, for volume {}".format(slice_counter, out_volume.shape, name))

                # Create nibabel volume
                img = nibabel.Nifti1Image(out_volume, header.get_base_affine(), header=header)
                img.set_data_dtype(np.uint8)
                name_parts = os.path.splitext(os.path.basename(name))
                nibabel.save(img, os.path.join(data_dir, name_parts[0] + args.out_postfix + name_parts[1]))

            # Parameters of new volume
            name = parameters["file_names"][1]
            header = parameters["nifti_header"]
            dimensions = header.get_data_shape()[:3]

            # Reset variables
            slice_counter = edge_radius
            out_volume = np.zeros(dimensions, dtype=np.uint8)

        # Prepare network input
        np_inputs[0, :, :, :] = inputs[0]

        feed_dict = {
            tf_inputs: np_inputs,
            tf_keep_prob: 1.0
        }

        if not args.unet:
            feed_dict[tf_training] = False

        # Run network and obtain prediction
        np_prediction = sess.run(tf_prediction, feed_dict=feed_dict)

        # Re-orient and re-size prediction to fit original volume
        np_prediction = np.transpose(np_prediction[0, :, :])

        zooms = np.asarray(dimensions[0:2], dtype=np.float) / np.asarray([256., 256.], dtype=np.float)
        np_prediction = scipy.ndimage.zoom(np_prediction, zooms, order=0)

        out_volume[:, :, slice_counter] = np.round(np_prediction).astype(np.uint8)

        slice_counter += 1

    if slice_counter != out_volume.shape[-1] - edge_radius:
        raise RuntimeError("slice_counter: {}, volume.shape: {}, for last volume: {}".format(slice_counter, out_volume.shape, name))

    # Create nibabel volume for last volume
    img = nibabel.Nifti1Image(out_volume, header.get_base_affine(), header=header)
    img.set_data_dtype(np.uint8)
    name_parts = os.path.splitext(os.path.basename(name))
    nibabel.save(img, os.path.join(data_dir, name_parts[0] + args.out_postfix + name_parts[1]))

# After Volume Creation

validation_pipeline.close()

print "Done!"
