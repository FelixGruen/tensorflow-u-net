import os.path
import argparse
import string
import shutil

import numpy as np
import tensorflow as tf
import nibabel
import scipy.ndimage

import preprocessing
import unet

# Change the path to the model file here
liver_model = "/compute_lesion_predictions/models/liver_model"
lesion_model = "/compute_lesion_predictions/models/lesion_model"

# Some parameters
input_identifier = "volume"
prediction_identifier = "prediction"
liver_prediction_identifier = "liver_prediction"
lesion_prediction_identifier = "lesion_prediction"

batch_size = 1
in_channels = 5
edge_radius = (in_channels - 1) / 2

# Read in the directory
parser = argparse.ArgumentParser()
parser.add_argument('directory', help='the directory which contains the CT volumes')

args = parser.parse_args()

data_dir = args.directory
data_dir = os.path.join(data_dir, '')

print "Initializing CT volumes..."

# Generate dummy volumes, because the preprocessing pipeline was written for supervised training and expects a ground truth volume
file_list = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if os.path.isfile(os.path.join(data_dir, f)) and input_identifier in f]

for f in file_list:

    nifti = nibabel.load(f)
    header = nifti.header

    dimensions = header.get_data_shape()[:3]
    out_volume = np.zeros(dimensions, dtype=np.uint8)

    img = nibabel.Nifti1Image(out_volume, header.get_base_affine(), header=header)
    img.set_data_dtype(np.uint8)

    nibabel.save(img, string.replace(f, input_identifier, prediction_identifier))

# Set up the tensorflow graph
tf.reset_default_graph()

tf_inputs, tf_logits, _, tf_keep_prob = unet.unet(in_channels=in_channels, out_channels=2, start_filters=64, input_side_length=256, sparse_labels=True, batch_size=batch_size, padded_convolutions=True)
tf_prediction = tf.to_int32(tf.argmax(tf_logits, 3, name='prediction'))

# Define the main-memory-side input
np_inputs = np.zeros([batch_size, 256, 256, in_channels], dtype=np.float32)

# tf.train.Saver is used to load the model parameters
loader = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)

# Setting up preprocessing
print "Computing pre-processing parameters for liver prediction..."

save_name = string.replace(data_dir, "/", "_")
save_name = save_name[:-1] if save_name[-1] == "_" else save_name
save_name = save_name[1:] if save_name[0] == "_" else save_name
save_name = "liver_preprocessing_parameters_" + save_name

liver_pipeline = preprocessing.liver(data_dir, input_identifier, prediction_identifier, save_name=save_name)

with tf.Session() as sess:

    print "Loading liver model..."

    loader.restore(sess, liver_model)

    print "Predicting liver position..."

    slice_counter = edge_radius
    name = ""
    out_volume = None
    header = None

    for inputs, parameters in liver_pipeline:

        if parameters["file_names"][1] != name:

            if out_volume is not None:

                if slice_counter != out_volume.shape[-1] - edge_radius:
                    raise RuntimeError("slice_counter: {}, volume.shape: {}, for volume {}".format(slice_counter, out_volume.shape, name))

                # Create nibabel volume
                img = nibabel.Nifti1Image(out_volume, header.get_base_affine(), header=header)
                img.set_data_dtype(np.uint8)

                # Remove dummy volume
                os.remove(os.path.join(data_dir, name))

                # Save lesion prediction
                nibabel.save(img, os.path.join(data_dir, string.replace(name, prediction_identifier, liver_prediction_identifier)))

            name = parameters["file_names"][1]
            header = parameters["nifti_header"]

            slice_counter = edge_radius

            dimensions = header.get_data_shape()[:3]
            out_volume = np.zeros(dimensions, dtype=np.uint8)

        np_inputs[0, :, :, :] = inputs[0]

        feed_dict = {
            tf_inputs: np_inputs,
            tf_keep_prob: 1.0
        }

        # normal execution
        np_prediction = sess.run(tf_prediction, feed_dict=feed_dict)
        np_prediction = np.transpose(np_prediction[0, :, :])

        zooms = np.asarray(dimensions[0:2], dtype=np.float) / np.asarray([256., 256.], dtype=np.float)
        np_prediction = scipy.ndimage.zoom(np_prediction, zooms, order=0)

        out_volume[:, :, slice_counter] = np.round(np_prediction).astype(np.uint8)

        slice_counter += 1

    if slice_counter != out_volume.shape[-1] - edge_radius:
        raise RuntimeError("slice_counter: {}, volume.shape: {}, for last volume: {}".format(slice_counter, out_volume.shape, name))

    # Create nibabel volume
    img = nibabel.Nifti1Image(out_volume, header.get_base_affine(), header=header)
    img.set_data_dtype(np.uint8)

    # Remove dummy volume
    os.remove(os.path.join(data_dir, name))

    # Save lesion prediction
    nibabel.save(img, os.path.join(data_dir, string.replace(name, prediction_identifier, liver_prediction_identifier)))

liver_pipeline.close()

print "Computing pre-processing parameters for lesion prediction..."
lesion_pipeline = preprocessing.lesions(data_dir, input_identifier, liver_prediction_identifier, save_name=string.replace(save_name, "liver", "lesion"))

with tf.Session() as sess:

    print "Loading lesion model..."

    loader.restore(sess, lesion_model)

    print "Predicting lesion position..."

    slice_counter = edge_radius
    name = ""
    out_volume = None
    header = None

    for inputs, parameters in lesion_pipeline:

        if parameters["file_names"][1] != name:

            if out_volume is not None:

                if slice_counter != out_volume.shape[-1] - edge_radius:
                    raise RuntimeError("slice_counter: {}, volume.shape: {}, for volume {}".format(slice_counter, out_volume.shape, name))

                # Create nibabel volume
                img = nibabel.Nifti1Image(out_volume, header.get_base_affine(), header=header)
                img.set_data_dtype(np.uint8)

                # Remove liver volume
                os.remove(os.path.join(data_dir, name))

                # Save lesion prediction
                nibabel.save(img, os.path.join(data_dir, string.replace(name, liver_prediction_identifier, lesion_prediction_identifier)))

            name = parameters["file_names"][1]
            header = parameters["nifti_header"]

            slice_counter = edge_radius

            dimensions = header.get_data_shape()[:3]
            out_volume = np.zeros(dimensions, dtype=np.uint8)

        if 1 in inputs[1]:

            np_inputs[0, :, :, :] = inputs[0]

            feed_dict = {
                tf_inputs: np_inputs,
                tf_keep_prob: 1.0
            }

            # prediction
            np_prediction = sess.run(tf_prediction, feed_dict=feed_dict)

            np_prediction = np.transpose(np_prediction[0, :, :])

            crop_indices = parameters["crop_indices"]
            side_lengths = parameters["crop_canvas_size"]
            image_indices = parameters["image_indices"]

            zooms = np.asarray(side_lengths, dtype=np.float) / np.asarray([256., 256.], dtype=np.float)
            np_prediction = scipy.ndimage.zoom(np_prediction, zooms, order=0)

            np_prediction = np.round(np_prediction).astype(np.uint8) * 2
            out_volume[image_indices[0]:image_indices[1], image_indices[2]:image_indices[3], slice_counter] = np_prediction[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]

        slice_counter += 1

    if slice_counter != out_volume.shape[-1] - edge_radius:
        raise RuntimeError("slice_counter: {}, volume.shape: {}, for last volume: {}".format(slice_counter, out_volume.shape, name))

    # Create nibabel volume
    img = nibabel.Nifti1Image(out_volume, header.get_base_affine(), header=header)
    img.set_data_dtype(np.uint8)

    # Remove liver volume
    os.remove(os.path.join(data_dir, name))

    # Save lesion prediction
    nibabel.save(img, os.path.join(data_dir, string.replace(name, liver_prediction_identifier, lesion_prediction_identifier)))

# After Volume Creation

lesion_pipeline.close()

shutil.rmtree("temp")

print "Done!"
