import argparse
import glob
import math
import os
import os.path
import shutil
import time

import numpy as np
import tensorflow as tf

import dpp

import utils.measurements as measurements
import utils.lesion_preprocessing as preprocessing

import architecture.layers as layers
import architecture.networks as networks


def main():

    # Command Line Arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', help='the model file', default=None)
    parser.add_argument('--snapshot_dir', '-s', help='the directory where the snapshots will be saved', required=True)
    parser.add_argument('--summary_dir', '-y', help='the directory where the summary file will be saved', required=True)
    parser.add_argument('--train_data_dir', '-t', help='the directory which contains the training data', required=True)
    parser.add_argument('--validation_data_dir', '-v', help='the directory which contains the validation data', required=True)
    parser.add_argument('--validation_interval', '-i', type=int, help='the number of training examples which will be used between validations', default=1000)
    parser.add_argument('--batch_size', '-b', type=int, help='the batch size for training', default=1)
    parser.add_argument('--name', '-n', help='the name of the experiment', default=None)
    parser.add_argument('--factor', '-f', help='the false positive factor to be used', type=float, required=True)
    parser.add_argument('--weights', '-w', help='the weights to be used for the different labels', nargs='+', type=float, default=[1, 1])
    args = parser.parse_args()

    validation_interval = args.validation_interval
    batch_size = args.batch_size

    if validation_interval % batch_size != 0:
        raise RuntimeError("The validation interval ({0}) must be a multiple of the batch size ({1})!".format(
                           validation_interval, batch_size))

    # Preprocessing

    print "Setting up preprocessing for training..."
    name = args.name + "_train" if args.name is not None else "train"
    training_pipeline = preprocessing.training(args.train_data_dir, save_name=name)

    print "Setting up preprocessing for validation..."
    name = args.name + "_validation" if args.name is not None else "validation"
    validation_pipeline = preprocessing.validation(args.validation_data_dir, save_name=name)
    validation_set = list(validation_pipeline)

    validation_pipeline.close()
    print "Validation examples: {}".format(len(validation_set))

    # Save Directories

    summary_dir = os.path.join(args.summary_dir, args.name)
    snapshot_dir = os.path.join(args.snapshot_dir, args.name)

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    # Training

    train_graph(args.weights, summary_dir, snapshot_dir, training_pipeline, validation_set,
                model_path=args.model,
                validation_interval=validation_interval,
                false_positive_factor=args.factor,
                batch_size=args.batch_size,
                learning_rate=0.001,
                beta1=0.99,
                beta2=0.9999,
                epsilon=1.,
                keep_prob=0.95)


def train_graph(label_weights, summary_dir, snapshot_dir, training_pipeline, validation_set,
                model_path=None,
                validation_interval=1000,
                false_positive_factor=0.5,
                batch_size=1,
                learning_rate=0.001,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08,
                keep_prob=0.8):
    """
    Loads or initializes a model and trains it on the data provided by the training pipeline.

    Parameters
    ----------
    label_weights : list of floats
        A list of the weights associated with the different labels in the ground truth.
    summary_dir: string
        The directory where the TensorBoard summary files are saved.
    snapshot_dir: string
        The directory where the snapshot files are saved.
    training_pipeline: iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary and
        the first input in the list is the network input, while the second input is the corresponding ground truth.
    validation_set: list
        An list of a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary and
        the first input in the list is the network input, while the second input is the corresponding ground truth.
    model_path: string, optional
        An optional path to a snapshot from where to load the model parameters.
    validation_interval: int
        The number of training examples in between validation intervals.
    false_positive_factor: float
        False positives receive a loss weight of false_positive_factor * label_weights[1], the weight of the class of interest.
    batch_size: int
        The batch size.
    learning_rate: float
        The learning rate.
    beta1: float
        The beta1 value of the adam optimizer.
    beta2: float
        The beta2 value of the adam optimizer.
    epsilon: float
        The epsilon value of the adam optimizer.
    keep_prob: float
        The keep probability for drop out layers.
    """

    # User info
    print ""
    print "### Training parameters ###"
    print "Summary Dir: {}".format(summary_dir)
    print "Snapshot Dir: {}".format(snapshot_dir)
    print "Validation Interval: {}".format(validation_interval)
    print "Validation Examples: {}".format(len(validation_set))
    print "Batch Size: {}".format(batch_size)
    print "Drop out keep probability: {}".format(keep_prob)
    print ""
    print "### Loss Parameters ###"
    print "Weights: {}".format(label_weights)
    print "False Positive Factor: {}".format(false_positive_factor)
    print ""
    print "### Optimizer Parameters ###"
    print "Learning Rate: {}".format(learning_rate)
    print "Beta1: {}".format(beta1)
    print "Beta2: {}".format(beta2)
    print "Epsilon: {}".format(epsilon)
    print ""

    with tf.Graph().as_default() as graph:

        print "Building graph"

        # Load model with appropriate inputs and outputs
        tf_input, tf_logits, tf_ground_truth, tf_keep_prob, tf_training_bool = networks.parameter_efficient(in_channels=5, out_channels=2, start_filters=64, input_side_length=256, sparse_labels=True, batch_size=batch_size)
        
        # Loss weights
        with tf.name_scope('inputs'):
            tf_label_weights = tf.constant(label_weights, dtype=tf.float32, name='weights')

        # Loss
        tf_loss, tf_weight_map = layers.weighted_softmax_cross_entropy_loss_with_false_positive_weights(tf_logits, tf_ground_truth, tf_label_weights, false_positive_factor=false_positive_factor)
        
        # Optimizer
        tf_train_op, tf_global_step = layers.adam_optimizer(tf_loss, learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, update_ops=True)

        # Summaries
        with tf.name_scope('summaries'):

            tf_image_summary_op = init_image_records(tf_input, tf_logits, tf_ground_truth, tf_weight_map, batch_size)
            recorder = measurements.SegmentationRecorder(tf_logits, tf_ground_truth, summary_dir, loss=tf_loss, label=1, graph=graph)

        # Numpy inputs
        np_input = np.zeros([batch_size, 256, 256, 5], dtype=np.float32)
        np_ground_truth = np.zeros([batch_size, 256, 256], dtype=np.float32)

        # Other
        tf_init = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=10)
        save_path = os.path.join(snapshot_dir, 'unet_model')

        # List of best models so far
        best_saved = []

        # List of saved models since the last update of best_saved
        current_saved = []

        graph.finalize()

        # Start Session

        with tf.Session(graph=graph) as sess:

            # Load or initialize the model parameters (weights, biases, batch norm, ...)
            if model_path is not None:
                print "Loading model {}".format(model_path)
                saver.restore(sess, model_path)
            else:
                print "Initializing new model"
                sess.run(tf_init)

            step = sess.run(tf_global_step)

            print "Starting training"

            while True:

                # Training Interval

                source = dpp.run_on(training_pipeline, processes=4, buffer_size=20)

                # Do training steps
                for _ in xrange(validation_interval / batch_size):
                    step = run_network(sess, step, tf_input, tf_ground_truth, tf_keep_prob, tf_training_bool, tf_global_step, tf_image_summary_op, np_input, np_ground_truth, source, recorder, tf_train_op=tf_train_op, keep_prob=keep_prob)

                # Save
                saver.save(sess, save_path, global_step=step)
                recorder.save_measurements(sess, step, phase="training")

                # Inoperative processes sometimes seem to be killed by the operating system
                # Better close them cleanly
                source.close()

                # Validation Interval

                val_iter = iter(validation_set)

                # Do validation steps
                for i in xrange(len(validation_set) / batch_size):
                    run_network(sess, i, tf_input, tf_ground_truth, tf_keep_prob, tf_training_bool, tf_global_step, tf_image_summary_op, np_input, np_ground_truth, val_iter, recorder, keep_prob=1.0)

                # Update the list of saved models based on the recorded measurements
                best_saved, current_saved = save_best(snapshot_dir, recorder, best_saved, current_saved, step, keep=10)

                # Save
                recorder.save_measurements(sess, step, phase="validation")


def init_image_records(tf_input, tf_logits, tf_ground_truth, tf_weight_map, batch_size):
    """
    The TensorFlow operations to generate images of the input and output for TensorBoard during training.
    Returns the merged summary operations.

    Parameters
    ----------
    tf_input : TF tensor
        The network input.
    tf_logits: TF tensor
        The network output before SoftMax.
    tf_ground_truth: TF tensor
        The desired output from the ground truth.
    tf_weight_map: TF tensor
        The loss weights assigned to each pixel. Same dimensions as the tf_logits and tf_ground_truth.
    batch_size: int
        The batch size.

    Returns
    -------
    summary : TF operation
        The merged summary operations.
    """

    with tf.name_scope('inputs'):

        size_x = tf.shape(tf_logits)[1]
        size_y = tf.shape(tf_logits)[2]

        tf_image_out = tf.slice(tf_input, [0, 0, 0, 2], [1, size_x, size_y, 1])
        tf_image_summary = tf.summary.image('image', tf_image_out, max_outputs=1, collections=['image_summaries'])

        tf_labels_float = tf.to_float(tf_ground_truth)
        tf_labels_float = tf.slice(tf_labels_float, [0, 0, 0], [1, size_x, size_y])
        tf_labels_out = tf.reshape(tf_labels_float, [1, size_x, size_y, 1], name='reshape_labels')
        tf_labels_summary = tf.summary.image('ground_truth', tf_labels_out, max_outputs=1, collections=['image_summaries'])

    with tf.name_scope('outputs'):

        tf_prediction = tf.to_float(tf.argmax(tf_logits, 3, name='prediction_values'))
        tf_prediction = tf.slice(tf_prediction, [0, 0, 0], [1, size_x, size_y])
        tf_prediction_out = tf.reshape(tf_prediction, [1, size_x, size_y, 1], name='reshape_prediction')
        tf_prediction_summary = tf.summary.image('prediction', tf_prediction_out, max_outputs=1, collections=['image_summaries'])

        tf_weight_map = tf.reshape(tf_weight_map, [-1, size_x, size_y, 1], name='reshape_weight_map')
        tf_weight_map_out = tf.slice(tf_weight_map, [0, 0, 0, 0], [1, size_x, size_y, 1])
        tf_weight_map_summary = tf.summary.image('weight_map', tf_weight_map_out, max_outputs=1, collections=['image_summaries'])

    return tf.summary.merge_all("image_summaries")


def fill_batch(np_input, np_ground_truth, source):
    """
    Fill the input and and ground truth with data from the given source.

    Parameters
    ----------
    np_input : Numpy array
        The network input. (Passed by reference.)
    np_ground_truth: Numpy array
        The desired output from the ground truth. (Passed by reference.)
    source: iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary and
        the first input in the list is the network input, while the second input is the corresponding ground truth.
    """

    for i in xrange(np_input.shape[0]):
        try:
            inputs, _ = source.next()
        except StopIteration:
            raise StopIteration

        np_input[i, :, :, :] = inputs[0]
        np_ground_truth[i, :, :] = inputs[1]


def run_network(sess, step, tf_inputs, tf_ground_truth, tf_keep_prob, tf_training_bool, tf_global_step, tf_image_summary_op, np_input, np_ground_truth, source, recorder, tf_train_op=None, keep_prob=1.0):
    """
    Loads the input from the pre-processing pipeline, prepares the feed dictionary, executes the training or inference step,
    and saves the summaries and measurements.

    Parameters
    ----------
    sess : TF session
        The TensorFlow Session.
    step: int
        The current training step.
    tf_input : TF tensor
        The network input (GPU side).
    tf_ground_truth: TF tensor
        The desired output from the ground truth (GPU side).
    tf_keep_prob: TF float
        The TF variable holding the keep probability for drop out layers.
    tf_training_bool: TF bool
        The TF variable holding the boolean value, which switches batch normalization to training or inference mode.
    tf_global_step: TF int
        The TF variable holding the current training step.
    tf_image_summary_op: TF operation
        The merged image summary operations for TensorBoard.
    np_input: Numpy array
        The network input (CPU side).
    np_ground_truth: Numpy array
        The desired output from the ground truth (CPU side).
    source: iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary and
        the first input in the list is the network input, while the second input is the corresponding ground truth.
    recorder: Segmentation Recorder
        Records measurements during training.
    tf_train_op: TF optimizer
        The optimizer performing gradient descent.
    keep_prob: float
        The keep probability for drop out layers.

    Returns
    -------
    step: int
        The current training step.
    """

    training = tf_train_op is not None
    phase = "training" if training else "validation"

    fill_batch(np_input, np_ground_truth, source)

    feed_dict = {
        tf_inputs: np_input,
        tf_ground_truth: np_ground_truth,
        tf_keep_prob: keep_prob,
        tf_training_bool: training
    }

    runables = {"step": tf_global_step}

    if tf_train_op is not None:
        runables["train"] = tf_train_op

    runables = recorder.add_measurements(runables)

    if step % 50 == 0:
        runables["image"] = tf_image_summary_op

    results = sess.run(runables, feed_dict=feed_dict)

    if step % 50 == 0:
        recorder.save_summary(results["image"], results["step"], phase=phase)

    recorder.record_measurements(results)
    return results["step"]


def save_best(snapshot_dir, recorder, best_saved, current_saved, step, keep=20):
    """
    Keeps an continually updated list of the last saved snapshots, once the list reaches the keep limit, it is merged with the
    list of the best models. The snapshots of the best models are kept in a separate folder, the rest is deleted.

    Makes sure the files of the best models according to validation score are kept.

    Parameters
    ----------
    snapshot_dir : string
        The directory where the model files are saved.
    recorder: Segmentation Recorder
        Records measurements during training.
    best_saved : list
        The list of snapshots of the best models with validation score.
    current_saved: list
        The list of snapshots of the last models with validation score.
    step: int
        The current training step.
    keep: int
        The number of snapshots to keep. Should be at least as great as the value given to tf.train.Saver.

    Returns
    -------
    best_saved : list
        The list of snapshots of the best models with validation score.
    current_saved: list
        The list of snapshots of the last models with validation score.
    """

    if recorder.sum_ground_truth + recorder.sum_prediction > 0.:
        model_tuple = ('unet_model-' + str(step), 2. * recorder.sum_intersection / (recorder.sum_ground_truth + recorder.sum_prediction))
        current_saved.append(model_tuple)
    else:
        model_tuple = ('unet_model-' + str(step), 0.)
        current_saved.append(model_tuple)

    print "Added {} to the list of current models.".format(model_tuple)

    if len(current_saved) == keep:
        best_saved = _save_best(snapshot_dir, best_saved, current_saved)
        current_saved = []

    return best_saved, current_saved


def _save_best(snapshot_dir, best_saved, current_saved):

    print "Saving the best models"

    best_dir = os.path.join(snapshot_dir, "best")

    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    list.sort(current_saved, key=lambda x: x[1], reverse=True)

    if len(best_saved) == 0:

        for elem in current_saved:

            print "Adding {} to the list of best models.".format(elem)

            # copy new file
            for _file in glob.glob(os.path.join(snapshot_dir, elem[0] + ".*")):
                shutil.copy2(_file, best_dir)

        best_saved = current_saved

    else:

        for i in xrange(len(current_saved)):

            # compare list of best new, starting with the best entry, to list of best old, starting with the worst entry
            if current_saved[i][1] >= best_saved[-(i + 1)][1]:

                print "Adding {} to the list of best models.".format(current_saved[i])

                # remove old file
                for _file in glob.glob(os.path.join(best_dir, best_saved[-(i + 1)][0] + ".*")):
                    os.remove(_file)

                # copy new file
                for _file in glob.glob(os.path.join(snapshot_dir, current_saved[i][0] + ".*")):
                    shutil.copy2(_file, best_dir)

                # replace old model in list
                best_saved[-(i + 1)] = current_saved[i]

            else:
                break

        list.sort(best_saved, key=lambda x: x[1], reverse=True)

    print "Removing the other files"

    for f in os.listdir(snapshot_dir):
        f = os.path.join(snapshot_dir, f)
        if os.path.isfile(f):
            os.remove(f)

    return best_saved


if __name__ == '__main__':
    main()
