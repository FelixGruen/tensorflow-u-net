import os
import os.path

import numpy as np
import tensorflow as tf


def raw_dice_measurements(logits, ground_truth, label=1):
    """
    Computes the sum of pixels of the given label of interest in the prediction, in the the ground truth and
    in the intersection of the two. These values can be used to compute the Dice score.

    Parameters
    ----------
    logits: TF tensor
        The network output before SoftMax.
    ground_truth: TF tensor
        The desired output from the ground truth.
    label: int
        The label of interest.

    Returns
    -------
    sum_prediction : TF float
        The sum of pixels with the given label in the prediction.
    sum_ground_truth: TF float
        The sum of pixels with the given label in the ground truth.
    sum_intersection: TF float
        The sum of pixels with the given label in the intersection of prediction and ground truth.    
    """

    with tf.name_scope('measurements'):

        label_const = tf.constant(label, dtype=tf.int32, shape=[], name='label_of_interest')

        prediction = tf.to_int32(tf.argmax(logits, 3, name='prediction'))

        prediction_label = tf.equal(prediction, label_const)
        ground_truth_label = tf.equal(ground_truth, label_const)

        sum_ground_truth = tf.reduce_sum(tf.to_float(ground_truth_label), name='sum_ground_truth')
        sum_prediction = tf.reduce_sum(tf.to_float(prediction_label), name='sum_prediction')

        with tf.name_scope('intersection'):
            sum_intersection = tf.reduce_sum(tf.to_float(tf.logical_and(prediction_label, ground_truth_label)))

    return sum_prediction, sum_ground_truth, sum_intersection


class SegmentationRecorder(object):
    """
    The segmentation recorder, records a number of useful performance measurements. It creates the appropriate TensorFlow summary
    operations, adds them to the feed dictionary for the next training step, keeps past values to compute averages and saves
    measurements and other summaries for TensorBoard.
    """

    def __init__(self, logits, ground_truth, summaries_dir, label=1,
                 loss=None, dice=True, precision=True, sensitivity=True, graph=None):
        """
        Parameters
        ----------
        logits: TF tensor
            The network output before SoftMax.
        ground_truth: TF tensor
            The desired output from the ground truth.
        summaries_dir: string
            The directory where the TensorBoard summary files are saved.
        label: int
            The label of interest.
        loss: TF float
            The loss.
        dice: bool
            If true, computes the Dice score.
        precision: bool
            If true, computes the precision.
        sensitivity: bool
            If true, computes the sensitivity or recall.
        graph: TF graph
            The TensorFlow graph of the model for TensorBoard.
        """

        self.loss = loss

        self.dice = dice
        self.precision = precision
        self.sensitivity = sensitivity

        self.prediction_op, self.ground_truth_op, self.intersection_op = raw_dice_measurements(logits, ground_truth, label=label)
        self._init_records()
        self._init_writers(summaries_dir, graph=graph)

    def reset_records(self):
        """
        Resets the records of past values used to compute averages and the Dice score over data sets.
        """

        if self.loss is not None:
            self.loss_list = []
        if self.dice:
            self.dice_list = []
        if self.precision:
            self.precision_list = []
        if self.sensitivity:
            self.sensitivity_list = []

        self.sum_prediction = np.float64(0.)
        self.sum_ground_truth = np.float64(0.)
        self.sum_intersection = np.float64(0.)

    def _init_records(self):
        """
        Initializes the summary operations and placeholders, which are used to pass values to those operations.
        """

        self.reset_records()

        with tf.name_scope('measurements'):

            if self.loss is not None:
                self.mean_loss = tf.placeholder(tf.float64, shape=[], name='mean_loss')
                loss_summary = tf.summary.scalar('loss', self.mean_loss, collections=['measurements'])

            if self.dice:
                self.mean_dice = tf.placeholder(tf.float64, shape=[], name='mean_dice')
                self.dice_values = tf.placeholder(tf.float64, shape=[None], name='dice_values')
                dice_summary = tf.summary.scalar('dice', self.mean_dice, collections=['measurements'])
                dice_histogram = tf.summary.histogram('dice_histogram', self.dice_values, collections=['measurements'])

            if self.precision:
                self.mean_precision = tf.placeholder(tf.float64, shape=[], name='mean_precision')
                self.precision_values = tf.placeholder(tf.float64, shape=[None], name='precision_values')
                precision_histogram = tf.summary.histogram('precision_histogram', self.precision_values, collections=['measurements'])
                precision_summary = tf.summary.scalar('precision', self.mean_precision, collections=['measurements'])

            if self.sensitivity:
                self.mean_sensitivity = tf.placeholder(tf.float64, shape=[], name='mean_sensitivity')
                self.sensitivity_values = tf.placeholder(tf.float64, shape=[None], name='sensitivity_values')
                sensitivity_summary = tf.summary.scalar('sensitivity', self.mean_sensitivity, collections=['measurements'])
                sensitivity_histogram = tf.summary.histogram('sensitivity_histogram', self.sensitivity_values, collections=['measurements'])

        self.measurements_summary_op = tf.summary.merge_all('measurements')

    def _init_writers(self, summaries_dir, graph=None):
        """
        Initializes the FileWriters for training and validation.
        """

        train_path = os.path.join(summaries_dir, 'train')
        val_path = os.path.join(summaries_dir, 'val')

        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(val_path):
            os.makedirs(val_path)

        self.train_writer = tf.summary.FileWriter(train_path, graph=graph)
        self.validation_writer = tf.summary.FileWriter(val_path, graph=graph)

    def add_measurements(self, dictionary):
        """
        Adds TensorFlow operations for computing the raw Dice values to the provided feed dictionary, which can be used in a call
        to Session.run.

        Parameters
        ----------
        dictionary: dictionary
            Feed dictionary used in a call to Session.run.
        """
        if self.loss is not None:
            dictionary["loss"] = self.loss

        dictionary["sum_prediction"] = self.prediction_op
        dictionary["sum_ground_truth"] = self.ground_truth_op
        dictionary["sum_intersection"] = self.intersection_op

        return dictionary

    def record_measurements(self, dictionary):
        """
        Records the values of the TensorFlow operations for computing the raw Dice values after a call to Session.run.
        Requires that the summary operations were added to the feed dictionary by the function add_measurements.

        Parameters
        ----------
        dictionary: dictionary
            Feed dictionary used in a call to Session.run.
        """

        sum_prediction = dictionary["sum_prediction"]
        sum_ground_truth = dictionary["sum_ground_truth"]
        sum_intersection = dictionary["sum_intersection"]

        if self.loss is not None:
            self.loss_list.append(dictionary["loss"])

        if self.dice and (sum_ground_truth + sum_prediction) > 0.:
            self.dice_list.append(2. * sum_intersection / float((sum_ground_truth + sum_prediction)))
        if self.precision and sum_prediction > 0.:
            self.precision_list.append(sum_intersection / float(sum_prediction))
        if self.sensitivity and sum_ground_truth > 0.:
            self.sensitivity_list.append(sum_intersection / float(sum_ground_truth))

        self.sum_prediction += np.float64(sum_prediction)
        self.sum_ground_truth += np.float64(sum_ground_truth)
        self.sum_intersection += np.float64(sum_intersection)

    def save_measurements(self, sess, step, phase="training", reset=True):
        """
        Saves the measurements for TensorBoard, including averages and histogramms.

        Parameters
        ----------
        sess: TF session
            A TensorFlow Session.
        step: int
            The current training step.
        phase: string
            The current phase. Either "training" or "validation"
        reset: bool
            Whether or not to reset all records after saving.
        """

        feed_dict = {}

        if self.loss is not None:
            temp_loss = np.mean(self.loss_list)
            feed_dict[self.mean_loss] = temp_loss
            print "Phase: {}, Iteration: {}, Loss: {}".format(phase, step, temp_loss)

        if self.dice:
            feed_dict[self.mean_dice] = 2. * self.sum_intersection / (self.sum_ground_truth + self.sum_prediction)
            feed_dict[self.dice_values] = self.dice_list
        if self.precision:
            feed_dict[self.mean_precision] = self.sum_intersection / self.sum_prediction
            feed_dict[self.precision_values] = self.precision_list
        if self.sensitivity:
            feed_dict[self.mean_sensitivity] = self.sum_intersection / self.sum_ground_truth
            feed_dict[self.sensitivity_values] = self.sensitivity_list

        summary = sess.run(self.measurements_summary_op, feed_dict=feed_dict)
        self.save_summary(summary, step, phase=phase)

        if reset:
            self.reset_records()

    def save_summary(self, summary, step, phase="training"):
        """
        Saves the provided summary operation using the training or validation SummaryWriter of the SegmentationRecorder.

        Parameters
        ----------
        summary: TF summary op
            A TensorFlow summary op.
        step: int
            The current training step.
        phase: string
            The current phase. Either "training" or "validation"
        """

        if phase == "training":
            self.train_writer.add_summary(summary, step)
            self.train_writer.flush()
        elif phase == "validation":
            self.validation_writer.add_summary(summary, step)
            self.validation_writer.flush()
        else:
            raise ValueError("Please specify either \"training\" or \"validation\". Received: {}".format(phase))
