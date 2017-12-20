import os
from os.path import isfile, join, basename
from warnings import warn

import numpy as np

import helper


def _file_paths(directory, input_identifier, ground_truth_identifier, split=None, split_selector=None, seed=None):

    if not type(directory) is str:
        raise TypeError("Directory must be a string. Received: %s" % type(directory))

    if not type(input_identifier) is str:
        raise TypeError("Input Identifier must be a string. Received: %s" % type(input_identifier))

    if not type(ground_truth_identifier) is str:
        raise TypeError("Ground Truth Identifier must be a string. Received: %s" % type(ground_truth_identifier))

    if split is not None and split_selector is None:
        raise ValueError("Split Selector may not be None, when Split parameter is used.")

    if split is None and split_selector is not None:
        raise ValueError("Split may not be None, when Split Selector parameter is used.")

    input_list = [f for f in sorted(os.listdir(directory)) if isfile(join(directory, f)) and input_identifier in f]
    ground_truth_list = [f for f in sorted(os.listdir(directory)) if isfile(join(directory, f)) and ground_truth_identifier in f]

    if len(input_list) != len(ground_truth_list):
        raise RuntimeError("Directory \"%s\" contains %d input items, but %d ground truth items!" % (directory, len(input_list), len(ground_truth_list)))

    combined_list = zip(input_list, ground_truth_list)

    if split is not None:

        start = int(round(np.sum(split[:split_selector]) * len(combined_list)))
        end = int(round(np.sum(split[:split_selector + 1]) * len(combined_list)))

        rs = np.random.RandomState(seed)
        combined_list = rs.permutation(combined_list)
        combined_list = combined_list[start:end]
        combined_list = sorted(combined_list, key=lambda x: x[0])

    for input_, ground_truth in combined_list:
        if input_.replace(input_identifier, "") != ground_truth.replace(ground_truth_identifier, ""):
            warn("Input item \"%s\" and ground truth item \"%s\" don't seem to match!" % (input_, ground_truth))

    return [(join(directory, f), join(directory, g)) for (f, g) in combined_list]


def file_paths(directory, input_identifier="image", ground_truth_identifier="label", random=True, iterations=0, split=None, split_selector=None, seed=None):
    """
    Returns an iterator that yields the paths to files in the given directory whose names contain the given identifiers.

    Files can be returned in random or sorted order, for the given number of iterations or indefinitely and, optionally,
    be split into several sets, such as training and test set, according to the given seed.

    Parameters
    ----------
    directory : str
    input_identifier : str
        A string that is part of the file name and uniquely identifies input files (such as images).
    ground_truth_identifier : str
        A string that is part of the file name and uniquely identifies ground truth files (such as class labels).
    random: bool
        Whether to return files names in random or sorted order.
    iterations: int
        The number of times each file name is to be returned. Set 0 for indefinitely.
    split: list of float, optional
        The relative proportions of each split of the dataset, e.g. use ``split=[0.75, 0.25]`` for a training set that
        contains 75% of the data and a test set that contains the remaining 25%. If ``None``, the default, data will
        not be split.
    split_selector: int, optional
        Zero based index. Specifies which part of the split to return, e.g. for the example above use ``split_selector=0``
        to return file paths from the training set and ``split_selector=1`` to return file paths from the test set.
    seed: int, optional
        The seed that is used to split the data. Use this parameter to ensure that training and test set reader use the
        same split.

    Returns
    -------
    gen : generator
        A generator that outputs a tuple of a list of file paths and a parameter dictionary or in short tuple(list(str), dict).
    """
    def path_generator(iterations):
        
        combined_list = _file_paths(directory, input_identifier, ground_truth_identifier, split=split, split_selector=split_selector, seed=seed)
        
        if split is None:
            print "Loading {} elements...".format(len(combined_list))
        else:
            print "Loading {} elements for split {} and split selector {}...".format(len(combined_list), split, split_selector)

        while True:

            if random:
                permutation = np.random.permutation(len(combined_list))
            else:
                permutation = np.asarray(xrange(len(combined_list)))

            for i in permutation:
                parameters = dict()
                parameters["file_names"] = [basename(combined_list[i][0]), basename(combined_list[i][1])]
                yield (combined_list[i], parameters)

            if iterations > 0:
                iterations -= 1
                if iterations == 0:
                    break

    gen = path_generator(iterations)
    helper.sign_up(gen)
    return gen
