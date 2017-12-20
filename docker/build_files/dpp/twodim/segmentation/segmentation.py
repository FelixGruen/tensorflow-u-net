import collections
import numbers

import numpy as np
import scipy.ndimage

from ... import helper


def clip_img_values(source, minimum, maximum):
    """
    Sets all values of the image that are above the maximum value to the maximum and
    all values that are below the minimum value to the minimum.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    minimum: number or None
        The minimum value. Set None for no lower bound.
    maximum: number or None
        The maximum value. Set None for no upper bound.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    if not isinstance(minimum, numbers.Number):
        raise TypeError("Minimum must be a number! Received: {}".format(type(minimum)))

    if not isinstance(maximum, numbers.Number):
        raise TypeError("Maximum must be a number! Received: {}".format(type(maximum)))

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        np.clip(inputs[0], minimum, maximum, out=inputs[0])

        return (inputs, parameters)

    return helper.apply(source, transformation)


def random_crop(source, height, width, probability=1.0):
    """
    Extracts a crop of the given height and width from a random position for a random selection of the input. The position varies
    between datapoints, but is the same for all inputs of a single datapoint.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    height: int
        The height of the crop, the length along the first dimension.
    width: int
        The width of the crop, the length along the second dimension.
    probability: float
        The probability of cropping the input. If it is below 1, some inputs will be passed through unchanged.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """
    if not isinstance(height, numbers.Integral):
        raise TypeError("Height must be an integer! Received: {}".format(type(height)))

    if not isinstance(width, numbers.Integral):
        raise TypeError("Width must be an integer! Received: {}".format(type(width)))

    if not isinstance(probability, numbers.Number):
        raise TypeError("Probability must be a number! Received: {}".format(type(probability)))

    if probability > 1.0 or probability < 0.0:
        raise ValueError("Probability must be between 0.0 and 1.0! Received: {}".format(probability))

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        if(np.random.rand() < probability):  # do not apply translations always, just sometimes

            shape = inputs[0].shape

            max_x = shape[0] - height
            start_x = np.random.randint(max_x + 1) if max_x > 0 else 0
            end_x = start_x + height

            max_y = shape[1] - width
            start_y = np.random.randint(max_y + 1) if max_y > 0 else 0
            end_y = start_y + width

            for index, image in enumerate(inputs):
                inputs[index] = image[start_x:end_x, start_y:end_y, ...]

        return (inputs, parameters)

    return helper.apply(source, transformation)


def img_clahe(source):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """
    import skimage.exposure

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        image = inputs[0]

        # there doesn't seem to be a minmax function
        minimum = np.min(image)
        maximum = np.max(image)

        image = (image - minimum) / (maximum - minimum)
        image = skimage.exposure.equalize_adapthist(image)
        inputs[0] = image * (maximum - minimum) + minimum

        return (inputs, parameters)

    return helper.apply(source, transformation)


def random_translation(source, probability=1.0, border_usage=0.5, default_border=0.25, label_of_interest=None, default_pixel=None, default_label=None):
    """
    Translates a random selection of the input by a random amount. The translation varies between datapoints, but is the same for
    all inputs of a single datapoint.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    probability: float
        The probability of translating the input. If it is below 1, some inputs will be passed through unchanged.
    border_usage: float
        If a label_of_interest is set, the border is defined as the area between the rectangular bounding box around the region of
        interest and the edge of the image. This parameter defines how much of that border may be used for translation or, in other
        words, how much of that border may end up outside of the new image.
    default_border: float
        The amount of translation that is possible with respect to the input size, if label_of_interest is either not specified or
        not found in the input. border_usage does not apply to the default_border.
    label_of_interest: int
        The label of interest in the ground truth.
    default_pixel: number or None
        The fill value for the image input. If None, the minimum value will be used.
    default_label: number or None
        The fill value for the ground truth input. If None, the minimum value will be used.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """
    if not isinstance(probability, numbers.Number):
        raise TypeError("Probability must be a number! Received: {}".format(type(probability)))

    if not isinstance(border_usage, numbers.Number):
        raise TypeError("Border usage must be a number! Received: {}".format(type(border_usage)))

    if not isinstance(default_border, numbers.Number):
        raise TypeError("Default border must be a number! Received: {}".format(type(default_border)))

    if label_of_interest is not None and not isinstance(label_of_interest, numbers.Number):
        raise TypeError("Label of interest must be a number! Received: {}".format(type(label_of_interest)))

    if default_pixel is not None and not isinstance(default_pixel, numbers.Number):
        raise TypeError("Default pixel must be a number! Received: {}".format(type(default_pixel)))

    if default_label is not None and not isinstance(default_label, numbers.Number):
        raise TypeError("Default label must be a number! Received: {}".format(type(default_label)))

    if probability > 1.0 or probability < 0.0:
        raise ValueError("Probability must be between 0.0 and 1.0! Received: {}".format(probability))

    if border_usage > 1.0 or border_usage < 0.0:
        raise ValueError("Border usage must be between 0.0 and 1.0! Received: {}".format(border_usage))

    if default_border > 1.0 or default_border < 0.0:
        raise ValueError("Default border must be between 0.0 and 1.0! Received: {}".format(default_border))

    def non(s):
        return s if s < 0 else None

    def mom(s):
        return max(0, s)

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        if(np.random.rand() < probability):

            img = inputs[0]
            label = inputs[1]

            if label_of_interest is None or label_of_interest not in label:
                xdist = default_border * label.shape[0]
                ydist = default_border * label.shape[1]

            else:
                itemindex = np.where(label == 1)

                xdist = min(np.min(itemindex[0]), label.shape[0] - np.max(itemindex[0])) * border_usage
                ydist = min(np.min(itemindex[1]), label.shape[1] - np.max(itemindex[1])) * border_usage

            ox = np.random.randint(-xdist, xdist) if xdist >= 1 else 0
            oy = np.random.randint(-ydist, ydist) if ydist >= 1 else 0

            fill_value = default_pixel if default_pixel is not None else np.min(img)
            shift_img = np.full_like(img, fill_value)
            shift_img[mom(ox):non(ox), mom(oy):non(oy), ...] = img[mom(-ox):non(-ox), mom(-oy):non(-oy), ...]
            inputs[0] = shift_img

            fill_value = default_label if default_label is not None else np.min(label)
            shift_label = np.full_like(label, fill_value)
            shift_label[mom(ox):non(ox), mom(oy):non(oy), ...] = label[mom(-ox):non(-ox), mom(-oy):non(-oy), ...]
            inputs[1] = shift_label

            parameters["translation"] = (ox, oy)
        else:
            parameters["translation"] = (0, 0)

        return (inputs, parameters)

    return helper.apply(source, transformation)


def random_rotation(source, probability=1.0, upper_bound=180):
    """
    Rotates a random selection of the input by a random amount. The rotation varies between datapoints, but is the same for
    all inputs of a single datapoint.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    probability: float
        The probability of rotating the input. If it is below 1, some inputs will be passed through unchanged.
    upper_bound: number
        The maximum rotation in degrees.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """
    if not isinstance(probability, numbers.Number):
        raise TypeError("Probability must be a number! Received: {}".format(type(probability)))

    if not isinstance(upper_bound, numbers.Number):
        raise TypeError("Upper bound must be a number! Received: {}".format(type(upper_bound)))

    if upper_bound < 0:
        raise ValueError("Upper bound must be greater than 0! Received: {}".format(upper_bound))
    elif upper_bound > 180:
        upper_bound = 180

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        if(np.random.rand() < probability):

            angle = np.random.randint(-upper_bound, upper_bound)
            angle = (360 + angle) % 360

            inputs[0] = scipy.ndimage.interpolation.rotate(inputs[0], angle, reshape=False, order=1, cval=np.min(inputs[0]), prefilter=False)  # order = 1 => biliniear interpolation
            inputs[1] = scipy.ndimage.interpolation.rotate(inputs[1], angle, reshape=False, order=0, cval=np.min(inputs[1]), prefilter=False)  # order = 0 => nearest neighbour

            parameters["rotation"] = angle
        else:
            parameters["rotation"] = 0

        return (inputs, parameters)

    return helper.apply(source, transformation)


def mask_img_background(source, background_label, pixel_value=None):
    """
    Masks the background, all pixels that have the background_label in the ground truth, with the given pixel value or
    the minimum pixel value if None is given.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    background_label: int
        The label that defines the background.
    pixel_value: float
        The value that each background pixel is set to. If None, the minimum value will be used.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """
    if not isinstance(background_label, numbers.Number):
        raise TypeError("Background label must be a number! Received: {}".format(type(background_label)))

    if pixel_value is not None and not isinstance(pixel_value, numbers.Number):
        raise TypeError("Pixel value must be a number! Received: {}".format(type(pixel_value)))

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        fill_value = pixel_value if pixel_value is not None else np.min(inputs[0])

        inputs[0][inputs[1] == background_label] = fill_value

        return (inputs, parameters)

    return helper.apply(source, transformation)


def robust_img_scaling(source, ignore_values=[], initialization_generator=None, initialize=True):
    """
    Applies scaling to the image pixel values, that is robust to outliers. The scaler centers the input values on the median, and
    scales the values according to the interquantile range (IQR), the range between the 1st and 3rd quartile (25th and 75th quantile).

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    ignore_values: list of int
        A list of values that should be ignored when computing the median and IQR.
    initialization_generator: iterable
        An iterable that supplies the datapoints for initialization. If None and initialize is True, the initialization generator
        of the pipeline is used, if possible.
    initialize: bool
        If True, the median and scaling values will be pre-computed. Values will be saved together with the file name, so the
        file names in the output of the initialization generator should be unique.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    # Conditional import
    import sklearn.preprocessing as skpre

    scaler = skpre.RobustScaler(copy=False)

    if initialize:

        dictionary = {}

        def init_func(datapoint):
            inputs, parameters = datapoint

            file_name = parameters["file_names"][0]
            volume = inputs[0]
            for value in ignore_values:
                volume = np.ma.masked_values(volume, value, copy=False)
            volume = np.ma.compressed(volume)
            volume = volume.reshape(-1, 1)
            scaler.fit(volume)
            dictionary[file_name] = (scaler.center_, scaler.scale_)

        dictionary, _ = helper.initialize("dpp/twodim/segmentation/robust_img_scaling", init_func, dictionary, initialization_generator=initialization_generator)

        if dictionary is not None:

            def transformation(input_tuple):
                inputs, parameters = input_tuple

                file_name = parameters["file_names"][0]
                image = inputs[0]

                # flatten
                old_shape = image.shape
                image = image.reshape(-1, 1)

                # scale
                scaler_params = dictionary[file_name]
                scaler.center_ = scaler_params[0]
                scaler.scale_ = scaler_params[1]
                image = scaler.transform(image)

                # reshape
                inputs[0] = image.reshape(old_shape)

                return (inputs, parameters)

            return helper.apply(source, transformation)

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        image = inputs[0]

        # flatten
        old_shape = image.shape
        image = image.reshape(-1, 1)

        if len(ignore_values) > 0:
            img_fit = image
            for value in ignore_values:
                img_fit = np.ma.masked_values(img_fit, value, copy=False)
            img_fit = np.ma.compressed(img_fit)
            img_fit = img_fit.reshape(-1, 1)
            scaler.fit(img_fit)
        else:
            scaler.fit(image)

        # scale
        image = scaler.transform(image)

        # reshape
        inputs[0] = image.reshape(old_shape)

        return (inputs, parameters)

    return helper.apply(source, transformation)


def rescale_intensitiy(source, new_min=0, new_max=1, global_min=None, global_max=None):
    """
    Scales all image values, so that they lie in between new_min and new_max.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    new_min: float
        The new minimum pixel value
    new_max: float
        The new maximum pixel value.
    global_min: float or None
        If None, the minimum pixel value of the input will be used. Else, the value of global_min will be used as the old minimum.
    global_max: float or None
        If None, the maximum pixel value of the input will be used. Else, the value of global_min will be used as the old maximum.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    if not isinstance(new_min, numbers.Number):
        raise TypeError("New minimum must be a number! Received: {}".format(type(new_min)))

    if not isinstance(new_max, numbers.Number):
        raise TypeError("New maximum must be a number! Received: {}".format(type(new_max)))

    if global_min is not None and not isinstance(global_min, numbers.Number):
        raise TypeError("Global minimum must be a number! Received: {}".format(type(global_min)))

    if global_max is not None and not isinstance(global_max, numbers.Number):
        raise TypeError("Global maximum must be a number! Received: {}".format(type(global_max)))

    if new_min > new_max:
        raise ValueError("New minimum must be smaller than new maximum! Received: {} and {}".format(new_min, new_max))

    if global_min is not None and global_max is not None and global_min >= global_max:
        raise ValueError("Global minimum must be smaller than global maximum! Received: {} and {}".format(global_min, global_max))

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        image = inputs[0]

        if global_min is not None:
            old_min = global_min
        else:
            old_min = np.min(image)

        if global_max is not None:
            old_max = global_max
        else:
            old_max = np.max(image)

        image -= old_min  # minimum is now zero
        if old_max != old_min:  # avoid division by zero
            image *= (new_max - new_min) / (old_max - old_min)  # maximum is now (new_max - new_min), minimum is still zero
        image += new_min

        inputs[0] = image

        return (inputs, parameters)

    return helper.apply(source, transformation)


def pad(source, width, mode='reflect', image_only=True, default_pixel=None, default_label=None):
    """
    Pads the input with a padding of the given width and according to the given mode.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    width: int
        The width of the padding.
    mode: str
        One of the modes specified for numpy.pad.
    image_only: bool
        If True, only the image will be padded. Else all inputs will be padded.
    default_pixel: number or None
        If mode is 'constant', this value will be used to pad the image. If None, the minimum value will be used.
    default_label: number or None
        If mode is 'constant' and image_only is False, this value will be used to pad the ground truth. If None, the minimum
        value will be used.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    if default_pixel is not None and not isinstance(default_pixel, numbers.Number):
        raise TypeError("Default pixel must be a number! Received: {}".format(type(default_pixel)))

    if default_label is not None and not isinstance(default_label, numbers.Number):
        raise TypeError("Default label must be a number! Received: {}".format(type(default_label)))

    _constant = mode == 'constant'

    if image_only:

        def transformation(input_tuple):
            inputs, parameters = input_tuple

            if _constant:
                fill_value = default_pixel if default_pixel is not None else np.min(inputs[0])
                inputs[0] = np.pad(inputs[0], width, 'constant', constant_values=fill_value)
            else:
                inputs[0] = np.pad(inputs[0], width, mode)

            return (inputs, parameters)

        return helper.apply(source, transformation)

    else:

        def transformation(input_tuple):
            inputs, parameters = input_tuple

            if _constant:
                fill_value = default_pixel if default_pixel is not None else np.min(inputs[0])
                inputs[0] = np.pad(inputs[0], width, 'constant', fill_value)

                fill_value = default_label if default_label is not None else np.min(inputs[1])
                inputs[1] = np.pad(inputs[1], width, 'constant', fill_value)
            else:
                inputs[0] = np.pad(inputs[0], width, mode)
                inputs[1] = np.pad(inputs[1], width, mode)

            return (inputs, parameters)

        return helper.apply(source, transformation)


def reduce_to_single_label_slice(source):
    """
    Extracts the middle ground truth slice, if the input has a depth of more than 1 (for segmentation in 2.5 D). The depth
    is defined as the size of the last dimension.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    def reduction(input_tuple):
        inputs, parameters = input_tuple

        labels = inputs[1]

        while len(labels.shape) > 2:
            index = (labels.shape[-1] - 1) / 2
            labels = labels[..., index]

        inputs[1] = labels

        return (inputs, parameters)

    return helper.apply(source, reduction)


def fuse_labels_greater_than(source, label):
    """
    Fuses all labels in the ground truth such that all label values greater than the given label are set to 1 and all label values
    smaller or equal to the given label are set to 0.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    label: int
        The cut off value.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    if not isinstance(label, numbers.Number):
        raise TypeError("Label must be a number! Received: {}".format(type(label)))

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        inputs[1] = (inputs[1] > label).astype(inputs[1].dtype)

        return (inputs, parameters)

    return helper.apply(source, transformation)


def fuse_labels_other_than(source, label_number):
    """
    Fuses all labels in the ground truth such that all label values that are equal to the given label_number are set to 1 and
    all label values other than the given label are set to 0.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    label_number: int
        The label of interest.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    if not isinstance(label_number, numbers.Number):
        raise TypeError("Label Number must be a number! Received: {}".format(type(label_number)))

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        inputs[1] = (inputs[1] == label_number).astype(inputs[1].dtype)

        return (inputs, parameters)

    return helper.apply(source, transformation)


def __resize(inputs, desired_size):

    zooms = desired_size / np.array(inputs[0].shape[:2], dtype=np.float)

    image_zooms = np.append(zooms, np.ones(len(inputs[0].shape) - 2, dtype=zooms.dtype))
    inputs[0] = scipy.ndimage.zoom(inputs[0], image_zooms, order=1)  # order = 1 => biliniear interpolation

    label_zooms = np.append(zooms, np.ones(len(inputs[1].shape) - 2, dtype=zooms.dtype))
    inputs[1] = scipy.ndimage.zoom(inputs[1], label_zooms, order=0)  # order = 0 => nearest neighbour

    return inputs


def resize(source, desired_size):
    """
    Resizes the input to the desired_size.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    desired_size: array or list of length 2
        The height and width of the output.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    if not isinstance(desired_size, collections.Sequence) and not isinstance(desired_size, np.ndarray):
        TypeError("Desired size must be a sequence or array! Received: {}".format(type(desired_size)))

    desired_size = np.asarray(desired_size, dtype=np.int)

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        inputs = __resize(inputs, desired_size)
        # parameters["spacing"] = tuple(np.array(parameters["spacing"]) * (parameters["size"] / desired_size.astype(np.float)))
        parameters["size"] = tuple(desired_size)

        return (inputs, parameters)

    return helper.apply(source, transformation)


def __random_resize(inputs, desired_size, factor, default_pixel, default_label):

    if factor > 1.:

        zooms = (desired_size * factor) / np.array(inputs[0].shape[:2], dtype=np.float)
        scaled_size = np.rint(desired_size * factor).astype(np.int)

        # If the resized image would be larger than the desired size
        # use a center crop from the resized image
        x_start = (scaled_size[0] - desired_size[0]) / 2
        y_start = (scaled_size[1] - desired_size[1]) / 2
        x_end = x_start + desired_size[0]
        y_end = y_start + desired_size[1]

        new_size = list(desired_size)
        new_size.extend(inputs[0].shape[2:])
        image_zooms = np.append(zooms, np.ones(len(inputs[0].shape) - 2, dtype=zooms.dtype))

        output = np.zeros(new_size, dtype=inputs[0].dtype)
        image = scipy.ndimage.zoom(inputs[0], image_zooms, order=1)  # order = 1 => biliniear interpolation
        output[:, :, ...] = image[x_start:x_end, y_start:y_end, ...]
        inputs[0] = output

        new_size = list(desired_size)
        new_size.extend(inputs[1].shape[2:])
        label_zooms = np.append(zooms, np.ones(len(inputs[1].shape) - 2, dtype=zooms.dtype))

        output = np.zeros(new_size, dtype=inputs[1].dtype)
        labels = scipy.ndimage.zoom(inputs[1], label_zooms, order=0)  # order = 0 => nearest neighbour
        output[:, :, ...] = labels[x_start:x_end, y_start:y_end, ...]
        inputs[1] = output

    elif factor < 1.:

        zooms = (desired_size * factor) / np.array(inputs[0].shape[:2], dtype=np.float)
        scaled_size = np.rint(desired_size * factor).astype(np.int)

        # If the resized image would be smaller than the desired size
        # position the resized image in the center of the output image
        x_start = (desired_size[0] - scaled_size[0]) / 2
        y_start = (desired_size[1] - scaled_size[1]) / 2
        x_end = x_start + scaled_size[0]
        y_end = y_start + scaled_size[1]

        new_size = list(desired_size)
        new_size.extend(inputs[0].shape[2:])
        image_zooms = np.append(zooms, np.ones(len(inputs[0].shape) - 2, dtype=zooms.dtype))

        fill_value = default_pixel if default_pixel is not None else np.min(inputs[0])
        output = np.full(new_size, fill_value, dtype=inputs[0].dtype)
        image = scipy.ndimage.zoom(inputs[0], image_zooms, order=1)  # order = 1 => biliniear interpolation
        output[x_start:x_end, y_start:y_end, ...] = image[:, :, ...]
        inputs[0] = output

        new_size = list(desired_size)
        new_size.extend(inputs[1].shape[2:])
        label_zooms = np.append(zooms, np.ones(len(inputs[1].shape) - 2, dtype=zooms.dtype))

        fill_value = default_label if default_label is not None else np.min(inputs[1])
        output = np.full(new_size, fill_value, dtype=inputs[1].dtype)
        labels = scipy.ndimage.zoom(inputs[1], label_zooms, order=0)  # order = 0 => nearest neighbour
        output[x_start:x_end, y_start:y_end, ...] = labels[:, :, ...]
        inputs[1] = output

    else:

        inputs = __resize(inputs, desired_size)

    return inputs


def random_resize(source, desired_size, probability=1.0, lower_bound=0.9, upper_bound=1.1, default_pixel=None, default_label=None):
    """
    Resizes a random selection of the input to the desired_size. The new size varies randomly within the given bounds. If the
    resized image is larger than the desired size, a centered crop of the given size is returned. If the new size is smaller than
    the desired size, it will be padded to the given size. The size before cropping or padding varies between datapoints, but is
    the same for all inputs of a single datapoint.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    desired_size: array or list of length 2
        The height and width of the output.
    probability: float
        The probability of randomly resizing the input. If it is below 1, some inputs will simply be resized to the desired size.
    lower_bound: float
        The minimum relative size of the resized image before padding.
    upper_bound: float
        The maximum relative size of the resized image before cropping.
    default_pixel: number or None
        The fill value for the image input. If None, the minimum value will be used.
    default_label: number or None
        The fill value for the ground truth input. If None, the minimum value will be used.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    if not isinstance(desired_size, collections.Sequence) and not isinstance(desired_size, np.ndarray):
        TypeError("Desired size must be a sequence or array! Received: {}".format(type(desired_size)))

    if not isinstance(probability, numbers.Number):
        raise TypeError("Probability must be a number! Received: {}".format(type(probability)))

    if not isinstance(lower_bound, numbers.Number):
        raise TypeError("Lower bound must be a number! Received: {}".format(type(lower_bound)))

    if not isinstance(upper_bound, numbers.Number):
        raise TypeError("Upper bound must be a number! Received: {}".format(type(upper_bound)))

    if default_pixel is not None and not isinstance(default_pixel, numbers.Number):
        raise TypeError("Default pixel must be a number! Received: {}".format(type(default_pixel)))

    if default_label is not None and not isinstance(default_label, numbers.Number):
        raise TypeError("Default label must be a number! Received: {}".format(type(default_label)))

    if probability > 1.0 or probability < 0.0:
        raise ValueError("Probability must be between 0.0 and 1.0! Received: {}".format(probability))

    if lower_bound <= 0.0:
        raise ValueError("Lower bound must be greater than 0.0! Received: {}".format(lower_bound))

    if upper_bound <= lower_bound:
        raise ValueError("Upper bound must be greater than lower bound! Received: lower: {}, and upper: {}".format(lower_bound, upper_bound))

    desired_size = np.asarray(desired_size, dtype=np.int)

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        if(np.random.rand() < probability):

            # Zoom factor relative to desired size
            factor = np.random.rand() * (upper_bound - lower_bound) + lower_bound
            inputs = __random_resize(inputs, desired_size, factor, default_pixel, default_label)
            # parameters["spacing"] = tuple(np.array(parameters["spacing"]) * (parameters["size"] / scaled_size.astype(np.float)))
        else:

            inputs = __resize(inputs, desired_size)
            # parameters["spacing"] = tuple(np.array(parameters["spacing"]) * (parameters["size"] / desired_size.astype(np.float)))

        parameters["size"] = tuple(desired_size)

        return (inputs, parameters)

    return helper.apply(source, transformation)


def transpose(source):
    """
    Returns the transpose of each input. The transpose is equal to the original input with the first to axes interchanged.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        if len(inputs[0].shape) > 2:
            axes = np.arange(len(inputs[0].shape))
            axes[0] = 1
            axes[1] = 0
            inputs[0] = np.transpose(inputs[0], axes)
        else:
            inputs[0] = np.transpose(inputs[0])

        if len(inputs[1].shape) > 2:
            axes = np.arange(len(inputs[1].shape))
            axes[0] = 1
            axes[1] = 0
            inputs[1] = np.transpose(inputs[1], axes)
        else:
            inputs[1] = np.transpose(inputs[1])

        parameters["size"] = tuple(reversed(parameters["size"]))

        return (inputs, parameters)

    return helper.apply(source, transformation)


"""
def transpose(source, definition=None):

    if definition is None:

        def transformation(input_tuple):
            inputs, parameters = input_tuple

            inputs[0] = np.transpose(inputs[0])
            inputs[1] = np.transpose(inputs[1])

            parameters["size"] = tuple(reversed(parameters["size"]))
            # parameters["spacing"] = tuple(reversed(parameters["spacing"]))

            return (inputs, parameters)

        return helper.apply(source, transformation)

    else:

        if not isinstance(definition, collections.Sequence) and not isinstance(definition, np.ndarray):
            TypeError("Definition must be a sequence or array! Received: {}".format(type(definition)))

        def transformation(input_tuple):
            inputs, parameters = input_tuple

            inputs[0] = np.transpose(inputs[0], definition)
            inputs[1] = np.transpose(inputs[1], definition)

            # switch the size and spacing parameters for the different
            # dimensions according to the transpose definition
            size = np.asarray(parameters["size"])
            # spacing = np.asarray(parameters["spacing"])

            parameters["size"] = tuple(size[definition])
            # parameters["spacing"] = tuple(spacing[definition])

            return (inputs, parameters)

        return helper.apply(source, transformation)
"""


def normalization(source, mean=None, standard_deviation=None):
    """
    Normalizes the image input, by subtracting  the mean and deviding by the standard deviation.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    mean: number or None
        The mean to be subtracted. If None, the mean of the input image will be used.
    standard_deviation: number or None
        The standard deviation by which to devide the image values after mean subtraction. If None the standard deviation of the
        input image will be used.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    if mean is not None and not isinstance(mean, numbers.Number):
        raise TypeError("Mean must be a number! Received: {}".format(type(mean)))

    if standard_deviation is not None and not isinstance(standard_deviation, numbers.Number):
        raise TypeError("Standard deviation must be a number! Received: {}".format(type(standard_deviation)))

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        me = mean if mean is not None else np.mean(inputs[0])
        std = standard_deviation if standard_deviation is not None else np.std(inputs[0])

        inputs[0] -= me
        inputs[0] /= std

        return (inputs, parameters)

    return helper.apply(source, transformation)


def histogram_matching(source, template_generator):
    """
    Adjust the pixel values of a grayscale image such that its histogram matches that of a target image

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    templage_generator: iterable
        Same type as source. Used to compute the template histogram which is used to transform the input. The dimensions of the
        images returned by the template generator may differ from the dimensions of the images returned by source.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    def transformation(input_tuple):
        source_inputs, source_parameters = input_tuple

        template_inputs, _ = template_generator.next()

        source_img = source_inputs[0]
        template_img = template_inputs[0]

        old_shape = source_img.shape
        source_img = source_img.ravel()
        template_img = template_img.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        _, indices, s_counts = np.unique(source_img, return_inverse=True,
                                         return_counts=True)
        t_values, t_counts = np.unique(template_img, return_counts=True)

        # take the cumulative sum of the counts and normalize by the number of
        # pixels to get the empirical cumulative distribution functions for the
        # source and template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interpolated_s_values = np.interp(s_quantiles, t_quantiles, t_values)

        source_inputs[0] = interpolated_s_values[indices].reshape(old_shape)

        return (source_inputs, source_parameters)

    return helper.apply(source, transformation)


def crop_to_label(source, label, max_reduction=10, square_crop=True, default_pixel=None, default_label=None):
    """
    Crops the all inputs to the bounding box around the pixels with the given label.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    label: int
        The label to which to fit the inputs.
    max_reduction: float
        The maximum reduction. A maximum reduction of 10 means that the returned images are at least 1/10 of their original size.
    square_crop: bool
        If True, the shorter side of the bounding box will be symmetrically extended to create a square crop. If False, the crop
        will be fitted exactly.
    default_pixel: number or None
        The fill value for the image input. If None, the minimum value will be used. Only required if square_crop is True.
    default_label: number or None
        The fill value for the ground truth input. If None, the minimum value will be used. Only required if square_crop is True.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    if max_reduction is not None and not isinstance(max_reduction, numbers.Number):
        raise TypeError("Maximum zoom must be a number! Received: {}".format(type(max_reduction)))

    if default_pixel is not None and not isinstance(default_pixel, numbers.Number):
        raise TypeError("Default pixel must be a number! Received: {}".format(type(default_pixel)))

    if default_label is not None and not isinstance(default_label, numbers.Number):
        raise TypeError("Default label must be a number! Received: {}".format(type(default_label)))

    if max_reduction <= 0:
        raise ValueError("Maximum reduction must be greater than 0! Received: {}".format(max_reduction))

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        image = inputs[0]
        labels = inputs[1]
        label_map = labels == label
        image_shape = image.shape
        labels_shape = labels.shape

        # Find the minimum side lengths of the crop
        x_min_length = int((image_shape[0] - 1) / max_reduction) + 1
        y_min_length = int((image_shape[1] - 1) / max_reduction) + 1

        # Find the first and last row and column of the crop
        # First and last row and column which contain the label
        x = np.any(label_map, axis=1)
        y = np.any(label_map, axis=0)

        x_where = np.where(x)
        if len(x_where) == 0 or len(x_where[0]) == 0:
            return (inputs, parameters)
        xmin, xmax = x_where[0][[0, -1]]
        xmax += 1

        y_where = np.where(y)
        if len(y_where) == 0 or len(y_where[0]) == 0:
            return (inputs, parameters)
        ymin, ymax = y_where[0][[0, -1]]
        ymax += 1

        # Compute crop side lengths
        x_length = xmax - xmin
        y_length = ymax - ymin

        # Extend crop side lengths to the minimum side lengths
        if x_length < x_min_length:
            xmin -= (x_min_length - x_length) / 2
            xmax += (x_min_length - x_length + 1) / 2
            x_length = xmax - xmin

        if y_length < y_min_length:
            ymin -= (y_min_length - y_length) / 2
            ymax += (y_min_length - y_length + 1) / 2
            y_length = ymax - ymin

        # find and extend the shorter side for square crop
        if square_crop and x_length < y_length:

            # find new x_length. extend xmin and xmax for centered crop placement.
            xmin -= (y_length - x_length) / 2
            xmax += (y_length - x_length + 1) / 2
            x_length = y_length

        elif square_crop and y_length < x_length:

            # find new y_length. extend ymin and ymax for centered crop placement.
            ymin -= (x_length - y_length) / 2
            ymax += (x_length - y_length + 1) / 2
            y_length = x_length

        # start and end on the new canvas. if the centered crop extends over
        # the boarders of the old canvas those parts will be filled with
        # the default value or, if no default is provided, with the minimum value.
        xstart = 0
        xend = x_length

        ystart = 0
        yend = y_length

        if xmin < 0:
            xstart = abs(xmin)
            xmin = 0
        if xmax > image_shape[0]:
            xend = xend - (xmax - image_shape[0])
            xmax = image_shape[0]

        if ymin < 0:
            ystart = abs(ymin)
            ymin = 0
        if ymax > image_shape[1]:
            yend = yend - (ymax - image_shape[1])
            ymax = image_shape[1]

        fill_value = default_pixel if default_pixel is not None else np.min(image[xmin:xmax, ymin:ymax])
        image_size = [x_length, y_length]
        image_size.extend(image_shape[2:])
        image_crop = np.full(image_size, fill_value, dtype=np.float)
        image_crop[xstart:xend, ystart:yend, ...] = image[xmin:xmax, ymin:ymax, ...]

        fill_value = default_pixel if default_label is not None else np.min(labels[xmin:xmax, ymin:ymax])
        labels_size = [x_length, y_length]
        labels_size.extend(labels_shape[2:])
        labels_crop = np.full(labels_size, fill_value, dtype=np.float)
        labels_crop[xstart:xend, ystart:yend, ...] = labels[xmin:xmax, ymin:ymax, ...]

        # crop
        inputs[0] = image_crop
        inputs[1] = labels_crop

        parameters["crop_indices"] = [xstart, xend, ystart, yend]
        parameters["crop_canvas_size"] = [x_length, y_length]
        parameters["image_indices"] = [xmin, xmax, ymin, ymax]

        return (inputs, parameters)

    return helper.apply(source, transformation)
