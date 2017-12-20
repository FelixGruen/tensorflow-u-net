import nibabel
import numpy as np

from ... import helper


def __get_slice(volume, slice_index, dimension):

    if dimension == 0:
        out_slice = volume.dataobj[slice_index, :, :]
    elif dimension == 1:
        out_slice = volume.dataobj[:, slice_index, :]
    elif dimension == 2:
        out_slice = volume.dataobj[:, :, slice_index]
    else:
        raise Error("Unknown dimension: %s" % str(dimension))

    return np.asarray(out_slice)


def __get_slices(volume, start_index, end_index, dimension):

    if dimension == 0:
        out_slices = volume.dataobj[start_index:end_index + 1, :, :]
        out_slices = np.transpose(out_slices, axes=[1, 2, 0])
    elif dimension == 1:
        out_slices = volume.dataobj[:, start_index:end_index + 1, :]
        out_slices = np.transpose(out_slices, axes=[0, 2, 1])
    elif dimension == 2:
        out_slices = volume.dataobj[:, :, start_index:end_index + 1]
    else:
        raise Error("Unknown dimension: %s" % str(dimension))

    return np.asarray(out_slices)


def __set_parameters(parameters, header, dimension):

    if dimension == 0:
        parameters["spacing"] = header.get_zooms()[1:3]
        parameters["size"] = header.get_data_shape()[1:3]
    elif dimension == 1:
        parameters["spacing"] = np.concatenate([header.get_zooms()[0:1], header.get_zooms()[2:3]])
        parameters["size"] = np.concatenate([header.get_data_shape()[0:1], header.get_data_shape()[2:3]])
    elif dimension == 2:
        parameters["spacing"] = header.get_zooms()[:2]
        parameters["size"] = header.get_data_shape()[:2]
    else:
        raise Error("Unknown dimension: %s" % str(dimension))

    parameters["original_spacing"] = parameters["spacing"]
    parameters["original_size"] = parameters["size"]

    parameters["nifti_header"] = header

    return parameters


def load_slice(source, slice_type='axial', dtype=np.float32):
    """
    Loads a random slice from the medical volumes specified by the provided paths.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of file paths and a parameter dictionary.
    slice_type: 'coronal', 'sagittal', 'axial', or None
        The anatomical plane of the returned slices. If None, a plane will be drawn at random for each slice.
    dtype: data-type
        The desired data-type for the returned arrays.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    # If slice type is specified, set dimension
    if slice_type == 'coronal':
        dimension = 0
    elif slice_type == 'sagittal':
        dimension = 1
    elif slice_type == 'axial':
        dimension = 2
    elif slice_type is None:
        dimension = -1
    else:
        raise Error("Unknown slice_type: %s" % str(slice_type))

    def transformation(dimension):

        for inputs, parameters in source:

            # If slice type is not specified, set random dimension
            if slice_type is None:
                dimension = np.random.randint(3)

            outputs = []

            nifti = nibabel.load(inputs[0])
            slice_index = np.random.randint(nifti.header.get_data_shape()[dimension])

            img_slice = __get_slice(nifti, slice_index, dimension)
            outputs.append(img_slice.astype(dtype))

            header = nifti.header
            parameters = __set_parameters(parameters, header, dimension)

            nifti = nibabel.load(inputs[1])
            label_slice = __get_slice(nifti, slice_index, dimension)
            outputs.append(label_slice.astype(dtype))

            yield (outputs, parameters)

    gen = generator(dimension)
    helper.sign_up(gen)
    return gen


def load_slice_filtered(source, label_of_interest=2, label_required=1, min_frequency=0.8, max_tries=10, depth=1, single_label_slice=False, slice_type='axial', dtype=np.float32):
    """
    Loads a random slice which contains the given label of interest from the medical volumes specified by the provided paths. Slices
    are loaded in random order until either a slice with the given label of interest is found or the maximum number of tries has been
    reached.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of file paths and a parameter dictionary.
    label_of_interest: int
        A label that should be present in the returned slice. If depth is greater than 1, only the center plane will be checked.
    label_required: int
        A label that must be present in the returned slice. If depth is greater than 1, only the center plane will be checked.
        If this is not needed, it can be set to the same number as label_of_interest.
    min_frequency: float
        The minimum proportion of outputs with the given label of interest. As long as the actual proportion is greater than the given
        minimum frequency, the first randomly drawn slice will be returned from each volume. This can be used to mix slices without
        the label of interest into the training set.
    max_tries: int
        The maximum number of slices which are drawn in random order from each volume. Once this maximum is reached for a volume, the
        first slice with the given required label is returned.
    depth: int
        The depth of the slice. Must be an uneven integer greater than zero.
    single_label_slice: bool
        If False, the returned label slice will have the same depth as the returned image slice. If True, the label slice will have
        depth 1 and correspond to the center plane of the returned image slice.
    slice_type: 'coronal', 'sagittal', 'axial', or None
        The anatomical plane of the returned slices. If None, a plane will be drawn at random for each volume.
    dtype: data-type
        The desired data-type for the returned arrays.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    # If slice type is specified, set dimension
    if slice_type == 'coronal':
        dimension = 0
    elif slice_type == 'sagittal':
        dimension = 1
    elif slice_type == 'axial':
        dimension = 2
    elif slice_type is None:
        dimension = -1
    else:
        raise Error("Unknown slice_type: {}".format(slice_type))

    if depth < 1 or 2 % 1 != 0 or ((depth - 1) / 2) * 2 + 1 != depth:
        raise ValueError("Depth must be a positive uneven integer. Is: {}".format(depth))

    radius = (depth - 1) / 2

    def generator(dimension):

        counter = 0  # counts the number of slices with the label_of_interest
        total = 0  # counts the total number of slices

        for inputs, parameters in source:

            # If slice type is not specified, set random dimension
            if slice_type is None:
                dimension = np.random.randint(3)

            label_volume = nibabel.load(inputs[1])

            min_ = radius
            max_ = label_volume.header.get_data_shape()[dimension] - radius
            indices = np.random.permutation(np.arange(min_, max_))
            indices = [int(i) for i in indices]  # nibabel doesn't like numpy type indices

            i = 0
            found = False

            if total == 0 or counter / float(total) < min_frequency:

                for i in xrange(min(max_tries, len(indices))):
                    label_slice = __get_slice(label_volume, indices[i], dimension)
                    if label_required in label_slice and label_of_interest in label_slice:
                        if depth > 1 and not single_label_slice:
                            label_slice = __get_slices(label_volume, indices[i] - radius, indices[i] + radius, dimension)
                        found = True
                        counter += 1
                        break

            if not found:
                for i in xrange(i, len(indices)):
                    label_slice = __get_slice(label_volume, indices[i], dimension)
                    if label_required in label_slice:
                        found = True
                        if label_of_interest in label_slice:
                            counter += 1
                        if depth > 1 and not single_label_slice:
                            label_slice = __get_slices(label_volume, indices[i] - radius, indices[i] + radius, dimension)
                        break

            if not found:
                continue

            total += 1
            image_volume = nibabel.load(inputs[0])

            outputs = []

            # image slice first
            if depth > 1:
                outputs.append(__get_slices(image_volume, indices[i] - radius, indices[i] + radius, dimension).astype(dtype))
            else:
                outputs.append(__get_slice(image_volume, indices[i], dimension).astype(dtype))
            outputs.append(label_slice.astype(dtype))

            header = image_volume.header
            parameters = __set_parameters(parameters, header, dimension)

            parameters["slices_total"] = total
            parameters["slices_label_of_interest"] = counter

            yield (outputs, parameters)

    gen = generator(dimension)
    helper.sign_up(gen)
    return gen


def load_all_slices_filtered(source, label_required=1, depth=1, single_label_slice=False, slice_type='axial', dtype=np.float32):
    """
    Loads all slices which contain the given required label from the medical volumes specified by the provided paths and returns
    them one by one.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of file paths and a parameter dictionary.
    label_required: int
        A label that must be present in the returned slice. If depth is greater than 1, only the center plane will be checked.
    depth: int
        The depth of the slices. Must be an uneven integer greater than zero. If depth is greater than 1, the returned slices will
        overlap.
    single_label_slice: bool
        If False, the returned label slice will have the same depth as the returned image slice. If True, the label slice will have
        depth 1 and correspond to the center plane of the returned image slice.
    slice_type: 'coronal', 'sagittal', 'axial', or None
        The anatomical plane of the returned slices. If None, a plane will be drawn at random for each volume.
    dtype: data-type
        The desired data-type for the returned arrays.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """
    return load_all_slices(source, label_required=label_required, depth=depth, single_label_slice=single_label_slice, slice_type=slice_type, dtype=dtype)


def load_all_slices(source, label_required=None, depth=1, single_label_slice=False, slice_type='axial', dtype=np.float32):
    """
    Loads all slices from the medical volumes specified by the provided paths and returns them one by one.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of file paths and a parameter dictionary.
    label_required: int or None
        A label that must be present in the returned slice. If depth is greater than 1, only the center plane will be checked.
    depth: int
        The depth of the slices. Must be an uneven integer greater than zero. If depth is greater than 1, the returned slices will
        overlap.
    single_label_slice: bool
        If False, the returned label slice will have the same depth as the returned image slice. If True, the label slice will have
        depth 1 and correspond to the center plane of the returned image slice.
    slice_type: 'coronal', 'sagittal', 'axial', or None
        The anatomical plane of the returned slices. If None, a plane will be drawn at random for each volume.
    dtype: data-type
        The desired data-type for the returned arrays.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    # If slice type is specified, set dimension
    if slice_type == 'coronal':
        dimension = 0
    elif slice_type == 'sagittal':
        dimension = 1
    elif slice_type == 'axial':
        dimension = 2
    elif slice_type is None:
        dimension = -1
    else:
        raise Error("Unknown slice_type: {}".format(slice_type))

    if depth < 1 or 2 % 1 != 0 or ((depth - 1) / 2) * 2 + 1 != depth:
        raise ValueError("Depth must be a positive uneven integer. Is: {}".format(depth))

    radius = (depth - 1) / 2

    def generator(dimension):

        for inputs, parameters in source:

            print "Loading {}".format(parameters["file_names"][0])

            # If slice type is not specified, set random dimension
            if slice_type is None:
                dimension = np.random.randint(3)

            image_volume = nibabel.load(inputs[0])
            header = image_volume.header

            image_volume = np.asarray(image_volume.dataobj).astype(dtype)
            label_volume = nibabel.load(inputs[1])
            label_volume = np.asarray(label_volume.dataobj).astype(dtype)

            parameters = __set_parameters(parameters, header, dimension)

            min_ = radius
            max_ = header.get_data_shape()[dimension] - radius

            for i in xrange(min_, max_):

                if dimension == 0:
                    label_slice = label_volume[i, :, :]
                elif dimension == 1:
                    label_slice = label_volume[:, i, :]
                elif dimension == 2:
                    label_slice = label_volume[:, :, i]

                if (not label_required is None) and (label_required not in label_slice):
                    continue

                outputs = []

                if depth > 1:

                    start_index = i - radius
                    end_index = i + radius

                    if dimension == 0:
                        image_slice = image_volume[start_index:end_index + 1, :, :]
                        image_slice = np.transpose(image_slice, axes=[1, 2, 0])
                    elif dimension == 1:
                        image_slice = image_volume[:, start_index:end_index + 1, :]
                        image_slice = np.transpose(image_slice, axes=[0, 2, 1])
                    elif dimension == 2:
                        image_slice = image_volume[:, :, start_index:end_index + 1]

                    if not single_label_slice:

                        if dimension == 0:
                            label_slice = label_volume[start_index:end_index + 1, :, :]
                            label_slice = np.transpose(label_slice, axes=[1, 2, 0])
                        elif dimension == 1:
                            label_slice = label_volume[:, start_index:end_index + 1, :]
                            label_slice = np.transpose(label_slice, axes=[0, 2, 1])
                        elif dimension == 2:
                            label_slice = label_volume[:, :, start_index:end_index + 1]
                
                else:

                    if dimension == 0:
                        image_slice = image_volume[i, :, :]
                    elif dimension == 1:
                        image_slice = image_volume[:, i, :]
                    elif dimension == 2:
                        image_slice = image_volume[:, :, i]

                outputs = [image_slice, label_slice]
                yield (outputs, parameters.copy())

    gen = generator(dimension)
    helper.sign_up(gen)
    return gen
