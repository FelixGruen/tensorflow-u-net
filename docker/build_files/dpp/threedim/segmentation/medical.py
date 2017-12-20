import nibabel
import numpy as np

from ... import helper


def load_volume(source):
    """
    Loads the medical volumes specified by the provided paths.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of file paths and a parameter dictionary.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        outputs = []
        niftis = [nibabel.load(inpt) for inpt in inputs]
        outputs = [np.asarray(nifti.dataobj).astype(np.float32) for nifti in niftis]

        nifti = niftis[0]

        parameters["spacing"] = nifti.header.get_zooms()[:3]  # the last value is the time between scans, no need to keep it
        parameters["original_spacing"] = parameters["spacing"]

        parameters["size"] = nifti.header.get_data_shape()
        parameters["original_size"] = parameters["size"]

        parameters["nifti_header"] = nifti.header

        return (outputs, parameters)

    return helper.apply(source, transformation)


def random_deform(source, control_points, std_def, probability=1.0):
    """
    Applys random b-spline transformations to a random selection of the inputs.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of file paths and a parameter dictionary.
    control_points: int
        The number of control points.
    std_def: float
        The standard deformation.
    probability: float
        The probability of applying the deformation to the input. If it is below 1, some inputs will be passed through unchanged.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """

    # Conditional import
    import SimpleITK as sitk

    def transformation(input_tuple):
        inputs, parameters = input_tuple

        if(np.random.rand() < probability):  # do not apply deformations always, just sometimes

            minmax = sitk.MinimumMaximumImageFilter()
            minmax.Execute(inputs[0])
            minimum = minmax.GetMinimum()

            mesh_size = [control_points] * inputs[0].GetDimension()

            transform = sitk.BSplineTransformInitializer(inputs[0], mesh_size)

            params = np.asarray(transform.GetParameters(), dtype=float)
            params = params + np.random.randn(params.size) * std_def
            params[0:(len(params) / 3)] = 0.0  # remove z deformations! The resolution in z is usually too bad

            transform.SetParameters(tuple(params))

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(inputs[0])
            resampler.SetDefaultPixelValue(minimum)
            resampler.SetTransform(transform)
            resampler.SetInterpolator(sitk.sitkLinear)

            inputs[0] = resampler.Execute(inputs[0])

            minmax.Execute(inputs[1])
            minimum = minmax.GetMinimum()

            resampler.SetDefaultPixelValue(minimum)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)

            inputs[1] = resampler.Execute(inputs[1])

        return (inputs, parameters)

    return helper.apply(source, transformation)
