import dpp
import dpp.twodim.segmentation as seg
import dpp.threedim.segmentation.medical as vol


def lesions(data_dir, input_identifier, prediction_identifier, save_name=""):

    # Volumes for initialization #
    init = _lesions_initialization_generator(data_dir, input_identifier, prediction_identifier)

    with dpp.Pipeline(storage_name=save_name, initialization_generator=init) as pipe:

        node = dpp.reader.file_paths(data_dir, input_identifier=input_identifier, ground_truth_identifier=prediction_identifier, random=False, iterations=1)
        node = seg.medical.load_all_slices(node, slice_type='axial', depth=5, single_label_slice=False)
        
        # Adjust labels and colours
        node = seg.crop_to_label(node, 1, max_reduction=10, square_crop=True, default_label=0)

        node = seg.mask_img_background(node, 0, pixel_value=0.)
        node = seg.reduce_to_single_label_slice(node)
        node = seg.clip_img_values(node, 0., 200.)

        # Prepare as network input
        node = seg.resize(node, [256, 256])
        node = seg.transpose(node)

        node = seg.robust_img_scaling(node, ignore_values=[0., 200.], initialize=True)

    return pipe


def _lesions_initialization_generator(data_dir, input_identifier, prediction_identifier):

    with dpp.Pipeline() as init:
        node = dpp.reader.file_paths(data_dir, input_identifier=input_identifier, ground_truth_identifier=prediction_identifier, random=False, iterations=1)
        node = vol.load_volume(node)
        node = seg.mask_img_background(node, 0, pixel_value=0.)
        node = seg.clip_img_values(node, 0., 200.)

    return init


def liver(data_dir, input_identifier, prediction_identifier, save_name=""):

    # Volumes for initialization #
    init = _liver_initialization_generator(data_dir, input_identifier, prediction_identifier)

    with dpp.Pipeline(storage_name=save_name, initialization_generator=init) as pipe:

        node = dpp.reader.file_paths(data_dir, input_identifier=input_identifier, ground_truth_identifier=prediction_identifier, random=False, iterations=1)
        node = seg.medical.load_all_slices(node, slice_type='axial', depth=5, single_label_slice=False)

        node = seg.reduce_to_single_label_slice(node)
        node = seg.clip_img_values(node, -100., 400.)

        # Prepare as network input
        node = seg.resize(node, [256, 256])
        node = seg.transpose(node)

        node = seg.robust_img_scaling(node, ignore_values=[-100., 400.], initialize=True)

    return pipe


def _liver_initialization_generator(data_dir, input_identifier, prediction_identifier):

    with dpp.Pipeline() as init:
        node = dpp.reader.file_paths(data_dir, input_identifier=input_identifier, ground_truth_identifier=prediction_identifier, random=False, iterations=1)
        node = vol.load_volume(node)
        node = seg.clip_img_values(node, -100., 400.)

    return init
