import numpy as np
import cv2
import os


def calculate_PDF(colorChannel):
    totalIntensities = 0
    pdf = np.zeros((256, 1))

    for i in range(colorChannel.shape[0]):
        for j in range(colorChannel.shape[1]):
            intensity_value = colorChannel[i, j]
            pdf[intensity_value] += 1
            totalIntensities += 1

    return pdf / float(totalIntensities)


# Method that calculates CDF from PDF
def calculate_CDF(inputChannel):
    pdf = calculate_PDF(inputChannel)
    cdf = np.zeros((256, 1))
    for i in range(256):
        cdf[i] = sum(pdf[0: i + 1])

    return cdf


# Method that matches the histogram of single channel
# Returns the matched histogram output
def match_single_channel_histogram(color_channel_input, color_channel_target):
    cdf_input = calculate_CDF(color_channel_input)
    cdf_target = calculate_CDF(color_channel_target)
    look_up_table = np.zeros((256, 1))
    gj = 0
    for gi in range(256):
        while cdf_target[gj] < cdf_input[gi] and gj < 255:
            gj += 1
        look_up_table[gi] = gj

    color_channel_output = np.uint8(look_up_table[color_channel_input])
    color_channel_output = color_channel_output.reshape(color_channel_output.shape[0], color_channel_output.shape[1])
    return color_channel_output


# Method that calculates the histogram of single channel
# Returns a histogram vector
def histogram_calculator(color_channel):
    histogram_vector = np.zeros((256, 1))
    for i in range(color_channel.shape[0]):
        for j in range(color_channel.shape[1]):
            histogram_vector[color_channel[i, j]] += 1
    return histogram_vector


def merge_color_channels(channel_1, channel_2, channel_3):
    output_image = cv2.merge((channel_1, channel_2, channel_3))
    return output_image


def histogram_match(_input_path=None, _ref_path=None, format="RGB", do_channel_1=True, do_channel_2=True,
                    do_chanenel_3=True):
    input_img = cv2.imread(_input_path)
    ref_img = cv2.imread(_ref_path)

    if format == "BGR":
        pass
    elif format == "HSV":
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)

    elif format == "HLS":
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HLS)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HLS)

    elif format == "Lab":
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2Lab)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2Lab)

    elif format == "Luv":
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2Luv)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2Luv)

    elif format == "YCC":
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCrCb)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2YCrCb)

    input_channel_1, input_channel_2, input_channel_3 = cv2.split(input_img)
    ref_channel_1, ref_channel_2, ref_channel_3 = cv2.split(ref_img)

    if do_channel_1:
        output_channel_1 = match_single_channel_histogram(input_channel_1, ref_channel_1)
    else:
        output_channel_1 = input_channel_1

    if do_channel_2:
        output_channel_2 = match_single_channel_histogram(input_channel_2, ref_channel_2)
    else:
        output_channel_2 = input_channel_2

    if do_chanenel_3:
        output_channel_3 = match_single_channel_histogram(input_channel_3, ref_channel_3)
    else:
        output_channel_3 = input_channel_3

    output_img = cv2.merge((output_channel_1, output_channel_2, output_channel_3))

    if format == "BGR":
        pass
    elif format == "HSV":
        output_img = cv2.cvtColor(output_img, cv2.COLOR_HSV2BGR)

    elif format == "HLS":
        output_img = cv2.cvtColor(output_img, cv2.COLOR_HLS2BGR)

    elif format == "Lab":
        output_img = cv2.cvtColor(output_img, cv2.COLOR_Lab2BGR)

    elif format == "Luv":
        output_img = cv2.cvtColor(output_img, cv2.COLOR_Luv2BGR)

    elif format == "YCC":
        output_img = cv2.cvtColor(output_img, cv2.COLOR_YCrCb2BGR)

    return output_img


def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)


def match_channel_histogram_fast(_input_path, _ref_path, multichannel=False):

    image = cv2.imread(_input_path)
    reference = cv2.imread(_ref_path)

    # shape = image.shape
    # image_dtype = image.dtype

    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number of channels.')

    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and reference '
                             'image must match!')

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel], reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference)

    return matched


if __name__ == '__main__':
    src_path = r".\img_histogram_match\src.jpg"
    ref_path = r".\img_histogram_match\ref.jpg"
    res_dir = r".\img_histogram_match\res"

    img = histogram_match(_input_path=src_path, _ref_path=ref_path, format="BGR")
    cv2.imwrite(os.path.join(res_dir, "BGR.jpg"), img)

    img = match_channel_histogram_fast(_input_path=src_path, _ref_path=ref_path, multichannel=True)
    cv2.imwrite(os.path.join(res_dir, "BGR_fast.jpg"), img)

    img = histogram_match(_input_path=src_path, _ref_path=ref_path, format="HSV")
    cv2.imwrite(os.path.join(res_dir, "HSV.jpg"), img)

    img = histogram_match(_input_path=src_path, _ref_path=ref_path, format="HLS")
    cv2.imwrite(os.path.join(res_dir, "HLS.jpg"), img)

    img = histogram_match(_input_path=src_path, _ref_path=ref_path, format="Lab")
    cv2.imwrite(os.path.join(res_dir, "Lab.jpg"), img)

    img = histogram_match(_input_path=src_path, _ref_path=ref_path, format="Luv")
    cv2.imwrite(os.path.join(res_dir, "Luv.jpg"), img)

    img = histogram_match(_input_path=src_path, _ref_path=ref_path, format="YCC")
    cv2.imwrite(os.path.join(res_dir, "YCC.jpg"), img)





