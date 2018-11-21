# testing field for things

import cv2 as cv
import numpy as np
from PIL import Image
import random

import scipy.ndimage

import scipy.misc

from matplotlib import pyplot

from src.data_manager import load_img, simple_flow

from src.data_manager import tuples_from_custom

def vector_direction_deg(x, y):
    """
    direction of a vector in degrees in range [0, 360)
    """
    return (np.arctan2(x, y) * (180 / np.pi) + 180) % 360

def angle_difference(a1, a2):
    """
    return the difference in angle between two angles in range [0, 180]
    """
    abs_diff = abs(a1 - a2)
    return min(abs_diff, 360 - abs_diff)

def is_single_direction(flow, check_vectors_magnitude_ratio = 0.8, check_vectors_max_angle_difference=8, check_vectors_max_error_ratio=0.1):
    """
    return a tuple (True|False, avg_direction)
    """

    avg_direction = flow.sum(axis=(0,1))
    avg_direction /= np.linalg.norm(avg_direction)

    # create array of non-zero flow vectors
    flow_vecs = []
    for y in range(flow.shape[0]):
        for x in range(flow.shape[1]):
            vec = flow[y,x]
            # do not consider very small flow vectors
            if np.linalg.norm(flow[y,x] > 0.5):
                flow_vecs.append(flow[y, x])

    # too little movement in the patch
    if len(flow_vecs) < 100:
        return (False, avg_direction)

    avg_direction_angle = vector_direction_deg(avg_direction[0], avg_direction[1])

    #map into angle, magnitude pairs
    flow_vecs = list(map(lambda x: (vector_direction_deg(x[0], x[1]), np.linalg.norm(x)), flow_vecs))

    #sort the array by vector magnitudes
    flow_vecs = sorted(flow_vecs, key=lambda x: x[1], reverse=True)

    # find the sum of all magnitudes
    sum_magnitudes = 0
    for v in flow_vecs:
        sum_magnitudes += v[1]

    #check the vectors for an angle conforming with the avg direction by at most the required difference
    processed_magnitude = 0
    checked_vecs = 0
    num_check_fails = 0
    for v in flow_vecs:
        checked_vecs += 1
        processed_magnitude += v[1]
        if processed_magnitude > check_vectors_magnitude_ratio * sum_magnitudes:
            break

        diff = angle_difference(v[0], avg_direction_angle)
        if diff > check_vectors_max_angle_difference:
            num_check_fails += 1
    
    fail_ratio = num_check_fails / float(checked_vecs)

    return (fail_ratio < check_vectors_max_error_ratio, avg_direction)

print(len(tuples_from_custom("video/raw/frames")))

img1 = load_img("video/raw/frames/FjU_x1106pg/255000.jpg")
img2 = load_img("video/raw/frames/FjU_x1106pg/255004.jpg")

pil_to_numpy = lambda x: np.array(x)[:, :, ::-1]

img1 = pil_to_numpy(img1)
img2 = pil_to_numpy(img2)

#flow = cv.optflow.calcOpticalFlowSF(img1, img2, layers=3, averaging_block_size=2, max_flow=4)

img_h = img1.shape[0]
img_w = img1.shape[1]
patch_h = 150
patch_w = 150
flow_threshold = 6

assert(patch_h == patch_w)

patch_diagonal = int(((patch_w ** 2 + patch_h ** 2) ** 0.5)+1)

patch_num_w = int(img_w / patch_diagonal)
patch_num_h = int(img_h / patch_diagonal)

print(f"check {patch_num_h * patch_num_w} patches per image")

patch_cover_w = patch_num_w * patch_diagonal
patch_cover_h = patch_num_h * patch_diagonal

right_space = (img_w - patch_cover_w) // 2
top_space = (img_h - patch_cover_h) // 2

for pw in range(patch_num_w):
    for ph in range(patch_num_h):
        i = top_space + ph * patch_diagonal
        j = right_space + pw * patch_diagonal

        img1_patch = img1[i:i+patch_diagonal, j:j + patch_diagonal]
        img2_patch = img2[i:i+patch_diagonal, j:j + patch_diagonal]

        patch = cv.optflow.calcOpticalFlowSF(img1_patch, img2_patch, layers=3, averaging_block_size=2, max_flow=4)

        n = np.sum(1 - np.isnan(patch), axis=(0,1))
        patch[np.isnan(patch)] = 0

        flow_magnitude = np.linalg.norm( patch.sum(axis=(0,1)) / n)

        if random.random() < flow_magnitude / flow_threshold:
            result = is_single_direction(patch, check_vectors_magnitude_ratio = 0.4, check_vectors_max_angle_difference=12, check_vectors_max_error_ratio=0.40)
            if result[0]:
                avg_direction = result[1]

                anglePreRotation = vector_direction_deg(avg_direction[0], avg_direction[1])

                targetRotation = 90 if random.random() > 0.5 else 270

                r = targetRotation - anglePreRotation

                i1rotated = scipy.ndimage.interpolation.rotate(img1_patch, r, reshape=False)
                i2rotated = scipy.ndimage.interpolation.rotate(img2_patch, r, reshape=False)

                mx = patch_diagonal // 2
                my = patch_diagonal // 2

                halfW = patch_w // 2
                halfH = patch_h // 2

                final_patch1 = i1rotated[my - halfH : my + halfH, mx - halfW : mx + halfW]
                final_patch2 = i2rotated[my - halfH : my + halfH, mx - halfW : mx + halfW]

                _, axarr = pyplot.subplots(2,2)

                scipy.misc.imsave('/ImbaKeks/final_patch1.png', final_patch1)
                scipy.misc.imsave('/ImbaKeks/final_patch2.png', final_patch2)

                axarr[0,0].imshow(img1_patch)
                axarr[1,0].imshow(final_patch1)
                axarr[0,1].imshow(img2_patch)
                axarr[1,1].imshow(final_patch2)

                pyplot.show()






