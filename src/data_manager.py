#
# KTH Royal Institute of Technology
#

import json
import random
import zipfile
import numpy as np
import cv2 as cv
from joblib import Parallel, delayed
from timeit import default_timer as timer
from torchvision.transforms.functional import crop as crop_image
from torchvision.transforms.functional import rotate as rotate_image
import torchvision.transforms.functional
from os.path import exists, join, basename, isdir, splitext
from os import makedirs, remove, listdir, rmdir, rename
from six.moves import urllib
from PIL import Image

import time

import torch
from .flownet2.models import FlowNet2CS

import src.config as config


############################################# UTILITIES #############################################

def load_img(file_path):
    """
    Reads an image from disk.
    :param file_path: Path to the image file
    :return: PIL.Image object
    """
    return Image.open(file_path).convert('RGB')


def is_image(file_path):
    """
    Checks if the specified file is an image
    :param file_path: Path to the candidate file
    :return: Whether or not the file is an image
    """
    return any(file_path.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_images(directory_path):
    """
    Reads all images that are directly inside the given directory as pil images
    """
    results = []

    image_files = []

    for f in listdir(directory_path):
        if is_image(f):
            image_files.append(f)

    # don't want to trust whatever sorting listdir comes up with. Files are named as frame numbers and should be sorted as such
    image_files = sorted(image_files, key=lambda x: int(splitext(x)[0]))

    for f in image_files:
        f = join(directory_path, f)
        results.append(load_img(f))

    return results

def load_tuples(root_path, stride, tuple_size, paths_only=True):
    """
    Reads the content of a directory coupling the files together in tuples.
    :param root_path: Path to the directory
    :param stride: Number of steps from one tuple to the next
    :param tuple_size: Size of each tuple
    :param paths_only: If true, the tuples will contain paths rather than PIL.Image objects
    :return: List of tuples containing the images or their paths
    """

    frames = [join(root_path, x) for x in listdir(root_path)]
    frames = [x for x in frames if is_image(x)]
    frames.sort()

    if not paths_only:
        frames = [load_img(x) for x in frames]

    tuples = []
    for i in range(1 + (len(frames) - tuple_size) // stride):
        tuples.append(tuple(frames[i * stride + j] for j in range(tuple_size)))

    return tuples

def load_patch(patch):
    """
    Reads the three images of a patch from disk and returns them already cropped.
    :param patch: Dictionary containing the details of the patch
    :return: Tuple of PIL.Image objects corresponding to the patch
    """

    if "frames" in patch:
        paths = patch["frames"]
    else:
        paths = (patch['left_frame'], patch['middle_frame'], patch['right_frame'])
    i, j = (patch['patch_i'], patch['patch_j'])
    h, w = config.PATCH_SIZE
    imgs = [load_img(x) for x in paths]

    if not patch['custom']:
        return tuple(crop_image(x, i, j, h, w) for x in imgs)
    else:
        diag = patch['patch_diagonal']
        angle = patch['rotation']
        # first crop out the region to be rotated
        cropped = [crop_image(x, i, j, diag, diag) for x in imgs]

        mx = diag // 2
        my = diag // 2

        half_w = w // 2
        half_h = h // 2

        top = my - half_h
        left = mx - half_w

        # then rotate
        rotated = [rotate_image(x, angle, resample=Image.BICUBIC) for x in cropped]
        final_crop = [crop_image(x, top, left, h, w) for x in rotated]

        # at last crop out the center pixels that make up the finished patch
        return tuple(final_crop)


def load_cached_patch(cached_patch):
    """
    Reads the cached images of a patch from disk. Can only be used if the patches
    have been previously cached.
    :param cached_patch: Patch as a tuple (path_to_left, path_to_middle, path_to_right)
    :return: Tuple of PIL.Image objects corresponding to the patch
    """
    return tuple(load_img(x) for x in cached_patch)


############################################### DAVIS ###############################################

def get_davis_16(dataset_dir):
    return _get_davis(dataset_dir, "DAVIS", "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip")


def get_davis_17_test(dataset_dir):
    return _get_davis(dataset_dir, "DAVIS17-test", "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip")


def get_davis_17(dataset_dir):
    return _get_davis(dataset_dir, "DAVIS17", "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip")


def _get_davis(dataset_dir, folder, url):
    """
    Returns the local path to the DAVIS dataset, given its root directory. The dataset
    is downloaded if not found on disk.
    :param dataset_dir: Path to the dataset directory
    :return: Path to the DAVIS dataset
    """
    davis_dir = join(dataset_dir, folder)
    tmp_dir = join(dataset_dir, 'tmp')

    if not exists(davis_dir):

        if not exists(dataset_dir):
            makedirs(dataset_dir)

        if not exists(tmp_dir):
            makedirs(tmp_dir)

        print("===> Downloading {}...".format(folder))
        response = urllib.request.urlopen(url)
        zip_path = join(dataset_dir, basename(url))
        with open(zip_path, 'wb') as f:
            f.write(response.read())

        zip_ref = zipfile.ZipFile(zip_path, 'r')

        print("===> Extracting data...")
        zip_ref.extractall(tmp_dir)
        zip_ref.close()

        # Move folder to desired path
        extracted_folder = join(tmp_dir, listdir(tmp_dir)[0])
        rename(extracted_folder, davis_dir)

        # Remove temporary files
        remove(zip_path)
        rmdir(tmp_dir)

    return davis_dir


def tuples_from_davis(davis_dir, res='480p'):
    """
    Finds all images of the specified resolution from the DAVIS dataset. The found paths
    are returned as tuples of three elements.
    :param davis_dir: Path to the DAVIS dataset directory
    :param res: Resolution of the DAVIS images (either '480p' or '1080p')
    :return: List of paths as tuples (path_to_left, path_to_middle, path_to_right)
    """

    subdir = join(davis_dir, "JPEGImages/" + res)

    video_dirs = [join(subdir, x) for x in listdir(subdir)]
    video_dirs = [x for x in video_dirs if isdir(x)]

    tuples = []
    for video_dir in video_dirs:

        frame_paths = [join(video_dir, x) for x in listdir(video_dir)]
        frame_paths = [x for x in frame_paths if is_image(x)]
        frame_paths.sort()

        for i in range(len(frame_paths) // 3):
            x1, t, x2 = frame_paths[i * 3], frame_paths[i * 3 + 1], frame_paths[i * 3 + 2]
            tuples.append((x1, t, x2))

    return tuples

def tuples_from_custom(custom_dir):
    tuples = []

    for f in listdir(custom_dir):
        if isdir(join(custom_dir, f)) and not f == "cache" and not f == "tmp":
            image_files = listdir(join(custom_dir, f))
            image_files = [x for x in image_files if is_image(join(custom_dir, f, x))]
            # don't want to trust whatever sorting listdir comes up with. Files are named as frame numbers and should be sorted as such
            image_files = sorted(image_files, key=lambda x: int(splitext(x)[0]))
            
            frame_paths = [join(custom_dir, f, x) for x in image_files]

            stride = config.CUSTOM_STRIDE # number of frames between frames packed togather as tuples
            frameskip = config.CUSTOM_FRAMESKIP # number of frames skipped after a set of frames used for a tuple

            added = 0

            i = 0
            while i + config.MAX_SEQUENCE_LENGTH * stride < len(frame_paths):
                tuple = []
                for tindex in range(config.MAX_SEQUENCE_LENGTH):
                    tuple.append(frame_paths[i + tindex * stride])
                added += 1
                tuples.append(tuple)
                i += config.MAX_SEQUENCE_LENGTH * stride
                i += frameskip

            print(f"Got {added} tuples from {join(custom_dir, f)}")

    print(f"Will consider {len(tuples)} custom tuples")

    return tuples


def get_selected_davis(dataset_dir=None, res='480p'):

    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR

    davis16_dir = get_davis_16(dataset_dir)
    root = join(davis16_dir, 'JPEGImages', res)

    tuples = [
        ('horsejump-low/00030.jpg', 'horsejump-low/00031.jpg', 'horsejump-low/00032.jpg'),
        ('parkour/00069.jpg', 'parkour/00070.jpg', 'parkour/00071.jpg'),
        ('breakdance/00060.jpg', 'breakdance/00061.jpg', 'breakdance/00062.jpg'),
        ('drift-turn/00045.jpg', 'drift-turn/00046.jpg', 'drift-turn/00047.jpg'),
        ('rhino/00027.jpg', 'rhino/00028.jpg', 'rhino/00029.jpg'),
        ('motocross-jump/00009.jpg', 'motocross-jump/00010.jpg', 'motocross-jump/00011.jpg'),
        ('flamingo/00006.jpg', 'flamingo/00007.jpg', 'flamingo/00008.jpg'),
        ('scooter-black/00027.jpg', 'scooter-black/00028.jpg', 'scooter-black/00029.jpg'),
        ('boat/00006.jpg', 'boat/00007.jpg', 'boat/00008.jpg'),
        ('dance-twirl/00054.jpg', 'dance-twirl/00055.jpg', 'dance-twirl/00056.jpg')
    ]

    return [tuple(join(root, y) for y in x) for x in tuples]


########################################## PATCH EXTRACTION #########################################

def simple_flow(frame1, frame2):
    """
    Runs SimpleFlow given two consecutive frames.
    :param frame1: Numpy array of the frame at time t
    :param frame2: Numpy array of the frame at time t+1
    :return: Numpy array with the flow for each pixel. Shape is same as input
    """
    flow = cv.optflow.calcOpticalFlowSF(frame1, frame2, layers=3, averaging_block_size=2, max_flow=4)
    n = np.sum(1 - np.isnan(flow), axis=(0, 1))
    flow[np.isnan(flow)] = 0
    return np.linalg.norm(np.sum(flow, axis=(0, 1)) / n)


def is_jumpcut(frame1, frame2, threshold=np.inf):
    """
    Detects a jumpcut between the two frames.
    :param frame1: Numpy array of the frame at time t
    :param frame2: Numpy array of the frame at time t+1
    :param threshold: Maximum difference allowed for the frames to be considered consecutive
    :return: Whether or not there is a jumpcut between the two frames
    """
    pixels_per_channel = frame1.size / 3
    hist = lambda x: np.histogram(x.reshape(-1), 8, (0, 255))[0] / pixels_per_channel
    err = lambda a, b: ((hist(a) - hist(b)) ** 2).mean()

    return err(frame1[:, :, 0], frame2[:, :, 0]) > threshold or \
           err(frame1[:, :, 1], frame2[:, :, 1]) > threshold or \
           err(frame1[:, :, 2], frame2[:, :, 2]) > threshold


def _extract_patches_worker(tuples, max_per_frame=1, trials_per_tuple=100, flow_threshold=0.0,
                            jumpcut_threshold=np.inf):
    """
    Extracts small patches from the original frames. The patches are selected to maximize
    their contribution to the training.
    :param tuples: List of tuples containing the input frames as (left, middle, right)
    :param max_per_frame: Maximum number of patches that can be extracted from a frame
    :param trials_per_tuple: Number of random crops to test for each tuple
    :param flow_threshold: Minimum average optical flow for a patch to be selected
    :param jumpcut_threshold: ...
    :return: List of dictionaries representing each patch
    """

    patch_h, patch_w = config.PATCH_SIZE
    n_tuples = len(tuples)
    all_patches = []
    jumpcuts = 0
    flowfiltered = 0
    total_iters = n_tuples * trials_per_tuple

    pil_to_numpy = lambda x: np.array(x)[:, :, ::-1]

    for tup_index in range(n_tuples):
        tup = tuples[tup_index]

        left, middle, right = (load_img(x) for x in tup)
        img_w, img_h = left.size

        left = pil_to_numpy(left)
        middle = pil_to_numpy(middle)
        right = pil_to_numpy(right)

        selected_patches = []

        for _ in range(trials_per_tuple):

            i = random.randint(0, img_h - patch_h)
            j = random.randint(0, img_w - patch_w)

            left_patch = left[i:i + patch_h, j:j + patch_w, :]
            right_patch = right[i:i + patch_h, j:j + patch_w, :]
            middle_patch = middle[i:i + patch_h, j:j + patch_w, :]

            if is_jumpcut(left_patch, middle_patch, jumpcut_threshold) or \
                    is_jumpcut(middle_patch, right_patch, jumpcut_threshold):
                jumpcuts += 1
                continue

            avg_flow = simple_flow(left_patch, right_patch)
            if random.random() > avg_flow / flow_threshold:
                flowfiltered += 1
                continue

            selected_patches.append({
                "left_frame": tup[0],
                "middle_frame": tup[1],
                "right_frame": tup[2],
                "patch_i": i,
                "patch_j": j,
                "avg_flow": avg_flow,
                "custom": False
            })

        selected_patches = sorted(selected_patches, key=lambda x: x['avg_flow'], reverse=True)
        all_patches += selected_patches[:max_per_frame]
        # print("===> Tuple {}/{} ready.".format(tup_index+1, n_tuples))

    print('===> Processed {} tuples, {} patches extracted, {} discarded as jumpcuts, {} filtered by flow'.format(
        n_tuples, len(all_patches), 100.0 * jumpcuts / total_iters, 100.0 * flowfiltered / total_iters
    ))

    return all_patches

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

# for debugging usage, have a look at generated flows
def writeFlowImage(flowArray, filePath):
    flow = flowArray

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros(flow.shape[:2] + (3,), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

    img = Image.fromarray(rgb, mode='RGB')
    img.save(filePath, 'JPEG', quality=95)

def getPadFlowNet2InputBatch(frameWidth, frameHeight):
    reqDivisibility = 64

    paddedWidth = frameWidth
    paddedHeight = frameHeight

    if paddedWidth % reqDivisibility != 0:
        paddedWidth = (paddedWidth // reqDivisibility + 1) * reqDivisibility

    if paddedHeight % reqDivisibility != 0:
        paddedHeight = (paddedHeight // reqDivisibility + 1) * reqDivisibility

    wPad = paddedWidth - frameWidth
    hPad = paddedHeight - frameHeight

    left = wPad // 2
    right = wPad - left

    top = hPad // 2
    bottom = hPad - top

    inputPad = torch.nn.ReplicationPad2d([left, right, top, bottom])
    outputPad = torch.nn.ReplicationPad2d([-left, -right, -top, -bottom])

    return inputPad, outputPad

class AttributeHolder(object):
    pass

flowNet2 = None

# call this once before starting to use flownet2 to init global flownet2 instance
def initFlowNet2():
    global flowNet2

    print("===> Init Flownet2...")

    args = AttributeHolder()
    args.rgb_max = 255
    args.fp16 = False

    flowNet2 = FlowNet2CS(args).cuda()
    netState = torch.load(config.FLOWNET2_CS_TRAINED_WEIGHTS)
    flowNet2.load_state_dict(netState["state_dict"])
    
    print("===> Flownet 2 initialized")

# call after flowNet2 usage
def freeFlowNet2():
    global flowNet2
    del flowNet2
    flowNet2 = None
    torch.cuda.empty_cache()

def callFlowNet2(numpyInputBatch):
    global flowNet2

    im = torch.from_numpy(numpyInputBatch).cuda()

    with torch.no_grad():
        # pad inputs to image size that is a multiple of 64 for the network
        inPad, outPad = getPadFlowNet2InputBatch(im.shape[4], im.shape[3])
        inPad = inPad.cuda()
        outPad = outPad.cuda()
        bDims = im.shape
        im = inPad(im.reshape(bDims[0], bDims[1] * bDims[2], bDims[3], bDims[4]))
        im = im.reshape(bDims[0], bDims[1], bDims[2], im.shape[2], im.shape[3])

        #process the image pair to obtain the flow
        netOut = flowNet2(im)

        #remove any padding applied before the input
        result = outPad(netOut)

    return result.data.cpu().numpy()

def to_numpy(x):
    return np.array(x)

def _extract_custom_patches_worker_flownet(tuples, flow_threshold, jumpcut_threshold, force_horizontal):
    assert not config.FORCE_HORIZONTAL, "flow horizontal is not supported with flownet dataset creation, TODO: maybe change that"

    patch_h, patch_w = config.PATCH_SIZE

    assert(patch_h == patch_w)

    patch_diagonal = int(((patch_w ** 2 + patch_h ** 2) ** 0.5)+1)

    patches = []

    jumpcuts = 0
    flowfiltered = 0
    directionfiltered = 0

    total_count = 0

    lastProcTimes = []

    # processing tuples:
    # for each tuple collect candidate patch indices
    #   -> that do not contain jump cuts
    # then push the candidate patches (on default settings it will be 3 sequences of 9 patches, handle these as one batch) through flownet2.
    # flownet2 input has this shape: [BATCH_INDEX, RGB_CHANNELS=3, IMAGE_PAIR=2, HEIGHT, WIDTH]
    # flownet2 output has the shape: [BATCH_INDEX, VECTOR_AXIS=2, HEIGHT, WIDTH]

    for tup_index in range(len(tuples)):
        procStart = time.time()

        tup = tuples[tup_index]

        imgs = [load_img(x) for x in tup]

        img_w, img_h = imgs[0].size

        imgs = [to_numpy(x) for x in imgs]

        patch_num_w = int(img_w / patch_diagonal)
        patch_num_h = int(img_h / patch_diagonal)

        right_space = (img_w - patch_num_w * patch_diagonal) // 2
        top_space = (img_h - patch_num_h * patch_diagonal) // 2

        # contains lists: [i, j, candidate_seq_list_of_patch_images]
        # if too many of the optical flows are determined to be too small the sequence is not used
        candidatePatches = []

        for pw in range(patch_num_w):
            for ph in range(patch_num_h):
                
                total_count += 1

                i = top_space + ph * patch_diagonal
                j = right_space + pw * patch_diagonal

                candidate_seq = [x[i:i+patch_diagonal, j:j+patch_diagonal] for x in imgs]

                skip = False

                for pair_idx in range(1, len(imgs)):
                    if is_jumpcut(candidate_seq[pair_idx-1], candidate_seq[pair_idx], jumpcut_threshold):
                        jumpcuts += 1
                        skip = True
                        break

                if skip:
                    continue # next candidate position

                candidatePatches.append([i, j, candidate_seq])

        if len(candidatePatches) == 0:
            continue # next tuple

        # use flownet2 to find flows of the relevant patches
        netInputData = np.zeros((len(candidatePatches) * (len(imgs)-2), 3, 2, patch_diagonal, patch_diagonal), dtype=np.float32)
        bIndex = 0
        for candidate in candidatePatches:
            candidate_seq = candidate[2]
            for pair_idx in range(2, len(imgs)):
                img1 = candidate_seq[pair_idx - 2]
                img2 = candidate_seq[pair_idx]
                imagePair = [img1, img2]
                imagePair = np.array(imagePair)
                imagePair = imagePair.transpose(3, 0, 1, 2)
                netInputData[bIndex] = imagePair

                bIndex += 1

        netOutputData = callFlowNet2(netInputData)

        bIndex = 0

        # set to a directory to output image patches and flow visualization of the patches to verify everything works as expected
        dbgOutputDir = None 

        for candidate in candidatePatches:
            candidate_seq = candidate[2]

            flow_fails = 0

            for pair_idx in range(2, len(imgs)):
                flowByNetwork = netOutputData[bIndex].transpose(1, 2, 0)

                # nan handling is from simple flow stuff. Probably not even required, why would a neural net output NaNs anyway?
                # but better safe than sorry
                n = np.sum(1 - np.isnan(flowByNetwork), axis=(0,1))
                flowByNetwork[np.isnan(flowByNetwork)] = 0
                flow_magnitude = np.linalg.norm(flowByNetwork.sum(axis=(0,1)) / n)

                bIndex += 1

                if random.random() > flow_magnitude / flow_threshold:
                    flow_fails += 1

                if flow_magnitude > config.MAX_FLOW:
                    flow_fails += len(imgs) # block it out completely!
                    # do NOT break to not screw up bIndex counting

                if dbgOutputDir is not None and flow_magnitude < config.MAX_FLOW and flow_magnitude > config.FLOW_THRESHOLD:
                    iLeft = candidate_seq[pair_idx - 2]
                    iRight = candidate_seq[pair_idx]

                    print(tup_index, bIndex, flow_magnitude)
                    nameBase = str(tup_index) + "_" + str(bIndex) + "_"
                    leftName = nameBase + "1left.jpg"
                    rightName = nameBase + "2right.jpg"
                    flowName = nameBase + "3flow.jpg"

                    writeFlowImage(flowByNetwork, join(dbgOutputDir, flowName))
                    Image.fromarray(iLeft, mode='RGB').save(join(dbgOutputDir, leftName), 'JPEG')
                    Image.fromarray(iRight, mode='RGB').save(join(dbgOutputDir, rightName), 'JPEG')
            
            if flow_fails >= len(imgs) // 3:
                flowfiltered += 1
            else:
                patches.append({
                    "frames": tup,
                    "patch_i": candidate[0],
                    "patch_j": candidate[1],
                    "patch_diagonal": patch_diagonal,
                    "rotation": 0,
                    "custom": True
                })

        procTime = time.time() - procStart
        
        lastProcTimes.append(procTime)
        if len(lastProcTimes) > 250:
            del lastProcTimes[0]

        timePerTupleAvg = np.mean(lastProcTimes)
        tuplesPerSec = 1.0 / timePerTupleAvg
        remTuples = len(tuples) - tup_index - 1
        remSecs = remTuples / tuplesPerSec
        remMins = remSecs // 60 + 1
        remHours = int(remMins // 60)
        remMins = int(remMins % 60)

        print(f"Worker starting at frame {basename(tuples[0][0])} is {100.0 * tup_index / len(tuples)} % complete with {len(patches)} interessting patches found. {tuplesPerSec:.2f} tuples per second. ETA {remHours} hours, {remMins} minutes")

    print('===> Processed {} tuples, {} patches extracted, {} discarded as jumpcuts, {} filtered by flow, {} filtered by direction'.format(
        len(tuples), len(patches), 100.0 * jumpcuts / total_count, 100.0 * flowfiltered / total_count, 100.0 * directionfiltered / total_count
    ))

    return patches

# not used, this is the old SimpleFlow based version, used instead is the flownet2 version, which is ~50x faster.
def _extract_custom_patches_worker(tuples, flow_threshold, jumpcut_threshold, force_horizontal):
    assert config.MAX_SEQUENCE_LENGTH == 3 or not config.FORCE_HORIZONTAL, "using a sequence length of above 3 with force horizontal is not implemented to work correctly"

    patch_h, patch_w = config.PATCH_SIZE

    assert(patch_h == patch_w)

    patch_diagonal = int(((patch_w ** 2 + patch_h ** 2) ** 0.5)+1)

    # this turns the image into a BGR image.
    pil_to_numpy = lambda x: np.array(x)[:, :, ::-1]

    patches = []

    jumpcuts = 0
    flowfiltered = 0
    directionfiltered = 0
    total_count = 0

    lastProcTimes = []

    for tup_index in range(len(tuples)):
        procStart = time.time()

        tup = tuples[tup_index]

        imgs = [load_img(x) for x in tup]

        img_w, img_h = imgs[0].size

        imgs = [pil_to_numpy(x) for x in imgs]
        patch_num_w = int(img_w / patch_diagonal)
        patch_num_h = int(img_h / patch_diagonal)

        right_space = (img_w - patch_num_w * patch_diagonal) // 2
        top_space = (img_h - patch_num_h * patch_diagonal) // 2

        for pw in range(patch_num_w):
            for ph in range(patch_num_h):

                # don't process all parts of all frames, increase variation accross more scenes... also save some cpu time. Not needed when using flownet
                if random.random() > 0.7:
                    continue

                total_count += 1

                i = top_space + ph * patch_diagonal
                j = right_space + pw * patch_diagonal

                candidate_seq = [x[i:i+patch_diagonal, j:j+patch_diagonal] for x in imgs]

                skip = False

                for pair_idx in range(1, len(imgs)):
                    if is_jumpcut(candidate_seq[pair_idx-1], candidate_seq[pair_idx], jumpcut_threshold):
                        jumpcuts += 1
                        skip = True
                        break

                if skip:
                    continue

                flow_fails = 0

                for pair_idx in range(2, len(imgs)):
                    img1 = candidate_seq[pair_idx - 2]
                    img2 = candidate_seq[pair_idx]

                    flow = cv.optflow.calcOpticalFlowSF(img1, img2, layers=3, averaging_block_size=2, max_flow=4)

                    n = np.sum(1 - np.isnan(flow), axis=(0,1))

                    flow[np.isnan(flow)] = 0

                    flow_magnitude = np.linalg.norm(flow.sum(axis=(0,1)) / n)

                    rx = random.random()
                    if rx > flow_magnitude / flow_threshold:
                        flow_fails += 1

                    if force_horizontal:
                        dir_ok, avg_direction = is_single_direction(flow, check_vectors_magnitude_ratio=0.20, 
                            check_vectors_max_angle_difference=10, check_vectors_max_error_ratio=0.25)
                        if not dir_ok:
                            directionfiltered += 1
                            skip = True
                            break

                        anglePreRotation = vector_direction_deg(avg_direction[0], avg_direction[1])

                        targetRotation = 90 if random.random() > 0.5 else 270

                        rotation = targetRotation - anglePreRotation
                    else:
                        rotation = 0

                if flow_fails >= len(imgs) // 3:
                    flowfiltered += 1
                    continue

                patches.append({
                    "frames": tup,
                    "patch_i": i,
                    "patch_j": j,
                    "patch_diagonal": patch_diagonal,
                    "rotation": rotation,
                    "custom": True
                })

        procTime = time.time() - procStart

        lastProcTimes.append(procTime)
        if len(lastProcTimes) > 250:
            del lastProcTimes[0]

        timePerTupleAvg = np.mean(lastProcTimes)
        tuplesPerSec = 1.0 / timePerTupleAvg
        remTuples = len(tuples) - tup_index - 1
        remSecs = remTuples / tuplesPerSec
        remMins = remSecs // 60 + 1
        remHours = int(remMins // 60)
        remMins = int(remMins % 60)

        print(f"Worker starting at frame {basename(tuples[0][0])} is {100.0 * tup_index / len(tuples)} % complete with {len(patches)} interessting patches found. {tuplesPerSec:.2f} tuples per second. ETA {remHours} hours, {remMins} minutes")

    print('===> Processed {} tuples, {} patches extracted, {} discarded as jumpcuts, {} filtered by flow, {} filtered by direction'.format(
        len(tuples), len(patches), 100.0 * jumpcuts / total_count, 100.0 * flowfiltered / total_count, 100.0 * directionfiltered / total_count
    ))

    return patches
        

def _extract_patches(tuples, max_per_frame=1, trials_per_tuple=100, flow_threshold=25.0, jumpcut_threshold=np.inf,
                     workers=0):
    """
    Spawns the specified number of workers running _extract_patches_worker().
    Call this with workers=0 to run on the current thread.
    """

    tick_t = timer()
    print('===> Extracting patches...')

    if workers != 0:
        parallel = Parallel(n_jobs=workers, backend='threading', verbose=5)
        tuples_per_job = len(tuples) // workers + 1
        result = parallel(
            delayed(_extract_patches_worker)(tuples[i:i + tuples_per_job], max_per_frame, trials_per_tuple,
                                             flow_threshold, jumpcut_threshold) for i in
            range(0, len(tuples), tuples_per_job))
        patches = sum(result, [])
    else:
        patches = _extract_patches_worker(tuples, max_per_frame, trials_per_tuple, flow_threshold, jumpcut_threshold)

    tock_t = timer()
    print("Done. Took ~{}s".format(round(tock_t - tick_t)))

    return patches

def _extract_custom_patches(tuples, flow_threshold, jumpcut_threshold, force_horizontal, workers):
    tick_t = timer()

    initFlowNet2()

    print("===> Extracting custom patches...")

    # if workers != 0:
    #     parallel = Parallel(n_jobs=workers, backend="threading", verbose=5)
    #     tuples_per_job = len(tuples) // workers + 1
    #     result = parallel(delayed(_extract_custom_patches_worker_flownet)(tuples[i:i+tuples_per_job], flow_threshold,
    #                             jumpcut_threshold, force_horizontal) for i in range(0, len(tuples), tuples_per_job))
    #     patches = sum(result, [])
    # else:

    # flownet processing is gpu limited anyway
    patches = _extract_custom_patches_worker_flownet(tuples, flow_threshold, jumpcut_threshold, force_horizontal)
    
    freeFlowNet2()

    tock_t = timer()
    print("Done. Took ~{}s".format(round(tock_t - tick_t)))

    return patches

############################################### CACHE ###############################################

def get_cached_patches(dataset_dir=None):
    """
    Finds the cached patches (stored as images) from disk and returns their paths as a list of tuples
    :param dataset_dir: Path to the dataset folder
    :return: List of paths to patches as tuples (path_to_left, path_to_middle, path_to_right)
    """

    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR

    cache_dir = join(dataset_dir, 'cache')

    frame_paths = [join(cache_dir, x) for x in listdir(cache_dir)]
    frame_paths = [x for x in frame_paths if is_image(x)]
    frame_paths.sort()

    tuples = []

    for i in range(len(frame_paths) // config.MAX_SEQUENCE_LENGTH):
        foo = (frame_paths[i * config.MAX_SEQUENCE_LENGTH + ix] for ix in range(config.MAX_SEQUENCE_LENGTH))
        tuples.append(list(foo))

    return tuples


def _cache_patches_worker(cache_dir, patches):
    """
    Writes to disk the specified patches as images.
    :param cache_dir: Path to the cache folder
    :param patches: List of patches
    """
    for p in patches:
        patch_id = str(random.randint(1e10, 1e16))
        frames = load_patch(p)
        for i in range(config.MAX_SEQUENCE_LENGTH):
            file_name = '{}_{}.jpg'.format(patch_id, i)
            frames[i].save(join(cache_dir, file_name), 'JPEG', quality=95)


def _cache_patches(cache_dir, patches, workers=0):
    """
    Spawns the specified number of workers running _cache_patches_worker().
    Call this with workers=0 to run on the current thread.
    """

    if exists(cache_dir):
        rmdir(cache_dir)

    makedirs(cache_dir)

    tick_t = timer()
    print('===> Caching patches...')

    if workers != 0:
        parallel = Parallel(n_jobs=workers, backend='threading', verbose=5)
        patches_per_job = len(patches) // workers + 1
        parallel(delayed(_cache_patches_worker)(cache_dir, patches[i:i + patches_per_job]) for i in
                 range(0, len(patches), patches_per_job))
    else:
        _cache_patches_worker(cache_dir, patches)

    tock_t = timer()
    print("Done. Took ~{}s".format(round(tock_t - tick_t)))


################################################ MAIN ###############################################

def prepare_dataset(dataset_dir=None, force_rebuild=False):
    """
    Performs all necessary operations to get the training dataset ready, such as
    selecting patches, caching the cropped versions if necessary, etc..
    :param dataset_dir: Path to the dataset folder
    :param force_rebuild: Whether or not the patches should be extracted again, even if a cached version exists on disk
    :return: List of patches
    """

    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR

    workers = config.NUM_WORKERS
    json_path = join(dataset_dir, 'patches.json')
    cache_dir = join(dataset_dir, 'cache')

    if exists(json_path) and not force_rebuild:

        print('===> Patches already processed, reading from JSON...')
        with open(json_path) as f:
            patches = json.load(f)

        if config.CACHE_PATCHES and not exists(cache_dir):
            _cache_patches(cache_dir, patches, workers)

        return patches

    if config.DATASET == "DAVIS":
        print("Working with DAVIS dataset")
        davis_dir = get_davis_17(dataset_dir)
        tuples = tuples_from_davis(davis_dir, res='480p')

        patches = _extract_patches(
            tuples,
            max_per_frame=20,
            trials_per_tuple=30,
            flow_threshold=config.FLOW_THRESHOLD,
            jumpcut_threshold=8e-3,
            workers=workers
        )

    else:
        ddir = join(config.DATASET_DIR, "frames")
        print(f"Working with dataset in {ddir}")
        tuples = tuples_from_custom(ddir)
        patches = _extract_custom_patches(
            tuples,
            flow_threshold=config.FLOW_THRESHOLD,
            jumpcut_threshold=8e-3,
            force_horizontal=config.FORCE_HORIZONTAL,
            workers=workers
        )


    # shuffle patches before writing to file
    random.shuffle(patches)

    print('===> Saving JSON...')
    with open(json_path, 'w') as f:
        json.dump(patches, f)

    if config.CACHE_PATCHES:
        _cache_patches(cache_dir, patches, workers)

    return patches
