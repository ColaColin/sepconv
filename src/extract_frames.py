#
# KTH Royal Institute of Technology
#

import imageio
import argparse
import os
from os.path import join
from timeit import default_timer as timer
from PIL import Image


def extract_frames(video_path, handler):

    def convert_frame(arg):
        return Image.fromarray(arg[:, :, :3], mode='RGB')

    video_reader = imageio.get_reader(video_path)
    fps = video_reader.get_meta_data().get('fps', None)

    idx = 0
    for x in video_reader:
        handler(idx, convert_frame(x))
        idx += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video Frame Extraction')
    parser.add_argument('--src', type=str, required=True, help='path to the video')
    parser.add_argument('--dest', type=str, required=True, help='path to the output directory')
    params = parser.parse_args()

    tick_t = timer()

    if not os.path.exists(params.dest):
        os.makedirs(params.dest)

    def handleFrame(i, frame):
        file_name = '{:05d}.jpg'.format(i)
        file_path = join(params.dest, file_name)
        frame.save(file_path, 'JPEG', quality=90)
        print("Stored frame %s" % file_path)

    extract_frames(params.src, handleFrame)

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))
