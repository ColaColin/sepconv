import torch
import numpy as np
import argparse

from .models import FlowNet2CS
from .utils.frame_utils import read_gen
from PIL import Image

import time

import cv2

def writeFlowImage(flowArray, filePath):
    flow = flowArray

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros(flow.shape[:2] + (3,), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

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

if __name__ == '__main__':
    #obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    #initial a Net
    net = FlowNet2CS(args).cuda()
    #load the state_dict
    dict = torch.load("/ImbaKeks/git/sepconv/video/FlowNet2-CS_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])

    pim1 = read_gen("/ImbaKeks/git/sepconv/video/raw/high_quality_datasets/frames/6qGiXY1SB68/00007.jpg")
    pim2 = read_gen("/ImbaKeks/git/sepconv/video/raw/high_quality_datasets/frames/6qGiXY1SB68/00008.jpg")

    #load the image pair, you can find this operation in dataset.py
    # pim1 = read_gen("/ImbaKeks/flownet2-pytorch/0000007-img0.ppm")
    # pim2 = read_gen("/ImbaKeks/flownet2-pytorch/0000007-img1.ppm")
    images = [pim1, pim2]
    images = np.array(images)
    print(images.shape)
    images = images.transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    sTime = time.time()

    print("Input shape", im.shape)
    
    with torch.no_grad():
        # pad inputs to image size that is a multiple of 64 for the network
        inPad, outPad = getPadFlowNet2InputBatch(im.shape[4], im.shape[3])
        inPad = inPad.cuda()
        outPad = outPad.cuda()
        bDims = im.shape
        im = inPad(im.reshape(bDims[0], bDims[1] * bDims[2], bDims[3], bDims[4]))
        im = im.reshape(bDims[0], bDims[1], bDims[2], im.shape[2], im.shape[3])

        #process the image pair to obtain the flow
        netOut = net(im)

        #remove any padding applied before the input
        result = outPad(netOut)

    print("Output shape", result.shape)

    # remove the batchdimension, since this is a single image pair example
    result = result.squeeze()

    data = result.data.cpu().numpy().transpose(1, 2, 0)

    print(f"Took {(time.time() - sTime):.2f}s")

    writeFlowImage(data, "/ImbaKeks/flownet2-pytorch/0000007-img.jpg")