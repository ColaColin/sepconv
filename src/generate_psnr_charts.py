import argparse
import os
import json

import matplotlib.pyplot as plt

import numpy as np

def openJson(path):
    with open(path) as f:
        parsed = json.load(f)
    return parsed

def createDiagram(name, psnrFile, targetFile):
    psnrDict = openJson(psnrFile)

    nums = []
    psnrs = []

    for k in psnrDict:
        imgNum = int(k) + 1
        imgPsnr = psnrDict[k]
        nums.append(imgNum)
        psnrs.append(imgPsnr)

    zero3 = 0
    zCnt = 0
    for k in range(len(nums)):
        if psnrs[k] == 0:
            zero3 = k + 1
            zCnt += 1
            if zCnt == 3:
                break

    nums = nums[:zero3]
    psnrs = psnrs[:zero3]

    maxPsnr = np.max(psnrs)
    minPsnr = maxPsnr
    for psnr in psnrs:
        if psnr < minPsnr and psnr != 0:
            minPsnr = psnr

    minY = int(minPsnr - 0.001)
    maxY = int(maxPsnr * 1.05)

    fig, ax = plt.subplots()

    bwidth = 0.65

    plt.ylim(minY, maxY)

    ax.set_axisbelow(True)

    ax.grid()

    ax.bar(nums, psnrs, bwidth, color='b', label='Predicted frames')

    zeroBarIdx = []
    for n in range(len(nums)):
        if psnrs[n] == 0:
            zeroBarIdx.append(nums[n])
    
    ax.bar(zeroBarIdx, np.ones(len(zeroBarIdx)) * maxY, bwidth, color='r', label='Input frames')

    ax.set_xlabel("Image")
    ax.set_ylabel("PSNR")
    ax.set_title("PSNR per image with config " + name)
    ax.set_xticks([n for n in nums if n % 3 == 0])
    ax.legend()
    

    fig.tight_layout()

    plt.savefig(targetFile, bbox_inches='tight', format='eps', dpi=300)

    plt.close(fig)    


def procDir(baseDir, dirName):
    files = os.listdir(dirName)

    for f in files:
        fPath = os.path.join(dirName, f)
        if os.path.isdir(fPath):
            procDir(baseDir, fPath)
        elif f == "psnr.json":
            print(f"Found a psnr.json in {dirName}")
            name = dirName.replace(baseDir, "")
            createDiagram(name, fPath, os.path.join(dirName, "psnr.eps"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PSNR chart generator")
    parser.add_argument("--dir", type=str, required=True, help='path to directory to recursively search for psnr.json files')

    params = parser.parse_args()

    procDir(params.dir, params.dir)