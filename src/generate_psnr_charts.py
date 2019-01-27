import argparse
import os

import matplotlib.pyplot as plt

import numpy as np

from src.utilities import openJson

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

    minY = int(minPsnr - 0.005)
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

def findTForPsnrs(psnrDict):
    for i in range(1, len(psnrDict)):
        if psnrDict[str(i)] == 0:
            return i

def createAvgDiagram(name, psnrFiles, targetFile):
    pDicts = [openJson(x) for x in psnrFiles]

    ts = [findTForPsnrs(d) for d in pDicts]

    assert np.sum(ts) == ts[0] * len(ts), "All psnr files in the average operation should have the same t?"

    imgNum = len(pDicts[0])
    t = ts[0]

    avgs = np.zeros(t-1, dtype=np.float32)

    for pDict in pDicts:
        for i in range(imgNum):
            if i % t != 0:
                avgs[i % t - 1] += pDict[str(i)]

    intervalsNum = imgNum // t
    avgs /= len(pDicts) * intervalsNum

    maxValue = np.max(avgs)
    minValue = np.min(avgs)

    minY = int(minValue - 0.005)
    maxY = int(maxValue * 1.05)
    
    fig, ax = plt.subplots()

    bwidth = 0.65
    
    plt.ylim(minY, maxY)

    ax.set_axisbelow(True)

    ax.grid()

    ax.bar(np.arange(1, t), avgs, bwidth, color='b', label='Average values')

    ax.set_xlabel("Index between input frames")
    ax.set_ylabel("PSNR")
    ax.set_title("PSNR on average with config " + name)

    fig.tight_layout()

    plt.savefig(targetFile, bbox_inches='tight', format='eps', dpi=300)

    plt.close(fig)

def procDir(baseDir, dirName, currentDepth):
    files = os.listdir(dirName)

    psnrs = []

    name = os.path.basename(baseDir)

    for f in files:
        fPath = os.path.join(dirName, f)
        if os.path.isdir(fPath):
            for p in procDir(baseDir, fPath, currentDepth+1):
                psnrs.append(p)
        elif f == "psnr.json":
            print(f"Found a psnr.json in {dirName}")
            createDiagram(name, fPath, os.path.join(dirName, "psnr.eps"))
            psnrs.append(fPath)

    if 1 == currentDepth:
        createAvgDiagram(name, psnrs, os.path.join(dirName, "avg.eps"))
    elif 1 < currentDepth:
        return psnrs
    
    return []

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PSNR chart generator")
    parser.add_argument("--dir", type=str, required=True, help='path to directory to recursively search for psnr.json files')

    params = parser.parse_args()

    for f in os.listdir(params.dir):
        baseDir = os.path.join(params.dir, f)
        procDir(baseDir, baseDir, 0)

    