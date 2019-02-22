import argparse
import os

import matplotlib.pyplot as plt

import numpy as np

from src.utilities import openJson, writeTxt

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

def createPsnrTableContent(resultDict):
    """
    resultDict is a dict of keys that name network configurations
    to dicts that describe scores of keys that name datasets
    """

    avgKey = "AVG"

    # create a list of dataset keys and add the avgerage key to it.
    for k in resultDict:
        dKeys = list(resultDict[k].keys())
        break
    dKeys.sort()
    dKeys.append(avgKey)

    # calculate the average scores for the methods over the datasets
    for netC in resultDict:
        kAvg = 0.0
        kCnt = 0.0
        for d in resultDict[netC]:
            kAvg += resultDict[netC][d]
            kCnt += 1.0
        resultDict[netC][avgKey] = kAvg / kCnt

    # create a dict that encodes which network is the best one per dataset
    bestNetsPerDataSet = dict() # dataset key -> network key

    for dKey in dKeys:
        bestNetScore = 0
        bestNetKey = "none"

        for netC in resultDict:
            if resultDict[netC][dKey] > bestNetScore:
                bestNetScore = resultDict[netC][dKey]
                bestNetKey = netC

        bestNetsPerDataSet[dKey] = bestNetKey        

    # start producing the output string

    # one network per column, one dataset per row
    result = "\\begin{tabular}{l|"
    for nix, netCfg in enumerate(resultDict):
        result += "l"
        if nix != len(resultDict)-1:
            result += " "
    result += "}\n"

    result += "dataset & "

    for nix, netCfg in enumerate(resultDict):
        result += netCfg.replace("_", "\\_")
        if nix + 1 == len(resultDict):
            result += " \\\\\n"
        else:
            result += " & "
    result += "\\hline\n"

    for dKey in dKeys:
        result += dKey.replace("_", "\\_") + " & "
        
        for nix, netCfg in enumerate(resultDict):
            fstr = "${:0.3f}$".format(resultDict[netCfg][dKey]) 
            if bestNetsPerDataSet[dKey] == netCfg:
                result += "\\boldmath{" + fstr + "}"
            else:
                result += fstr

            if nix + 1 == len(resultDict):
                result += " \\\\\n"
            else:
                result += " & "

    result += "\\end{tabular}\n"

    return result


    # one network per column, one dataset per row
    # result = "\\begin{tabular}{l|"
    # for x in dKeys:
    #     result += "l"
    #     if x != dKeys[-1]:
    #         result += " "
    # result += "}\n"

    # result += "configuration & "

    # # begin work on the header line
    # for idx, k in enumerate(dKeys):
    #     result += k.replace("_", "\\_")
    #     if idx + 1 == len(dKeys):
    #         result += " \\\\\n"
    #     else:
    #         result += " & "

    # result += "\\hline\n"

    # # produce lines per network config
    # for netCfg in resultDict:
    #     netName = netCfg.replace("_", "\\_")
    #     result += netName + " & "

    #     for idx, dKey in enumerate(dKeys):
    #         fstr = "${:0.3f}$".format(resultDict[netCfg][dKey]) 
    #         if bestNetsPerDataSet[dKey] == netCfg:
    #             result += "\\boldmath{" + fstr + "}"
    #         else:
    #             result += fstr

    #         if idx + 1 == len(dKeys):
    #             result += " \\\\\n"
    #         else:
    #             result += " & "

    # result += "\\end{tabular}\n"

    # return result


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

def collectTableData(dirPath, aggFunc):
    netConfigs = []
    for d in os.listdir(dirPath):
        if os.path.isdir(os.path.join(dirPath, d)):
            netConfigs.append(d)
        
    tCfgs = []
    for d in os.listdir(os.path.join(dirPath, netConfigs[0])):
        if os.path.isdir(os.path.join(dirPath, netConfigs[0], d)):
            tCfgs.append(d)
        
    dSets = []
    for d in os.listdir(os.path.join(dirPath, netConfigs[0], tCfgs[0])):
        if os.path.isdir(os.path.join(dirPath, netConfigs[0], tCfgs[0], d)):
            dSets.append(d)

    result = dict()

    for tCfg in tCfgs:
        result[tCfg] = dict()

        for netC in netConfigs:
            result[tCfg][netC] = dict()

            for dSet in dSets:
                psnrFile = os.path.join(dirPath, netC, tCfg, dSet, "psnr.json")
                result[tCfg][netC][dSet] = aggFunc(openJson(psnrFile))

    return result

def minAggFunc(psnrDict):
    values = np.array(list(psnrDict.values()), dtype=np.float32)
    return np.min(values[values != 0])

def avgAggFunc(psnrDict):
    values = np.array(list(psnrDict.values()), dtype=np.float32)
    return np.mean(values[values != 0])
    
def writeTableTxt(tData, fPath, aggName):
    for tKey in tData:
        tTxt = createPsnrTableContent(tData[tKey])
        wPath = os.path.join(fPath, aggName + tKey + ".txt")
        writeTxt(wPath, tTxt)
        print("Wrote table file " + wPath)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PSNR chart generator")
    parser.add_argument("--dir", type=str, required=True, help='path to directory to recursively search for psnr.json files')

    params = parser.parse_args()

    # generate charts
    for f in os.listdir(params.dir):
        if os.path.isdir(os.path.join(params.dir, f)):
            baseDir = os.path.join(params.dir, f)
            procDir(baseDir, baseDir, 0)

    # generate latex tables
    tDataMin = collectTableData(params.dir, minAggFunc)
    tDataAvg = collectTableData(params.dir, avgAggFunc)
    writeTableTxt(tDataMin, params.dir, "min_")
    writeTableTxt(tDataAvg, params.dir, "avg_")
    
    
