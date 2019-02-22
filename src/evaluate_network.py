# input:
# trained network
# directory of directories with image sequences to evaluate on
# output directory

# output
# directory structure:
# <output dir>/t<range of ts>/<DatasetName>/<produced images-directory|psnr.json>

# the range of ts is 8,16,32

# run the script multiple times to produce a directory structure of:
# results/<networkname>/<script output>

# to then generate tables and charts, run generate_psnr_charts on the results directory

import argparse
import os

from src.model import Net

from src.main import run_parallax_view_generation0

def doEval(evalDatasets, outputDir, networkFile, netmode):
    print(f"Evaluating {networkFile}")
    tValues = [8,16,32]
    model = Net.from_file(networkFile, netmode)
    
    for t in tValues:
        tOutDir = os.path.join(outputDir, "t" + str(t))
        print(f"Working with t={t}")
        for datasetDir in os.listdir(evalDatasets):
            print(f"Working on dataset {datasetDir}")
            inputDir = os.path.join(evalDatasets, datasetDir)
            outDir = os.path.join(tOutDir, datasetDir)
            psnr = run_parallax_view_generation0(model, t, inputDir, outDir, netmode)
            print(f"Generated views t={t} on {inputDir} writing to {outDir}, worst psnr was {psnr} dB")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Network evaluation script")
    # parser.add_argument("--input", type=str, required=True, help='path to the directory with evaluation datasets')
    # parser.add_argument("--output", type=str, required=True, help='path to output directory')
    # parser.add_argument("--network", type=str, required=True, help='path to the network weights')
    # parser.add_argument("--netmode", type=str, required=True, help='2to1 or 3to2')

    # params = parser.parse_args()

    # doEval(params.input, params.output, params.network, params.netmode)

    evalRuns = [
        #{"input": "/ImbaKeks/git/sepconv/video/eval/", "output": "/ImbaKeks/git/sepconv/video/eval_result/lq_2to1_seq9", "network": "/ImbaKeks/git/sepconv/out_seq_len_9/model_epoch_25.pth", "netmode": "2to1"},
        #{"input": "/ImbaKeks/git/sepconv/video/eval/", "output": "/ImbaKeks/git/sepconv/video/eval_result/hq_2to1_seq9", "network": "/ImbaKeks/git/sepconv/out_seq_len_9_hq/model_epoch_15.pth", "netmode": "2to1"},

        #{"input": "/ImbaKeks/git/sepconv/video/eval/", "output": "/ImbaKeks/git/sepconv/video/eval_result/lq_2to1_seq3", "network": "/ImbaKeks/git/sepconv/out_run_seq_3/model_epoch_26.pth", "netmode": "2to1"},
        {"input": "/ImbaKeks/git/sepconv/video/eval/", "output": "/ImbaKeks/git/sepconv/video/eval_result/hq_2to1_seq3", "network": "/ImbaKeks/git/sepconv/out_seq_len_3_hq/model_epoch_16.pth", "netmode": "2to1"},

        #{"input": "/ImbaKeks/git/sepconv/video/eval/", "output": "/ImbaKeks/git/sepconv/video/eval_result/lq_3to2_seq5", "network": "/ImbaKeks/git/sepconv/out_run_3to2/model_epoch_21.pth", "netmode": "3to2"},
        #{"input": "/ImbaKeks/git/sepconv/video/eval/", "output": "/ImbaKeks/git/sepconv/video/eval_result/hq_3to2_seq5", "network": "/ImbaKeks/git/sepconv/out_3to2_hq/model_epoch_8.pth", "netmode": "3to2"},
    ]

    for params in evalRuns:
        doEval(params['input'], params['output'], params['network'], params['netmode'])
            



