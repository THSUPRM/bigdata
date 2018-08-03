from ths.nn.sequences.process import *
import sys


def main(optimizer):
    # P = ProcessTweetsGloveOnePassHyperParamPartionedData("data/cleantextlabels5.csv", "data/glove.6B.50d.txt", optimizer)
    P = ProcessTweetsGloveOnePassBestModels("data/cleantextlabels5.csv", "data/glove.6B.50d.txt", optimizer)

    P.process("trained/model14.json", "trained/model14.h5")


if __name__ == "__main__":
    main(sys.argv[1])
