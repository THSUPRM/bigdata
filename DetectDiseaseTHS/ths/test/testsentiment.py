from ths.nn.sequences.ml_process.best_models_multiclass import BestModelsMulticlass
from ths.nn.sequences.process import *
import sys


def main(optimizer):
    # P = ProcessTweetsGloveOnePassHyperParamPartionedData("data/cleantextlabels5.csv", "data/glove.6B.50d.txt", optimizer)
    # P = ProcessTweetsGloveOnePassBestModels("data/cleantextlabels5.csv", "data/glove.6B.50d.txt", optimizer)
    # P = ProcessTweetsGloveOnePassBestModelsMulticlass("data/cleantextlabels4.csv", "data/glove.6B.50d.txt", optimizer)
    # P.process("trained/model14.json", "trained/model14.h5")
    P = BestModelsMulticlass("data/cleantextlabels4.csv", "data/glove.6B.50d.txt", optimizer)
    P.process()


if __name__ == "__main__":
    main(sys.argv[1])
