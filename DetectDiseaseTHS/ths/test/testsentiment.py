from ths.nn.sequences.ml_process.best_models_multiclass import BestModelsMulticlass
from ths.nn.sequences.ml_process.sequential_model_best_GRU import SequentialModelBestGRU
from ths.nn.sequences.ml_process.best_models_multiclass import BestModelsMulticlass
from ths.nn.sequences.process import *
import sys


def main(optimizer):
    # P = ProcessTweetsGloveOnePassHyperParamPartionedData("data/cleantextlabels5.csv", "data/glove.6B.50d.txt", optimizer)
    # P = ProcessTweetsGloveOnePassBestModels("data/cleantextlabels5.csv", "data/glove.6B.50d.txt", optimizer)
    # P = ProcessTweetsGloveOnePassBestModelsMulticlass("data/cleantextlabels6.csv", "data/glove.6B.50d.txt", optimizer)
    # P.process("trained/model14.json", "trained/model14.h5")
    # P = ProcessTweetsGloveOnePassSequential("data/cleantextlabels6.csv", "data/glove.6B.50d.txt")
    # P.process("trained/model14.json", "trained/model14.h5")

    ################ Optimized ################
    P = BestModelsMulticlass("data/cleantextlabels7.csv", "data/glove.6B.50d.txt", optimizer, "models/RNN")
    P.process()

    # P = SequentialModelBestGRU("data/cleantextlabels7.csv", "data/glove.6B.50d.txt", optimizer, "models/GRU")
    # P.process()


if __name__ == "__main__":
    main(sys.argv[1])
