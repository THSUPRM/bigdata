from ths.nn.sequences.process import *
import sys

def main(optimizer):
    # P  = ProcessTweetsGlove("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    # P  = ProcessTweetsGloveOnePass("data/cleantextlabels2.csv","data/glove.6B.50d.txt")
    P = ProcessTweetsGloveOnePassHyperParam("data/cleantextlabels5.csv", "data/glove.6B.50d.txt", optimizer)
    # P = ProcessTweetsGloveOnePass("data/cleantextlabels3.csv", "data/glove.6B.50d.txt")

    P.process("trained/model14.json", "trained/model14.h5")

if __name__ == "__main__":
    main(sys.argv[1])
