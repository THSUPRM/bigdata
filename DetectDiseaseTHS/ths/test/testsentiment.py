from ths.nn.sequences.process import *
from ths.nn.sequences.process import ProcessTweetsGloveOnePassHyperParam


def main():
    # P  = ProcessTweetsGlove("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    # P  = ProcessTweetsGloveOnePass("data/cleantextlabels2.csv","data/glove.6B.50d.txt")
    P = ProcessTweetsGloveOnePassHyperParam("data/cleantextlabels4.csv", "data/glove.6B.50d.txt")
    # P = ProcessTweetsGloveOnePass("data/cleantextlabels3.csv", "data/glove.6B.50d.txt")

    P.process("trained/model14.json", "trained/model14.h5")

if __name__ == "__main__":
    main()
