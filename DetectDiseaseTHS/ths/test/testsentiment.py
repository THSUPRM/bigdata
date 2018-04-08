from ths.nn.sequences.process import ProcessTweetsGlove, ProcessTweetsGloveOnePass

def main():
    print("Working:")
    #P  = ProcessTweetsGlove("data/cleantextlabels.csv","data/glove.6B.50d.txt")
    P  = ProcessTweetsGloveOnePass("data/cleantextlabels2.csv","data/glove.6B.50d.txt")

    P.process("trained/model14.json", "trained/model14.h5")
#joderme
if __name__ == "__main__":
    main()