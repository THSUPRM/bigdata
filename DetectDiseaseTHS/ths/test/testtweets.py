from ths.utils.files import GloveEmbedding
from ths.nn.sequences.tweets import TweetSentiment2LSTM

def main():
    G = GloveEmbedding("glove.6B.50d.txt")

    T = TweetSentiment2LSTM(10, G)

    pass

if __name__ == "__main__":
    main()