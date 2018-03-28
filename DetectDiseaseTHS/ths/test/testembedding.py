from ths.utils.files import GloveEmbedding

def main():
    G = GloveEmbedding("glove.6B.50d.txt")
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    print("embedding shape: ", embedding.shape)
    print("idx hello: ", word_to_idx["hello"])
    print("word 20: ", idx_to_word[20])
    e = embedding[word_to_idx["hello"]]
    print("embedding hello: ", e)
    print("e.shape: ", e.shape)
    print("<UNK>: ", word_to_idx['<unk>'])
    print("embedding: <UNK>: ", embedding[word_to_idx['<unk>']])
if __name__ == "__main__":
    main()