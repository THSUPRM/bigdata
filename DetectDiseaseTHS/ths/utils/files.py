import numpy as np
import sys

class EmbeddingException(Exception):
    pass

class GloveEmbedding:
    def __init__(self, filename, dimensions = 50):
        if not filename :
            raise Exception("Illegal file name.")
        if dimensions < 1:
            raise Exception("Illegal value for dimensions")

        self.filename = filename
        self.dimensions = dimensions

    def get_dimensions(self):
        return self.dimensions

    def parse_embedding(self, data_vector):

        result = []
        for n in data_vector:
            result.append(float(n))
        return result

    def read_embedding_bad(self):
        try:
            data_in = open(self.filename, "r")
        except Exception as e:
            msg  = sys.exc_info()[0]
            raise EmbeddingException(msg) from e
        else:
            i = 0
            word_to_idx = {}
            idx_to_word = {}
            word_to_vect = []
            with data_in:
                for line in enumerate(data_in):
                    #print(line)
                    parts = line[1].split()
                    word_part = parts[0]
                    vector_parts = parts[1:]
                    idx_to_word[i] = word_part
                    word_to_idx[word_part] = i
                    i = i+ 1
                    word_to_vect.append(self.parse_embedding(vector_parts))
            #add <unk> token
            unk = np.random.rand(self.dimensions,)
            idx_to_word[i] = "<unk>"
            word_to_idx["<unk>"] = i
            word_to_vect.append(unk)
            np_word_to_vect = np.array(word_to_vect)
            return word_to_idx, idx_to_word, np_word_to_vect

    def read_embedding(self):
        try:
            data_in = open(self.filename, "r")
        except Exception as e:
            msg  = sys.exc_info()[0]
            raise EmbeddingException(msg) from e
        else:
            i = 1
            word_to_idx = {}
            word_to_idx['<EOF>'] = 0
            idx_to_word = {}
            idx_to_word[0] = None
            word_to_vect = []
            word_to_vect.append(np.zeros((self.dimensions,)))
            with data_in:
                for line in enumerate(data_in):
                    #print(line)
                    parts = line[1].split()
                    word_part = parts[0]
                    vector_parts = parts[1:]
                    idx_to_word[i] = word_part
                    word_to_idx[word_part] = i
                    i = i+ 1
                    word_to_vect.append(self.parse_embedding(vector_parts))
            #add <unk> token
            unk = np.random.rand(self.dimensions,)
            idx_to_word[i] = "<unk>"
            word_to_idx["<unk>"] = i
            word_to_vect.append(unk)
            np_word_to_vect = np.array(word_to_vect)
            return word_to_idx, idx_to_word, np_word_to_vect
