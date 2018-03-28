import numpy as np

class UnkWords:
    def __init__(self, word_to_idx):
        pass

class SentenceToIndices:

    def __init__(self, word_to_idx):
        self.word_to_idx = word_to_idx

    def map_sentence(self, sentence):
        result = []
        words = sentence.split()
        for w in words:
            if w in self.word_to_idx:
                result.append(self.word_to_idx[w])
            else:
                result.append(self.word_to_idx["<unk>"])
        return result

    def map_sentence_list(self, sentence_list):
        result = []
        max_len = 0
        for s in sentence_list:
            mapped = self.map_sentence(s)
            mapped_len = len(mapped)
            if mapped_len > max_len:
                max_len = mapped_len
            result.append(mapped)
        return result, max_len

class PadSentences:
    def __init__(self, max_len):
        self.max_len = max_len

    def pad(self, sentence):
        padding_len = self.max_len - len(sentence)
        padding = []
        if (padding_len > 0 ):
            r = range(0, padding_len)
            for _  in r:
                padding.append(0)
        return sentence + padding

    def pad_list(self, sentence_list):
        result = []
        for s in sentence_list:
            result.append(self.pad(s))
        return result


class SentenceToEmbedding:
    def __init__(self, word_to_idx, idx_to_word, word_to_vect):
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_to_vect = word_to_vect

    def map_sentence(self, sentence, max_len = 0):
        S = SentenceToIndices(self.word_to_idx)
        matrix = None
        mapped_sentence = S.map_sentence(sentence)
        for i in mapped_sentence:
            e = self.word_to_vect[i]
            if matrix is None:
                matrix = np.array(e)
            else:
                matrix = np.vstack([matrix, e])
        if max_len > 0:
            padding_len = max_len - len(mapped_sentence)
            #print("max_len: ", max_len)
            #print("len(mapped_sentence): ", len(mapped_sentence))
            #print("padding: ", padding_len)
            if padding_len > 0:
                shape = matrix[0].shape
                zero_vector = np.zeros(shape)
                for _ in range(0, padding_len):
                    matrix = np.vstack([matrix, zero_vector])
        return matrix



