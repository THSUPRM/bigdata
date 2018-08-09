import csv
import numpy as np


def main(input_name, model):
    tweets_cnn, label_cnn, pred_cnn = load_data(input_name)
    print("LENS: ", len(tweets_cnn), len(label_cnn), len(pred_cnn))
    get_tweet_lengths(tweets_cnn)
    # word_dict = get_first_col_glove_save("data/glove.6B.50d.txt", "data/first_column_glove_" + model + ".csv")
    word_dict = get_first_col("data/glove.6B.50d.txt")

    # for i in range(0, len(tweets_cnn)):
    all_words_no_present = set()
    total_words_no_present = 0
    text = ""
    all_text = ""
    # for i in range(10):
    for t in tweets_cnn:
        words_present = ""
        count = 0
        for w in t.split():
            # if w in word_dict:
            #     # print("palabra: " + w + " SI")
            #     words_present += "SI "
            # else:
            #     # print("palabra: " + w + " NO")
            #     words_present += w + " NO "
            if w not in word_dict:
                count += 1
                words_present += w + " "
                all_words_no_present.add(w)
                total_words_no_present += 1
        # print(t)
        text += ("\n" + t)
        if words_present != "":
            # print("NO DIC: " + words_present + " TOTAL WORDS: " + str(count))
            text += ("\n" + "NO DIC: " + words_present + " TOTAL WORDS: " + str(count))
    # print(str(all_words_no_present))
    all_text += str(all_words_no_present)
    # print("TOTAL words no present: " + str(total_words_no_present) + "---Len() set words no present: " +
    #       str(len(all_words_no_present)))
    all_text += ("\nTOTAL words no present: " + str(total_words_no_present) + "---Len() set words no present: " +
                 str(len(all_words_no_present)))
    # save_file(text.strip(), "data/tweets_with_count_no_ocurrences_" + model + ".txt")
    # save_file(all_text, "data/total_counts_no_ocurrences_" + model + ".txt")


def save_file(text, filename):
    with open(filename, "w") as f:
        f.write(text)


def get_first_col(input_filename):
    word_dict = []
    with open(input_filename, "r") as f:
        for line in f:
            word_dict.append(line.split()[0])
    return word_dict


def get_first_col_glove_save(input_filename, output_filename):
    word_dict = []
    with open(input_filename, "r") as f:
        with open(output_filename, "w") as f2:
            for line in f:
                word_dict.append(line.split()[0])
                f2.write(line.split()[0].strip().replace("\n", "").replace(" ", "") + "\n")
            f2.flush()
        word_dict = set(word_dict)
    return word_dict


def get_tweet_lengths(tweets):
    # print([len(a.split()) for a in tweets])
    lengths = [len(a.split()) for a in tweets]
    # lengths = sorted(set(sorted(lengths, reverse=True)), reverse=True)
    print("TOTAL lengths: ", lengths)
    # print("First 10 max lens: ", lengths[: 10])
    print("AVG(lengths): ", np.mean(lengths))


def load_data(input_name):
    tweets = []
    label = []
    pred = []
    with open(input_name, "r") as f:
        reader = csv.reader(f, delimiter="|")
        for r in reader:
            # print("Tweet:" + r[0] + " Label:" + r[1] + " Predicted:" + r[2])
            tweets.append(str(r[0]).strip())
            label.append(int(r[1]))
            pred.append(int(r[2]))
    # print("LENS: ", len(tweets), len(label), len(pred))
    return tweets, label, pred


if __name__ == "__main__":
    main("data/fixed_errorcnn.csv", "cnn")
    main("data/fixed_errorlstm.csv", "lstm")
