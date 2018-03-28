from ths.utils.cleaner import TweetCleaner
import csv

def main():
    input_name = "textlabels.csv"
    output_name = "cleantextlabels.csv"
    C = TweetCleaner(input_name=input_name, output_name=output_name)
    C.clean()
    print("Done")

    with open("cleantextlabels.csv", "r", encoding="ISO-8859-1") as f:
        reader = csv.reader(f, delimiter = ',')
        max_len = 0
        max_sent = None
        for row in reader:
            sentence = row[0]
            if len(sentence) > max_len:
                max_len = len(sentence)
                max_sent = sentence

        print("max_len: ", max_len)
        print("max_sent: ", max_sent)

if __name__ == "__main__":
    main()