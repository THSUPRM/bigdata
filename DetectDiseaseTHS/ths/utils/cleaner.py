import re
import csv
import string

from ths.utils.contractions import expandContractions

class TweetCleaner:

    def __init__(self, input_name, output_name):
        self.input_name = input_name
        self.output_name = output_name

    def clean(self):
        with open(self.input_name, "r") as f1:
            with open(self.output_name, "w") as f2:
                reader = csv.reader(f1, delimiter='|')
                writer = csv.writer(f2, delimiter=',')
                i = 0
                for row in reader:

                    tweet = row[0]
                    label = row[1]
                    # print("row: ", tweet)
                    # first make it lower case
                    tweet = tweet.lower()
                    # second remove http and https with web link
                    # tweet = re.sub(r'http\S+', 'link', tweet)
                    tweet = re.sub(r'http\S+', '', tweet)
                    # third remove #hashtag with hash tag
                    # tweet = re.sub(r'#\S+', 'hastag', tweet)
                    # tweet = re.sub(r'#\S+', '', tweet)
                    tweet = tweet.replace("#", "")
                    # fourth remove @user with twitter user
                    # tweet = re.sub(r'@\S+', 'mention', tweet)
                    tweet = re.sub(r'@\S+', '', tweet)
                    # fifth remove contractions
                    tweet = expandContractions(tweet)
                    # sixth remove punction marks
                    translator = str.maketrans('', '', string.punctuation)
                    tweet = tweet.translate(translator)

                    tweet = tweet.strip()
                    out = []
                    out.append(tweet)
                    out.append(label)
                    #print(i)
                    i = i  + 1
                    writer.writerow(out)

