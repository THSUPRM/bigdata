import json
import csv


def main():
    with open("FullTweets_1.json", "r") as tweet_data:

        # tweet_str = tweet_data.readline()
        # tweet = eval(tweet_str)
        # data = str(tweet['id']) + "," + tweet['full_text']
        # print("data: ", data)
        with open("unlabeledtweets.csv", "w") as unlabeled_tweets:
            for line in tweet_data:
                tweet_str = tweet_data.readline()
                tweet = eval(tweet_str)
                data = str(tweet['id']) + "|" + tweet['full_text'] + "\n"
                #print("data: ", data)
                unlabeled_tweets.write(data)
                unlabeled_tweets.flush()

if __name__ == "__main__":
    main()