import json

def main():
    i = 0
    with open("FullTweets_1.json") as json_data:
        next_line = json_data.readline()
        #print(next_line)
        tweet = json.loads(json.dumps(next_line))

        print("tweet: ", tweet, " ", type(tweet))
        T = eval(tweet)
        print(type(T))
        print(T["full_text"])
        if i == 10:
            return


if __name__ == "__main__":
    main()