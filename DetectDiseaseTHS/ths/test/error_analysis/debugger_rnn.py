import csv


def main():
    X_all = []
    Y_all = []
    with open("../data/cleantextlabels4.csv", "r") as f:
        i = 0
        csv_file = csv.reader(f, delimiter=',')
        for r in csv_file:
            if i != 0:
                tweet = r[0]
                label = r[1]
                print("Tweet[" + str(i) + "]: " + str(tweet).strip())
                print("Label: ", str(label).strip())
                X_all.append(str(tweet).strip())
                Y_all.append(int(label))
            i = i + 1

    print("Data Ingested")
    print("LEN: ", len(X_all))
    print("X_all[0]: ", X_all[0])

if __name__ == "__main__":
    main()