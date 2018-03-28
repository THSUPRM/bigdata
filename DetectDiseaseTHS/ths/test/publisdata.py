import csv

def main():
    with open("labeledtweetdata.csv", "r", encoding="ISO-8859-1") as f:
        with open("labeledtweets.csv", "w") as f2:
            reader = csv.reader(f, delimiter = '|')
            writer = csv.writer(f2, delimiter = '|')
            for row in reader:
                out = []
                out.append(row[0])
                out.append(row[2])
                writer.writerow(out)


if __name__ == "__main__":
    main()