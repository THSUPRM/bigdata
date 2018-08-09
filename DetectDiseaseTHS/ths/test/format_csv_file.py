import csv


def main(input_name, output_name):
    with open(input_name, "r") as f:
        reader = csv.reader(f, delimiter=' ')
        with open(output_name, "w") as f2:
            writer = csv.writer(f2, delimiter='|')
            for t in reader:
                writer.writerow(t)
            f2.flush()
    print("New file generated in the route: ", output_name)


if __name__ == "__main__":
    # main("data/errorlstm1.csv", "data/fixed_errorlstm.csv")
    main("data/errorcnn.csv", "data/fixed_errorcnn.csv")
