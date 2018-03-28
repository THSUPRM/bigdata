from ths.utils.contractions import expandContractions

def main():
    text = "Donald's trump isn't crazy, he's just a mother fucker."
    text2 = expandContractions(text)
    print("text: ", text)
    print("text2: ", text2)
if __name__ == "__main__":
    main()