from ths.nn.sequences.ml_process.model_loader import ModelLoader


def main():
    route = "models/GRU/"
    combination = "(0.003,0,5,32,50,0,0,0.1,50,0,0,0.1,64,0,3,'RMSPROP')"

    ModelLoader.load_evaluate_model(route + "model" + combination + ".json", route + "model" + combination + ".h5",
                                    "RMSPROP", 0.003, route + "x_test.txt", route + "y_test.txt")

    # ModelLoader.load_evaluate_model(route + "model" + combination + ".json", route + "model" + combination + ".h5",
    #                                 "RMSPROP", 0.003, route + "x_validation.txt", route + "y_validation.txt")


if __name__ == "__main__":
    main()
