from ths.nn.sequences.ml_process.evaluate_models_multiclass_cnn import EvaluateModelsMulticlassCNN
from ths.nn.sequences.ml_process.best_models_multiclass_cnn import EvaluateBestModelsMulticlassCNN

def main():
    P = EvaluateModelsMulticlassCNN("data/cleantextlabels7.csv", "data/glove.6B.50d.txt", 'NONE', "models/CNN")
    P.process()

    # P = EvaluateBestModelsMulticlassCNN("data/cleantextlabels7.csv", "data/glove.6B.50d.txt", 'NONE', "models/CNN")
    # P.process()


if __name__ == "__main__":
    main()
