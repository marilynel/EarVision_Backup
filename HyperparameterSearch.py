#runs a hyperparameter search
from Train import main as trainOneModel
import datetime

def hyperparameterSearch():
    print("-----------------------------------")
    print("COMMENCING HYPERPARAMETER SEARCH...")
    print("-----------------------------------")

    #okay, so should have way to say which hyperparameters to search through. Probably also want a seed for the validation split?


    searchStartTime = datetime.datetime.now()
    searchDir = "HyperparamSearch_" + searchStartTime.strftime("%m.%d.%y_%I.%M%p" + "/")


    hyperparams = {}

    trainOneModel(hyperparameterInput = hyperparams, searchResultDir = searchDir)

if __name__ == "__main__":
    hyperparameterSearch()