#runs a hyperparameter search
from Train import main as trainOneModel
import datetime


'''
Hyperparams to consider searching for:

-   trainable_backbone_layers
    ints between 0 and 5 
-   box_nms_thresh
    floats between 0 and 1
    pref usually ~0.5
-   box_score_thresh
    likely floats between 0 and 1
-   rpn_fg_iou_thresh
    floats 0 to 1
-   validation percentage?

Need:
-   starting hyperparam values
-   changing amount -> 0.1 for floats?
                    -> up or down...start at one end and work toward other?
-   struct/file to store ending hyperparam values and helpful metrics
    f1 once thats figured out 
    map?
    trans diffs     -> in/out diff          -> in is in training set, out is in validation set
                    -> in/out abs diff



'''


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