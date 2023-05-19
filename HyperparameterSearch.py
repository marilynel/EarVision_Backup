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

    trainable_backbone_layers_tuning = [0, 1, 2, 3, 4, 5]
    box_nms_thresh_tuning = [0.1, 0.3, 0.5, 0.7, 0.9]
    box_score_thresh_tuning = [0.1, 0.2, 0.4, 0.6, 0.8]


    # https://datagy.io/sklearn-gridsearchcv/


    hyperparams = {
        "validationPercentage" : 0.2,           # n/a
        "batchSize" : 16,                       # n/a
        "learningRate" : 0.0005,                # n/a
        "epochs" : 30,                          # n/a
        "rpn_pre_nms_top_n_train" : 3000,       # n/a
        "rpn_post_nms_top_n_train" : 3000,      # n/a
        "rpn_pre_nms_top_n_test" : 3000,        # n/a 
        "rpn_post_nms_top_n_test" : 3000,       # n/a
        "rpn_fg_iou_thresh" : 0.7,              # minimum IoU between the anchor and the GT box so that they can be considered as positive during training of the RPN.
        "rpn_batch_size_per_image" : 512,       # n/a
        "min_size" : 800,                       # n/a
        "max_size" : 1333,                      # n/a
        "trainable_backbone_layers" : trainable_backbone_layers_tuning,        # number of trainable (not frozen) layers starting from final block, values between 0 - 5, with 6 ==  all backbone layers are trainable. default == 3
        "box_nms_thresh" : box_nms_thresh_tuning,                 # NMS threshold for the prediction head. Used during inference
        "box_score_thresh" : box_score_thresh_tuning                # n/a
    }

    trainOneModel(hyperparameterInput = hyperparams, searchResultDir = searchDir)

if __name__ == "__main__":
    hyperparameterSearch()