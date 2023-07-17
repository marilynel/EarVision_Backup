'''
EarVision 2.0:
EarVision Hyperparameter Search

This script performs a crude hyperparameter search on EarVision's Train module. The search is completed using the 
grid-search tuning technique (examples here: https://datagy.io/sklearn-gridsearchcv/ and
https://discuss.pytorch.org/t/what-is-the-best-way-to-perform-hyper-parameter-search-in-pytorch/19943).

In order to use this script, go to the "Adjust hyperparams here" comment. The lists here will contain the 
hyperparameter numerical values to be tested for trainable_backbone_layers, box_nms_thresh, and box_score_thresh. These
can be changed as needed. If different hyperparameters need to be tested, go to the hyperparameters dictionary below 
and change the value there for the list of values you would like to iterate through. "NOTE" will be in the comments 
above places where the code can be easily customized to the current needs of the user.

The models produced by this script will be in C:\CornEnthusiast\Projects\EarVision\SavedModels\HyperparamSearch

Important note: the models produced are very large, and some may need to be manually deleted in order for there to be 
enough room to run the search for all values. This search script will develop models for every possible combination of 
the hyperparameters given. 

Marilyn Leary 2023 
https://github.com/marilynel
'''

from Train import main as trainOneModel
import datetime


def hyperparameterSearch():
    # NOTE: Change subfolderName to specify details for your searcg
    subfolderName = "HyperparamSearch_trainingSetAllImages" 

    print("-----------------------------------")
    print("COMMENCING HYPERPARAMETER SEARCH...")
    print("-----------------------------------")

    searchStartTime = datetime.datetime.now()
    searchDir = "HyperparamSearch/" + subfolderName + searchStartTime.strftime("%m.%d.%y_%I.%M%p" + "/")

    # NOTE: Adjust hyperparams here
    trainable_backbone_layers_tuning = [0, 1, 2, 3, 4, 5]                                              
    #box_nms_thresh_tuning = [0.1, 0.3, 0.5, 0.7, 0.9]
    #box_score_thresh_tuning = [0.1, 0.2, 0.4, 0.6, 0.8]
    #trainable_backbone_layers_tuning = []
    box_nms_thresh_tuning = [0.5]
    box_score_thresh_tuning = [0.2]          

    # NOTE: "ok" is used to help move search along if it gets interrupted. To use, set ok = False, then change the if 
    # statement in the loop below to restart the search where it left off. 
    ok = True #False    
    for i in range(len(trainable_backbone_layers_tuning)):
        for j in range(len(box_nms_thresh_tuning)):
            for k in range(len(box_score_thresh_tuning)):
                # NOTE: The following "if" statement can be adjusted and uncommented if needed. See comment above.
                # search along if it gets interrupted.
                # if (i > 3 and j > 0 and k > 3):
                #    ok = True
                if ok:
                    hyperparams = {
                        "validationPercentage" : 0.2,           
                        "batchSize" : 16,                       
                        "learningRate" : 0.0005,                
                        "epochs" : 30,                          
                        "rpn_pre_nms_top_n_train" : 3000,       
                        "rpn_post_nms_top_n_train" : 3000,      
                        "rpn_pre_nms_top_n_test" : 3000,         
                        "rpn_post_nms_top_n_test" : 3000,       
                        "rpn_fg_iou_thresh" : 0.7,              
                        "rpn_batch_size_per_image" : 512,       
                        "min_size" : 800,                       
                        "max_size" : 1333,                      
                        "trainable_backbone_layers" : trainable_backbone_layers_tuning[i],
                        "box_nms_thresh" : box_nms_thresh_tuning[j],                
                        "box_score_thresh" : box_score_thresh_tuning[k]               
                    }

                    trainOneModel(hyperparameterInput = hyperparams, searchResultDir = searchDir)

if __name__ == "__main__":
    hyperparameterSearch()