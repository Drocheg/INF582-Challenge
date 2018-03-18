The code is divided in multiple files, mainly the “main.py” files, where the core of the algorithm is, and the “feature_engineering.py” file, where the features are computed.
Because the features took a lot of time to compute, this are saved in a file. To load the precomputed features a variable “quick_eval_mode” need to be on true. This is the default mode. 
Also, the default mode is to use cross validation. This is recommended because the scores are better and the prediction and testing is all done together.
In case the features need to be recomputed to test the algorithm, it is recommended to use a subset of the data.
