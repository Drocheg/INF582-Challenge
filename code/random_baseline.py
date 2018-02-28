import numpy as np
import csv
from read_data import *

###################
# random baseline #
###################

path_to_predictions = "../predictions"

testing_set, training_set, node_info = read_data()

random_predictions = np.random.choice([0, 1], size=len(testing_set))
random_predictions = zip(range(len(testing_set)), random_predictions)

with open(path_to_predictions + "random_predictions.csv", "wb") as pred:
    csv_out = csv.writer(pred)
    for row in random_predictions:
        csv_out.writerow(row)

# note: Kaggle requires that you add "ID" and "category" column headers
