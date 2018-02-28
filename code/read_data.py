import csv
def read_data():
    path_to_data = "../data/"
    with open(path_to_data + "testing_set.txt", "r") as f:
        reader = csv.reader(f)
        testing_set = list(reader)
    testing_set = [element[0].split(" ") for element in testing_set]
    with open(path_to_data + "training_set.txt", "r") as f:
        reader = csv.reader(f)
        training_set = list(reader)
    training_set = [element[0].split(" ") for element in training_set]
    with open(path_to_data + "node_information.csv", "r") as f:
        reader = csv.reader(f)
        node_info = list(reader)
