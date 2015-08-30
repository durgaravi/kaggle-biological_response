from pandas import read_csv

def read_training_set(filename):

	training_data = read_csv(filename,header=0)
	y = training_data["Activity"]
	X = training_data[training_data.columns[1:]]
	
	return X,y

def read_test_set(filename):

	test_data = read_csv(filename,header=0)
	return test_data
