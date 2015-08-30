from sklearn import svm
from read_data import read_training_set,read_test_set

def get_results_svm():
	X_train, y = read_training_set("data/train.csv")
	X_test = read_test_set("data/test.csv")

	training_model = svm.SVR()
	training_model.fit(X_train,y)
	predicted = training_model.predict(X_test)
	return predicted

def write_results(results):
	with open("results.csv","w") as f:
		f.write("Activity_predicted"+"\n")
		f.writelines([str(x)+"\n" for x in results])

if __name__ == "__main__":
	predicted = get_results_svm()
	write_results(predicted)




