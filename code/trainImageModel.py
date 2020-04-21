import numpy as np
import glob
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

if __name__ == '__main__':
	# Argument parser
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True, help="path of input data")
	ap.add_argument("-o", "--output", required=True, help="name of output trained model")
	ap.add_argument("-trS", "--train_size", help="train size (from 0 to 1). Default = 0.75", default=0.75)
	ap.add_argument("-teS", "--test_size", help="test size (from 0 to 1). Default = 0.25", default=0.25)
	ap.add_argument("-nc", "--num_core", help="number of core to use", default=1)
	args = vars(ap.parse_args())

	files = glob.glob(args["input"]+"*")
	numData = len(files)
	print("[INFO] - Tot data: ", numData)
	dataset = [] # initialize dataset
	for file in files:
		imgFlatten = np.fromfile(file, dtype = np.uint8) # read flatten image
		if file.find('player1') != -1: # check if data is of player1
			dataset.append([imgFlatten.copy(), 1]) # add data to dataset
		elif file.find('player2') != -1: # check if data is of player2
			dataset.append([imgFlatten.copy(), 2]) # add data to dataset
		elif file.find('referee') != -1: # check if data is of referee
			dataset.append([imgFlatten.copy(), 3]) # add data to dataset
		else:
			print("[ERROR]")
			break

	# split data into source and target
	datasetSource = [dataset[i][0] for i in range(numData)]
	datasetTarget = [dataset[i][1] for i in range(numData)]

	# split data into test and train
	X_train, X_test, y_train, y_test = train_test_split(datasetSource,datasetTarget, train_size = args["train_size"], test_size = args["test_size"])

	# instantiate classifier
	model = RandomForestClassifier(n_jobs=args["num_core"])
	model.fit(X_train, y_train) # train the model

	p_train = model.predict(X_train) # predict train data
	p_test = model.predict(X_test) # predict test data

	acc_train = accuracy_score(y_train, p_train) # check accuracy
	acc_test = accuracy_score(y_test, p_test) # check accuracy

	modelName = args["ouput"]
	pickle.dump(model, open(modelName, 'wb')) # save model

	print(f'Train {acc_train}, test {acc_test}')