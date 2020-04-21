import cv2
import argparse
import numpy as np
import os
import glob
from matplotlib import pyplot as plt

if __name__ == "__main__":
	# Argument parser
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True, help="path of input data")
	ap.add_argument("-o", "--output", required=True, help="path to output data")
	ap.add_argument("-t", "--type", required=True, help="type of player (eg: player1, player2, referee, ecc.)")
	ap.add_argument("-an", "--add_noise", type=bool, default=False, help="Bool option if add noise. If true dataset will be 10 times bigger with noise addiction. Default is false")
	args = vars(ap.parse_args())

	files = glob.glob(args["input"]+"*.png") # read all '.png' images inside input folder
	print("[INFO] - Found: ", len(files), " images") # print number of files found
	for file in files:
		histName = args["output"] + args["type"] + "_" + file[file.rfind('/')+1:len(file)-4] # create histogram file name
		histFile = open(histName, 'wb') # open histogram file
		print("[INFO] - Opening: ", file)
		img = cv2.imread(file) # read image

		# initalize histograms
		histBlue = np.array((256,),dtype=np.float)
		histGreen = np.array((256,),dtype=np.float)
		histRed = np.array((256,),dtype=np.float)

		histBlue_norm = np.array((256,),dtype=np.float)
		histGreen_norm = np.array((256,),dtype=np.float)
		histRed_norm = np.array((256,),dtype=np.float)

		# calculate histogram of image
		histBlue = ((cv2.calcHist([img],[0],None, [256], [1,255])).reshape(256,)).astype(np.float)
		histGreen = ((cv2.calcHist([img],[1],None, [256], [1,255])).reshape(256,)).astype(np.float)
		histRed = ((cv2.calcHist([img],[2],None, [256], [1,255])).reshape(256,)).astype(np.float)

		# normalize them
		histBlue_norm = histBlue / histBlue.sum()
		histGreen_norm = histGreen / histGreen.sum()
		histRed_norm = histRed / histRed.sum()

		# save normalized histograms
		histBlue_norm.tofile(histFile)
		histGreen_norm.tofile(histFile)
		histRed_norm.tofile(histFile)

		histFile.close() # close file
		
		# add noise
		if args["add_noise"] == True:
			for i in range(10):

				histName = args["output"] + args["type"] + "_" + str(i) + "_" + file[file.rfind('/')+1:len(file)-4] # create histogram file name
				histFile = open(histName, 'wb') # open histogram file

				# inizialize histogram with noise
				histBlueNoise = np.random.randint(-15,15,size=(256,),dtype=np.int)
				histGreenNoise = np.random.randint(-15,15,size=(256,),dtype=np.int)
				histRedNoise = np.random.randint(-15,15,size=(256,),dtype=np.int)

				# sum noise to origin histograms
				histBlueNoise += histBlue.astype(np.int)
				histGreenNoise += histGreen.astype(np.int)
				histRedNoise += histRed.astype(np.int)

				# normalize them
				histBlue_norm = (histBlueNoise / histBlueNoise.sum()).astype(np.float)
				histGreen_norm = (histGreenNoise / histGreenNoise.sum()).astype(np.float)
				histRed_norm = (histRedNoise / histRedNoise.sum()).astype(np.float)

				# save normalized histograms
				histBlue_norm.tofile(histFile)
				histGreen_norm.tofile(histFile)
				histRed_norm.tofile(histFile)

				print("[INFO] - Saved: ",histName)
		
		print("[INFO] - Saved: ",histName)
	print("[INFO] - Done!")
