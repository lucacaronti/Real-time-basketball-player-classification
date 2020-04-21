import argparse
import glob
import cv2
import numpy as np

def progresBar(count, finish, numbar): # funtion that print status of algorithm
	status = int((count/finish)*numbar)
	print("Status: [", end='')
	for i in range(status):
		print("#", end = '')
	for i in range(numbar - status):
		print(' ', end= '')
	print(']', end='\r')

if __name__ == "__main__":
	# Argument parser
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True, help="path of input data")
	ap.add_argument("-o", "--output", required=True, help="path to output data")
	ap.add_argument("-t", "--type", required=True, help="type of player (eg: player1, player2, referee, ecc.)")
	args = vars(ap.parse_args())

	files = glob.glob(args["input"]+"*.png") # read all '.png' images inside input folder
	print("[INFO] - Found ", len(files), " images") # print number of files found
	count = 0 # counter for status bar
	for file in files:
		imgName = args["output"] + args["type"] + "_" + file[file.rfind('/')+1:len(file)-4] # create image file name
		imgFile = open(imgName, 'wb') # open image file
		img = cv2.imread(file) # read image
		arrayImg = np.array(img) # cast image into numpy array
		flattenImg = (arrayImg.flatten()).astype(np.uint8) # flatten array into one dimension
		flattenImg.tofile(imgFile) # save flatten image
		imgFile.close() # close file
		count += 1
		progresBar(count, len(files), 100) # print status bar
	print("\r\n[INFO] - Done!")

