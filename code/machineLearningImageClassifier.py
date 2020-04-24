import numpy as np
import argparse
import time
import cv2
import imutils
import pickle

def backgroundSubtraction(img, img_background,  kernel = np.ones((3,3),np.uint8)):
	"""
	@Description: do a background subtraction
	@Parameters:
		- img -> image to subtract the background from
		- img_background -> background image
		- kernel -> kernel for morphology operations
	@Return: image without background
	"""

	img_copy = img.copy()

	diffRGB = cv2.absdiff(img, img_background)

	_ , frame_BIN = cv2.threshold(diffRGB, 25, 255, cv2.THRESH_BINARY)

	frame_BIN = np.uint8(frame_BIN[:,:,0] + frame_BIN[:,:,1] + frame_BIN[:,:,2])

	frame_BIN = cv2.morphologyEx(frame_BIN ,cv2.MORPH_OPEN,kernel, iterations = 2)

	frame_BIN = cv2.morphologyEx(frame_BIN, cv2.MORPH_CLOSE, kernel, iterations = 8)

	img[:,:,0] = cv2.bitwise_and(img_copy[:,:,0], frame_BIN)
	img[:,:,1] = cv2.bitwise_and(img_copy[:,:,1], frame_BIN)
	img[:,:,2] = cv2.bitwise_and(img_copy[:,:,2], frame_BIN)

	return img

def normalizeHist(hist, numSample):
	"""
	@Description: Normalize histogram from 0 to 1
	@Parameters:
		- hist -> histogram
		- numSample -> number of sample inside the histogram
	@Return: normlized histogram
	"""
	hist_perc = hist/numSample
	hist_perc[0] = hist_perc[0] / hist_perc[0].sum()
	hist_perc[1] = hist_perc[1] / hist_perc[1].sum()
	hist_perc[2] = hist_perc[2] / hist_perc[2].sum()
	return hist_perc

def createMask(image, points):
	"""
	@Description: create a mask from given points
	@Parameters:
		- image -> source image, it only serves for dimensions.
		- points -> points of the polygon, the mask is created to eliminate points outside the polygon
	@Return: mask image, it's white inside the polygon and black outside. To apply the mask do an AND operation to image.
	"""
	maskImg = np.zeros(image.shape, dtype=np.uint8)
	cv2.fillConvexPoly( maskImg, points, (255,255,255) )
	return maskImg

if __name__ == "__main__":
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input",required=True, help="path to input video")
	ap.add_argument("-o", "--output",required=True, help="path to output video")
	ap.add_argument("-m", "--model", required=True, help="file of machine learning model")
	ap.add_argument("-b", "--background", help="path to background image", default="")
	args = vars(ap.parse_args())

	# load the COCO class labels our YOLO model was trained on
	LABELS = open("yolo-coco/coco.names").read().strip().split("\n")

	# load machine learning model
	model = pickle.load(open(args["model"], 'rb'))

	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	vs = cv2.VideoCapture(args["input"])
	if vs.isOpened() != True:
		print("[*ERROR*] {} isn't a valid video".format(args["input"]))
	else:
		prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
		total = int(vs.get(prop))
		print("[INFO] {} total frames in video".format(total))
		width_video = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
		height_video = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
		print("[INFO] fps: " + str(vs.get(cv2.CAP_PROP_FPS)))
		print("[INFO] width video: " + str(width_video))
		print("[INFO] height frame: " + str(height_video))

		pts = np.array([[750,1015],[3300,950],[3840,1310],[3840,1700],[1900,1790], [1,1700],[1,1530]], np.int32) # points to cut out the important part of image 
		pts = pts.reshape((-1,1,2)) # change array format

		_ , frame_range = vs.read() # read one frame of video

		maskCourt = createMask(frame_range, pts) # create a mask for the court

		if args["background"] == "":
			background = np.zeros((height_video,width_video,3), dtype=np.uint8)
		else:
			background = cv2.imread(args["background"]) # read background image
			background = cv2.bitwise_and(background, maskCourt) # cut out the important part of background

		# --- MENU --- #
		DNN_width = 1280
		DNN_height = 1280
		# -------------#


		# Variable initalization
		writer = None
		(W, H) = (None, None)

		num_frame_elab = 0
		num_detection = 0

		num_detection_squad_1 = 0
		num_detection_squad_2 = 0
		num_detection_referee = 0

		detection_squad_1_bool = False
		detection_squad_2_bool = False
		detection_referee_bool = False
		detection_overlapping_bool = False
		
		area = 0
		area_tot = 0
		count_area = 0
		mean = 0
		mean_acc = 0
		start1 = 0
		start2 = 0
		start3 = 0
		end1 = 0
		end2 = 0
		end3 = 0

		# loop over frames from the video file stream
		while True:

			start1 = time.time()
			num_frame_elab += 1
			# read the next frame from the file
			(grabbed, frame) = vs.read()

			# if the frame was not grabbed, then we have reached the end of the stream
			if not grabbed:
				break

			# if the frame dimensions are empty, grab them
			if W is None or H is None:
				(H, W) = frame.shape[:2]

			frame_court = cv2.bitwise_and(frame, maskCourt)

			# construct a blob from the input frame and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes
			# and associated probabilities
			blob = cv2.dnn.blobFromImage(frame_court, 1 / 255.0, (DNN_width, DNN_height), swapRB=True, crop=False)
			net.setInput(blob)
			start2 = time.time()
			layerOutputs = net.forward(ln)
			end2 = time.time()
			# check if the video writer is None
			if writer is None:
				# initialize our video writer
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)
		
				# some information on processing single frame
				if total > 0:
					elap = (end2 - start2)
					print("[INFO] single frame took {:.4f} seconds".format(elap))
					print("[INFO] estimated total time to finish: {:.4f}s = {:.4f}h".format(
						elap * total, (elap*total)/(60*60)))
			
			# initialize our lists of detected bounding boxes, confidences,
			# and class IDs, respectively
			boxes = []
			boxes_hist = []
			confidences = []
			classIDs = []
				# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability)
					# of the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]
		
					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > 0.5:
						# scale the bounding box coordinates back relative to
						# the size of the image, keeping in mind that YOLO
						# actually returns the center (x, y)-coordinates of
						# the bounding box followed by the boxes' width and
						# height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						width_hist = int(width - 0.3*width) # reduce box of 30%
						height_hist = int(height - 0.3*height) #reduce box of 30%

		
						# use the center (x, y)-coordinates to derive the top
						# and and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))
						x_hist = int(centerX - (width_hist / 2))
						y_hist = int(centerY - (height_hist / 2))
		
						# update our list of bounding box coordinates,
						# confidences, and class IDs
						boxes.append([x, y, int(width), int(height)])
						boxes_hist.append([x_hist, y_hist, width_hist, height_hist])
						confidences.append(float(confidence))
						classIDs.append(classID)
			
			# apply non-maxima suppression to suppress weak, overlapping
			# bounding boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
			frame_bck = frame.copy()
			frame = backgroundSubtraction(frame, background)
			start3 = time.time()
			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					if LABELS[classIDs[i]] == "person":
						# extract the bounding box coordinates
						(x_hist, y_hist) = (boxes_hist[i][0], boxes_hist[i][1])
						(w_hist, h_hist) = (boxes_hist[i][2], boxes_hist[i][3])
						(x, y) = (boxes[i][0], boxes[i][1])
						(w, h) = (boxes[i][2], boxes[i][3])

						# applying the area function
						area = w * h
						area_tot += area
						count_area=count_area + 1
						mean = (area_tot/count_area)
						mean_acc = mean / 3

						color = [0,0,255] # inizialize color -> default is red
						if w*1.2 < h: #check if width is at least 20% smaller than height
							num_detection += 1
							detection_img = frame[y_hist:(y_hist+h_hist),x_hist:(x_hist+w_hist),:] # cut out the detection

							detection_img_resized = cv2.resize(detection_img,(50,100),fx=0,fy=0, interpolation = cv2.INTER_CUBIC) # normalize detection dimensions
							arrayImg = np.array(detection_img_resized) # cast image into numpy array
							flattenImg = (arrayImg.flatten()).astype(np.uint8) # convert array into one dimension
							
							detection_squad_1_bool = False
							detection_squad_2_bool = False
							detection_referee_bool = False
							detection_overlapping_bool = False

							# predic output
							prediction = model.predict(flattenImg.reshape(1,-1))
							if prediction == [1]:
								detection_squad_1_bool = True
							elif prediction == [2]:
								detection_squad_2_bool = True
							elif prediction == [3]:
								detection_referee_bool = True
							elif prediction == [4]:
								detection_overlapping_bool = True

							if (detection_squad_1_bool == True) and (area >= mean_acc) :
								color = [255,255,255] # set color -> white

								num_detection_squad_1+=1 # increase number of detection for squad 1

							elif (detection_squad_2_bool == True) and (area >= mean_acc):
								color = [55,56,61] # set color -> black

								num_detection_squad_2+=1 # increase number of detection for squad 2

							elif (detection_referee_bool == True) and (area >= mean_acc):
								color = [255,0,0] # set color -> blue

								num_detection_referee += 1 # increase number of detection for referee
							elif (detection_overlapping_bool == True) and (area >= mean_acc):
								color = [0,255,0] # set color -> blue
							else:
								color = [0,0,255] # set color -> red
							cv2.rectangle(frame_bck, (x, y), (x + w, y + h), color, 2)

			end3 = time.time() # take end time 3
			video = cv2.resize(frame_bck,(1920,1080),fx=0,fy=0, interpolation = cv2.INTER_CUBIC) # resize the video
			cv2.imshow("video", video)
			writer.write(frame_bck)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			end1 = time.time() # take end time 1
			
			print("[INFO] num frame done: {}, num detection {}, avg squad 1: {:3f}, avg squad 2: {:3f}, avg referee {:3f}".format(num_frame_elab, num_detection, num_detection_squad_1/num_frame_elab, num_detection_squad_2/num_frame_elab, num_detection_referee/num_frame_elab))
			print("[INFO] Ttot: {:5f}, TDNN: {:5f}, TDec: {:5f}".format(end1 - start1, end2 - start2, end3 - start3)) # print stats

	vs.release()
	writer.release()
	cv2.destroyAllWindows()