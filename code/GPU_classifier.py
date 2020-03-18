from darknet import *
import cv2
import argparse
import time
import numpy as np

def backgroundSubtraction(img, img_background,  kernel = np.ones((3,3),np.uint8)):

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
	hist_perc = hist/numSample
	hist_perc[0] = hist_perc[0] / hist_perc[0].sum()
	hist_perc[1] = hist_perc[1] / hist_perc[1].sum()
	hist_perc[2] = hist_perc[2] / hist_perc[2].sum()
	return hist_perc

def array_to_image(arr):
	# need to return old values to avoid python freeing memory
	arr = arr.transpose(2,0,1)
	c = arr.shape[0]
	h = arr.shape[1]
	w = arr.shape[2]
	arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
	data = arr.ctypes.data_as(POINTER(c_float))
	im = IMAGE(w,h,c,data)
	return im, arr

def createMask(image, points):
	image_copy1 = image.copy()
	cv2.fillConvexPoly( image_copy1, points, (255,255,255) )
	image_copy2 = image - image_copy1
	
	_, image_copy2 = cv2.threshold(image_copy2,1,255 ,cv2.THRESH_BINARY)
	image_copy2[:,:,0] = np.uint8(image_copy2[:,:,0] + image_copy2[:,:,1] + image_copy2[:,:,2])
	image_copy2[:,:,1] = image_copy2[:,:,0]
	image_copy2[:,:,2] = image_copy2[:,:,0]
	return image_copy2

if __name__ == "__main__":
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", help="path to input video")
	ap.add_argument("-o", "--output", help="path to output video")
	ap.add_argument("-b", "--background", help="path to background image", default="")
	args = vars(ap.parse_args())

	net = load_net(b"yolo-coco/yolov3_GPU.cfg", b"yolo-coco/yolov3.weights", 0)
	meta = load_meta(b"yolo-coco/coco.data")

	cap = cv2.VideoCapture(args["input"])

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	print("Width " + str(width))
	print("Height " + str(height))
	print("FPS " + str(cap.get(cv2.CAP_PROP_FPS)))
	video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	print("N frame video " + str(video_frames))
	video_format = cap.get(cv2.CAP_PROP_FOURCC)
	print("Video format" + str(video_format))

	# --- MENU --- #
	learingFrame = 1 # frame number in which to enter the data manually in order to create the initial dataset
	saveVideo = False
	# -------------#

	# Variable initalization
	hist_squad_1 = np.zeros((3,256), dtype=np.uint64)
	hist_squad_1_perc = np.zeros((3,256), dtype=np.float)
	hist_squad_2 = np.zeros((3,256), dtype=np.uint64)
	hist_squad_2_perc = np.zeros((3,256), dtype=np.float)
	hist_referee = np.zeros((3,256), dtype=np.uint64)
	hist_referee_perc = np.zeros((3,256), dtype=np.float)
	histDetection_perc = np.zeros((3,256), dtype=np.float)

	num_frame_elab = 0

	num_detection_squad_1 = 0
	num_detection_squad_2 = 0
	num_detection_referee = 0

	detection_squad_1_bool = False
	detection_squad_2_bool = False
	detection_referee_bool = False

	compHist_1 = 0
	compHist_2 = 0
	compHist_3 = 0

	area = 0
	area_tot = 0
	count_area = 0
	mean = 0
	mean_acc = 0
	start_1 = 0
	start_2 = 0
	start_3 = 0
	end_1 = 0
	end_2 = 0
	end_3 = 0

	pts = np.array([[750,1015],[3300,950],[3840,1310],[3840,1700],[1900,1790], [1,1700],[1,1530]], np.int32) # points to cut out the important part of image 
	pts = pts.reshape((-1,1,2)) # change array format
	ret , frame_range = cap.read() # read one frame of video
	if ret :
		if saveVideo == True:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, int(cap.get(cv2.CAP_PROP_FPS)) , (frame_range.shape[1], frame_range.shape[0]), True)
		maskCourt = createMask(frame_range, pts) # create a mask for the court
		if args["background"] == "":
			background = np.zeros((height,width,3), dtype=np.uint8)
		else:
			background = cv2.imread(args["background"]) # read background image
			background = cv2.bitwise_and(background, maskCourt) # cut out the important part of background
	while 1:
		
		start_1 = time.time() # take the start time 1
		
		ret, frame = cap.read() # read one frame
		
		if not ret:
			break 

		frame_bck = frame.copy() # create a copy of the frame
		num_frame_elab += 1
		boxes = []
		boxes_hist = []
		confidences = []
		classIDs = []
		num_detection = 0

		frame_court = cv2.bitwise_and(frame, maskCourt) # apply the mask to frame
		start_2 = time.time() # take the start time 2
		im, arr = array_to_image(frame_court) # convert the array to image
		detect_objs = detect(net, meta, im) # decect the objects with neural network
		end_2 = time.time() # take the end time 2
		for detect_obj in detect_objs:
			# save detections proprieties
			num_detection += 1
			name = detect_obj[0]
			prob = detect_obj[1]
			pos = detect_obj[2]
			x1 = int(pos[0]) - int(pos[2]/2)
			y1 = int(pos[1]) - int(pos[3]/2)
			boxes.append([x1,y1, int(pos[2]), int(pos[3])])
			confidences.append(float(prob))
			classIDs.append(name)
		
		start_3 = time.time() # take start time 3
		for i in range(num_detection): # cycle through detections
			if classIDs[i] == b"person":
				x1 = boxes[i][0]
				y1 = boxes[i][1]
				width_detection = boxes[i][2]
				height_detection = boxes[i][3]
				x2 = x1 + width_detection
				y2 = y1 + height_detection
				if width_detection *1.2 < height_detection: # check if 120% of width is smaller then height
					area = width_detection * height_detection
					area_tot += area
					count_area=count_area + 1
					mean = (area_tot/count_area)
					mean_acc = mean / 3

					color = [0,0,255]
					detection_img = backgroundSubtraction(frame[y1:(y1 + height_detection),x1:(x1+width_detection),:], background[y1:(y1 + height_detection),x1:(x1+width_detection),:]) # subtract background
					detection_WB =  cv2.cvtColor(detection_img, cv2.COLOR_BGR2GRAY) # convert image to gray scale
					
					histBlue = (cv2.calcHist([detection_img],[0],None,[256],[1,256])).reshape(256,) # create blue histogram, removing first values
					histGreen = (cv2.calcHist([detection_img],[1],None,[256],[1,256])).reshape(256,) # create green histogram, removing first values
					histRed = (cv2.calcHist([detection_img],[2],None,[256],[1,256])).reshape(256,) # create red histogram, removing first values
					histWB = (cv2.calcHist([detection_WB],[0],None,[256],[1,256])).reshape(256,) # create gray scale histogram, removing black color
					
					tot_pixel = histWB.sum() # calculate not black pixel
					histDetection_perc[0] = (histBlue / tot_pixel) # normalize hist
					histDetection_perc[1] = (histGreen / tot_pixel) # normalize hist
					histDetection_perc[2] = (histRed / tot_pixel) # normalize hist
						
					if(num_detection_squad_1 != 0):
						hist_squad_1_perc = normalizeHist(hist_squad_1, num_detection_squad_1) # normalize and mediates hist
						compHist_1 = cv2.compareHist(np.float32(hist_squad_1_perc), np.float32(histDetection_perc), cv2.HISTCMP_BHATTACHARYYA) # compare hist
					if(num_detection_squad_2 != 0): 
						hist_squad_2_perc = normalizeHist(hist_squad_2, num_detection_squad_2) # normalize and mediates hist
						compHist_2 = cv2.compareHist(np.float32(hist_squad_2_perc), np.float32(histDetection_perc), cv2.HISTCMP_BHATTACHARYYA) # compoare hist

					if(num_detection_referee != 0):
						hist_referee_perc = normalizeHist(hist_referee, num_detection_referee) # normalize and mediates hist
						compHist_3 = cv2.compareHist(np.float32(hist_referee_perc), np.float32(histDetection_perc), cv2.HISTCMP_BHATTACHARYYA) # compoare hist
					
					detection_squad_1_bool = False
					detection_squad_2_bool = False
					detection_referee_bool = False
					if num_frame_elab <= learingFrame:
						cv2.imshow("Detection", detection_img) # display the image
						cv2.waitKey(1)
						user_selection = input("1) squad 1, 2) squad 2, 3) referee, 4) nothing: ") # select the player
						if user_selection == "1":
							detection_squad_1_bool = True
						elif user_selection == "2":
							detection_squad_2_bool = True
						elif user_selection == "3":
							detection_referee_bool = True
					else:
						BHATTACHARYYA_min = min(compHist_1, compHist_2, compHist_3) # choose the minimum of 3 comparison
						if (BHATTACHARYYA_min == compHist_1) and (compHist_1 < 0.45):
								detection_squad_1_bool = True
						elif (BHATTACHARYYA_min == compHist_2) and (compHist_2 < 0.45):
								detection_squad_2_bool = True
						elif (BHATTACHARYYA_min == compHist_3) and (compHist_3 < 0.45):
								detection_referee_bool = True

					if (detection_squad_1_bool == True) and (area >= mean_acc) :
						color = [255,255,255] # set color -> black

						num_detection_squad_1+=1 # increase number of detection for squad 1
						
						# update histogram of squad 1
						hist_squad_1[0] += np.uint64(histBlue)
						hist_squad_1[1] += np.uint64(histGreen)
						hist_squad_1[2] += np.uint64(histRed)

					elif (detection_squad_2_bool == True) and (area >= mean_acc):
						color = [55,56,61] # set color -> white

						num_detection_squad_2+=1 # increase number of detection for squad 2

						# update histogram of squad 2
						hist_squad_2[0] += np.uint64(histBlue)
						hist_squad_2[1] += np.uint64(histGreen)
						hist_squad_2[2] += np.uint64(histRed)

					elif (detection_referee_bool == True) and (area >= mean_acc):
						color = [255,0,0] # set color -> blue

						num_detection_referee += 1 # increase number of detection for referee

						# update histogram of referee
						hist_referee[0] += np.uint64(histBlue)
						hist_referee[1] += np.uint64(histGreen)
						hist_referee[2] += np.uint64(histRed)
					else:
						color = [0,0,255] # set color -> red
					cv2.rectangle(frame_bck, (x1,y1), (x2,y2), color, 3) # apply the rectange

		end_3 = time.time() # take end time 3
		video = cv2.resize(frame_bck,(1920,1080),fx=0,fy=0, interpolation = cv2.INTER_CUBIC) # resize the video
		cv2.imshow("video", video)
		if saveVideo == True:
			writer.write(frame_bck)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		end_1 = time.time() # take end time 1
		print("Frame: {}, Ttot: {:5f}, TDNN: {:5f}, TDec: {:5f}".format(num_frame_elab, end_1 - start_1, end_2 - start_2, end_3 - start_3)) # print stats

	if saveVideo == True:
		writer.release()
	cap.release()
	cv2.destroyAllWindows()