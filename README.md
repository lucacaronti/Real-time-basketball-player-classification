# Real-time-basketball-player-classification
## Version 1:
A real time basketball player classification code written in python3 and based on opencv, Darknet and YOLO.
## Version 2:
From the second version there is a possibility to train a model for player classification with machine learning library scikit-learn. The model can be trained from two different types of datasets: one made from histograms of player (the model will be able to classify histograms), and the other one from images (the model will be able to classify directly images from color and forms) .

# Project description
## Version 1:
The project goal is to rank the players according to the color of the shirt, including referees.
There are two main types of code, one uses CPU (slower) and the other the Nvidia GPU (faster).
Unfortunately, I was unable to upload the video used to try and develop the code because it is under copyright
## Version 2:
Now there is a possibility to train a model for player classification using machine learning library scikit-learn (for object detection YOLO is still used).
# Dependency installation
## For version 1:
All the procedure refers to Manjaro Linux but for Ubuntu or other distros it's similar.
```sh
$ cd ~/Document
$ git clone https://github.com/lucacaronti/Real-time-basketball-player-classification.git
$ sudo pacman -S opencv base-devel opencv-samples hdf5 cuda cudnn
$ pip3 install scipy matplotlib numpy imutils --user
$ git clone https://github.com/pjreddie/darknet
$ cd darknet
```
Now we need to change the makefile inside the darknet folder to allow the compilation with cuda and cuda neural network. So set
```makefile
GPU=1
CUDNN=1
```
And in my case I had to change some other path:
```makefile
NVCC=/./opt/cuda/bin/nvcc
COMMON+= -DGPU -I/opt/cuda/include/
LDFLAGS+= -L/opt/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
```
At this point, we must start the compilation.
```sh
$ make -j$(nproc)
```
Now copy the shared object inside our folder
```sh
$ cp ~/Documents/darknet/libdarknet.so ~/Documents/Real-time-basketball-player-classification/code/
```
It's also necessary to download weight for the neural network
```sh
$ wget https://pjreddie.com/media/files/yolov3.weights ~/Documents/Real-time-basketball-player-classification/code/yolo-coco/
```
## For version 2:
In addiction to version 1, you must download scikit-learn package
```sh
$ sudo pacman -S python-scikit-learn
```
# Code usage
## Verison 1:
---
#### videoAverage
It's a script that does the average of frames to eliminate moving objects and get the background (e.g. basketball court). This only works if the shot is taken from a single point without moving the camera.
```sh
$ python videoAverage.py --input ../video/your_video.avi --output ../image/im_out.png
```

---
#### GPU_classifier
It's the code which classifies players using the GPU.
Inside the code there is a variable called **learningFrame** that indicates how many frames must be used to create the initial dataset of player histograms. For each of these frames will compare the detection and program will ask you what kind of player is (player 1, player 2, referee, nothing). At each detection the histogram dataset will upgrade by yourself in order to improve over time.
Usage:
```sh
$ python GPU_classifier.py --input ../video/your_video.avi --output ../video/output.avi --background ../image/background.png
```
The **background** option is optional, if no image is given the background subtraction is not done, which leads to a decrease in precision.

###### How to increase deep neural network precision
Inside the file **yolov3_GPU** there are two variable called **width** and **height**. Higher these variables are, higher is the accuracy of the neural network, but also the video memory used increases so the accuracy depends on how much VRAM you have available. **Width** and **height** must be multiples of 32.

---
#### CPU_classifier
It's the same code used for GPU_classifier but using CPU. The usage is the same.
To increase deep neural network precision you can modify two variable called **DNN_width** and **DNN_height**.

---
## Version 2:
At first, you need to create a dataset of images that will be used to train a model. For this you can launch a **CPU_classifier.py** with flag `-s` or `--save_detection` including a path where save the images.
For a good classifier should be saved at least 1000 images for each class.
Ones you finish to save detection, there will be the folder with inside 3 folders, one for each class (player1, player2, referee). So you can go inside and delete manually all wrong images.
### Model with histograms
In order to create histogram dataset, you need to use **createHistDataset.py** script, like follow:
```sh
$ python createHistDataset.py --input img_dataset_path/player1/ --type player1 --output output_hist_dataset_path/
```
If you want to increase size of dataset you can use `--add_noise True` flag, that from each image create 10 more with noise addiction.
The same procedure should be done for each class:
```sh
$ python createHistDataset.py --input img_dataset_path/player2/ --type player2 --output output_hist_dataset_path/
$ python createHistDataset.py --input img_dataset_path/referee/ --type referee --output output_hist_dataset_path/
```
Now is time for model training.
```sh
$ python trainHistModel.py --input hist_dataset_path/ --output model_name.sav
```
Threre are also other flags:
`--train_size` to select train size (from 0 = 0% to 1 = 100%). Default is 0.75
`--test_size` to select test size (from 0 = 0% to 1 = 100%). Default is 0.25
`--num_core` to use more core for training
At this point you have your model. To use it:
```sh
$ python machinelearningHistClassifier.py --input ../video/your_video.avi --output ../video/output.avi --background ../image/background.png --model your_model.sav
```
### Model with images
```sh
$ python createImageDataset.py --input img_dataset_path/player1/ --type player1 --output output_image_dataset_path/
```
The same procedure should be done for each class:
```sh
$ python createHistDataset.py --input img_dataset_path/player2/ --type player2 --output output_image_dataset_path/
$ python createHistDataset.py --input img_dataset_path/referee/ --type referee --output output_image_dataset_path/
```
Now is time for model training.
```sh
$ python trainImageModel.py --input image_dataset_path/ --output model_name.sav
```
Threre are also other flags:
`--train_size` to select train size (from 0 = 0% to 1 = 100%). Default is 0.75 . (Pay attenction because training require a lot of RAM)
`--test_size` to select test size (from 0 = 0% to 1 = 100%). Default is 0.25 .
`--num_core` to use more core for training
At this point you have your model. To use it:
```sh
$ python machinelearningImageClassifier.py --input ../video/your_video.avi --output ../video/output.avi --background ../image/background.png --model your_model.sav
```
# Special thanks to
Special thanks to the guys who developed this beautiful framework and also made it open source -> https://pjreddie.com/darknet/
During my project, I've followed some tuorial like:
https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
https://www.youtube.com/playlist?list=PLa-sizbCyh93evwIevvnjWFEH94N5giIG (In italian)
Thanks also to https://github.com/LucreziaT and https://github.com/dtessaro that have developed this project with me.