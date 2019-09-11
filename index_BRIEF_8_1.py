import pickle
import sys
#Add lshash path to sys path to enable import
sys.path.insert(0, "/home/ratnakar/anaconda3/lib/python3.6/site-packages/lshash3-0.0.8/lshash")
import glob
from lshash import LSHash
import numpy as np
import cv2 as cv

#Store images from the image folder into a list
images = sorted(glob.glob('SIFT v2/images/*.jpg'))

#Write images to fileTable
fileTable = open('FileTable brief_8_1','wb')
pickle.dump(images,fileTable)
fileTable.close()

#Create LSHash object with appropriate parameters
lsh = LSHash(8, 32, 1,{"redis":{"host": "localhost", "port": 6379}}, "brief hashlen_8 hashtables_1.npz",True)

cnt = 0
keypoints = []
keypointIndex = 0

#Build feature descriptors for each image using BRIEF and store them
for img_ind in range(len(images)):
	img = cv.imread(images[img_ind])#If img is unable to be read we will see a runtime error in the form of an assertion failure
	# if img == None: 
	# 	raise Exception("could not load image ",images[img_ind])#If img is unable to be read we will see a runtime error in the form of an assertion failure
	star = cv.xfeatures2d.StarDetector_create()
	brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
	kp = star.detect(img,None)
	kp, des = brief.compute(img, kp)
	print(cnt)
	cnt += 1. 
	for keyPoint in des:
		#Index the keypoint into the LSHash object and update keypoints array
		lsh.index(keyPoint,(keypointIndex,));
		keypoints.append(img_ind)		
		keypointIndex += 1

#Store keypoints in a file
keypointTable = open('KeypointTable brief_8_1','wb')
pickle.dump(keypoints,keypointTable)
keypointTable.close()

"""
#Displaying File Table and Keypoint Table
print("Displaying fileTable")
fileTable = open('File Table','rb')
count = 0
while 1:
    try:
        print(pickle.load(fileTable))
        count+=1
#        print("Count = ",count)
    except (EOFError):
        break
fileTable.close()
print("Displaying keypointTable")
keypointTable = open('Keypoint Table','rb')
count = 1000
while 1:
    try:
        print(pickle.load(keypointTable))
        count+=1
#        print("Count = ",count)
    except (EOFError):
        break

#Displaying LSHash object (not in a very readable format)
print("\n\nLSHash object:")
data = np.load('brief hashlen_8 hashtables_1.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])
len(data['arr_0'])

"""
