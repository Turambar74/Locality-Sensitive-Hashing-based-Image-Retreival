import sys
#Add lshash path to sys path to enable import
sys.path.insert(0, "/home/ratnakar/anaconda3/lib/python3.6/site-packages/lshash3-0.0.8/lshash")
from lshash import LSHash
import pickle
import glob
import numpy as np
import cv2 as cv
from collections import Counter
from sklearn import preprocessing

#Change these three lines depending on which database is to be queried
lsh = LSHash(8, 32, 1, {"redis":{"host": "localhost", "port": 6379}}, "brief hashlen_8 hashtables_1.npz")
fileTablePath = 'FileTable brief_8_1'
keypointTablePath = 'KeypointTable brief_8_1'

#Load from fileTable and keypointTable
fileTable = open(fileTablePath,'rb')
images = pickle.load(fileTable)
fileTable.close()
keypointTable = open(keypointTablePath,'rb')
keypoints = pickle.load(keypointTable)
keypointTable.close()

#Query
test_images = sorted(glob.glob('SIFT v2/tests/*.jpg'))

"""
test_img = test_images[2]
img = cv.imread(test_img)
star = cv.xfeatures2d.StarDetector_create()
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
kp = star.detect(img,None)
kp, des = brief.compute(img, kp)
results = Counter()
count = 0
for kpt in des:
	print(count)
	count += 1
	matches = lsh.query(kpt,3,"hamming")
	for match in matches:
		results.update(Counter((keypoints[eval(match[0].decode())[1][0]],)))

multiplier = 100/len(des)
print("Best matches for ",test_img," are:")
for res in results.most_common(20):
	print(images[res[0]]," : ",res[1]*multiplier,"%")

"""
count = 0
for test_img in test_images:
	#Cal SIFT descriptors for each image
	img = cv.imread(test_img)
	#if img == None: 
	#	raise Exception("could not load image ",test_img)
	star = cv.xfeatures2d.StarDetector_create()
	brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
	kp = star.detect(img,None)
	kp, des = brief.compute(img, kp)
	#Store the best match for each descriptor in a Counter object
	print("Img count = ",count)
	count += 1
	results = Counter()
	count2 = 0
	for kpt in des:
		print(count2)
		count2+=1
		matches = lsh.query(kpt,3,"hamming")
		for match in matches:
			results.update(Counter((keypoints[eval(match[0].decode())[1][0]],)))
	#Display matches and strength of matches
	multiplier = 100/len(des)
	#matchStrength = [results[x]*multiplier for x in results]
	print("Best matches for ",test_img," are:")
	for res in results.most_common(5):
		print(images[res[0]]," : ",res[1]*multiplier,"%")
