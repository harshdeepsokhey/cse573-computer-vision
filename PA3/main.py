# main
# import the necessary packages
from searcher import Searcher
import numpy as np
import argparse
import os
import pickle
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
    help = "Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required = True,
    help = "Path to where we stored our index")
args = vars(ap.parse_args())
 
# load the index and initialize our searcher
index = pickle.loads(open(args["index"], "rb").read())
searcher = Searcher(index)

# loop over images in the index -- we will use each one as
# a query image
output = {}
for (query, queryFeatures) in index.items():
    # perform the search using the current query
    results = searcher.search(queryFeatures)
 
    # load the query image and display it
    path = os.path.join(args["dataset"], query)
    queryImage = cv2.imread(path)
    queryImage = cv2.resize(queryImage,(400,166))

    #cv2.imshow("Query", queryImage)
    print("query: {}".format(query))
 
    # initialize the two montages to display our results --
    # we have a total of 25 images in the index, but let's only
    # display the top 10 results; 5 images per montage, with
    # images that are 400x166 pixels
    montageA = np.zeros((166 * 5, 400, 3), dtype = "uint8")
    montageB = np.zeros((166 * 5, 400, 3), dtype = "uint8")
 
    output[str(query)] = []
    # loop over the top ten results
    for j in range(0, 10):
        # grab the result (we are using row-major order) and
        # load the result image
        (score, imageName) = results[j]
        path = os.path.join(args["dataset"], imageName)
        result = cv2.imread(path)
        result = cv2.resize(result, (400,166))
        print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))
 
        output[str(query)].append([imageName, score])
        # check to see if the first montage should be used
        if j < 5:
            montageA[j * 166:(j + 1) * 166, :] = result
 
        # otherwise, the second montage should be used
        else:
            montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = result
 
    # show the results

    cv2.imwrite('./results/query_'+query,queryImage)
    cv2.imwrite('./results/mA_'+query, montageA)
    cv2.imwrite('./results/mB_'+query, montageB)

