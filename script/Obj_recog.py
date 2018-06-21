import cv2
import os
import numpy as np
import Obj_utilscv

# A Python class, called ImageFeature, has been created
# That will contain for each of the images in the database,
# Information needed to compute object recognition.
class ImageFeature(object):
    def __init__(self, nameFile, shape, imageBinary, kp, desc):
        # File name
        self.nameFile = nameFile
        # Shape of the image
        self.shape = shape
        # Binary image data
        self.imageBinary = imageBinary
        # Keypoints of the image once the feature detection algorithm is applied
        self.kp = kp
        # Descriptors of detected features
        self.desc = desc
        # Matchings of the image of the database with the image of the webcam
        self.matchingWebcam = []
        # Matching the webcam with the current image of the database.
        self.matchingDatabase = []

    # Allows to empty previously calculated matching for a new image
    def clearMatchingMutuos(self):
        self.matchingWebcam = []
        self.matchingDatabase = []

# Calculating function, for each one of the methods of calculation of features,
# The features of each of the images in the directory "a"
def loadModelsFromDirectory():
    # The method returns a dictionary. The key is the feature algorithm
    # While the value is a list of objects of type ImageFeature
    # Where all the data of the features of the images of the
    # Database.
    dataBase = dict([('SIFT', []), ('AKAZE', []), ('SURF', []),
                    ('ORB', []), ('BRISK', [])])
    # The number of features has been limited to 250, so that the algorithm goes smoothly.
    surf = cv2.xfeatures2d.SURF_create(2500)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
    akaze = cv2.AKAZE_create()
    orb = cv2.ORB_create(400)
    brisk = cv2.BRISK_create()
    for imageFile in os.listdir("a"):
        # Upload image with OpenCV
        colorImage = cv2.imread("a/" + str(imageFile)) #colorImage = cv2.imread("a_h6" + str(imageFile))
        # We pass the grayscale image
        currentImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
        # We perform a resize of the image, so that the compared image is equal
        kp, desc = surf.detectAndCompute(currentImage, None)
        # Features are loaded with SURF
        dataBase["SURF"].append(ImageFeature(colorImage, currentImage.shape, colorImage, kp, desc))
        # Features are loaded with SIFT
        kp, desc = sift.detectAndCompute(currentImage, None)
        dataBase["SIFT"].append(ImageFeature(colorImage, currentImage.shape, colorImage, kp, desc))
        # Features are loaded with AKAZE
#        kp, desc = akaze.detectAndCompute(currentImage, None)
#        dataBase["AKAZE"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
        # Features are loaded with ORB
#        kp, desc = orb.detectAndCompute(currentImage, None)
#        dataBase["ORB"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
        # Features are loaded with BRISK
 #       kp, desc = brisk.detectAndCompute(currentImage, None)
 #       dataBase["BRISK"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
    return dataBase

# Function responsible for calculating mutual matching, but nesting loops
# It is a very slow solution because it does not take advantage of Numpy power
# We do not even put a slider to use this method as it is very slow
def findMatchingMutuos(selectedDataBase, desc, kp):
    for imgFeatures in selectedDataBase:
        imgFeatures.clearMatchingMutuos()
        for i in range(len(desc)):
            primerMatching = None
            canditatoDataBase = None
            matchingsecond = None
            candidateWebCam = None
            for j in range(len(imgFeatures.desc)):
                valueMatching = np.linalg.norm(desc[i] - imgFeatures.desc[j])
                if (primerMatching is None or valueMatching < primerMatching):
                    primerMatching = valueMatching
                    canditatoDataBase = j
            for k in range(len(desc)):
                valueMatching = np.linalg.norm(imgFeatures.desc[canditatoDataBase] - desc[k])
                if (matchingsecond is None or valueMatching < matchingsecond):
                    matchingsecond = valueMatching
                    candidateWebCam = k
            if not candidateWebCam is None and i == candidateWebCam:
               imgFeatures.matchingWebcam.append(kp[i].pt)
               imgFeatures.matchingDatabase.append(imgFeatures.kp[canditatoDataBase].pt)
    return selectedDataBase

# Function responsible for calculating the mutual matching of a webcam image,
# With all the images in the database. Receive as input parameter
# The database based on the method of calculation of features used
# In the image input of the webcam.
def findMatchingMutuosOptimizado(selectedDataBase, desc, kp):
    # The algorithm is repeated for each image in the database.
    for img in selectedDataBase:
        img.clearMatchingMutuos()
        for i in range(len(desc)):
            # The standard of difference of the current descriptor is calculated with all
            # Descriptors of the image of the database. we got
            # No loops and making use of Numpy broadcasting, all distances
            # Between the current descriptor with all descriptors of the current image
            distanceListFromWebCam = np.linalg.norm(desc[i] - img.desc, axis=-1)
            # Obtain the candidate that is the shortest distance from the current descriptor
            candidatoDataBase = distanceListFromWebCam.argmin()
            # It is checked if the matching is mutual, that is, if it is true
            # In the other direction. That is, it is verified that the candidateDatabase
            # Has the current descriptor as best matching
            distanceListFromDataBase = np.linalg.norm(img.desc[candidatoDataBase] - desc, axis=-1)
            candidatoWebCam = distanceListFromDataBase.argmin()
            # If mutual matching is fulfilled, it is stored for later processing
            if (i == candidatoWebCam):
                img.matchingWebcam.append(kp[i].pt)
                img.matchingDatabase.append(img.kp[candidatoDataBase].pt)
        # For convenience they become Numpy ND-Array
        img.matchingWebcam = np.array(img.matchingWebcam)
        img.matchingDatabase = np.array(img.matchingDatabase)
    return selectedDataBase


# This function calculates the best image based on the number of inliers
# That each image in the database has with the image obtained from
# The web camera.
def calculateBestImageByNumInliers(selectedDataBase, projer, minInliers):
    if minInliers < 10: #15
        minInliers = 10
    bestIndex = None
    bestMask = None
    numInliers = 0
    # For each of the images
    for index, imgWithMatching in enumerate(selectedDataBase):
        # Compute the RANSAC algorithm to calculate the number of inliers
        _, mask = cv2.findHomography(imgWithMatching.matchingDatabase, imgWithMatching.matchingWebcam, cv2.RANSAC, projer)
        if not mask is None:
            # Check, from the mask the number of inliers.
            # If the number of inliers is greater than the minimum number of inliers,
            # And is a maximum (has more inliers than the previous image),
            # Then it is considered to be the image that matches the object
            # Stored in the database.
            countNonZero = np.count_nonzero(mask)
            if (countNonZero >= minInliers and countNonZero > numInliers):
                numInliers = countNonZero
                bestIndex = index
                bestMask = (mask >= 1).reshape(-1)
    # If an image has been obtained as the best image and, therefore,
    # Must have a minimum number of inilers, then it is finally calculated
    # The keypoints that are inliers from the mask obtained in findHomography
    # And is returned as the best image.
    if not bestIndex is None:
        bestImage = selectedDataBase[bestIndex]
        inliersWebCam = bestImage.matchingWebcam[bestMask]
        inliersDataBase = bestImage.matchingDatabase[bestMask]
        return bestImage, inliersWebCam, inliersDataBase
    return None, None, None


# This function calculates the affinity matrix A, paints a rectangle around
# Of detected object and paint in a new window the image of the database
# Corresponding to the recognized object.
def calculateAffinityMatrixAndDraw(bestImage, inliersDataBase, inliersWebCam, imgout):
    # The affinity mat A
    A = cv2.estimateRigidTransform(inliersDataBase, inliersWebCam, fullAffine=True)
    A = np.vstack((A, [0, 0, 1]))

    # Calculate the points of the rectangle occupied by the recognized object
    a = np.array([0, 0, 1], np.float)
    b = np.array([bestImage.shape[1], 0, 1], np.float)
    c = np.array([bestImage.shape[1], bestImage.shape[0], 1], np.float)
    d = np.array([0, bestImage.shape[0], 1], np.float)
    center = np.array([float(bestImage.shape[0]) / 2,
                       float(bestImage.shape[1]) / 2, 1], np.float)

    # Multiply the points of the virtual space, to convert them into
    # real image points
    a = np.dot(A, a)
    b = np.dot(A, b)
    c = np.dot(A, c)
    d = np.dot(A, d)
    center = np.dot(A, center)

    # The points are dehomogenized
    areal = (int(a[0] / a[2]), int(a[1] / b[2]))
    breal = (int(b[0] / b[2]), int(b[1] / b[2]))
    creal = (int(c[0] / c[2]), int(c[1] / c[2]))
    dreal = (int(d[0] / d[2]), int(d[1] / d[2]))
    centerreal = (int(center[0] / center[2]), int(center[1] / center[2]))

    # The polygon and the file name of the image are painted in the middle of the polygon
    points = np.array([areal, breal, creal, dreal], np.int32)
    cv2.polylines(imgout, np.int32([points]), 1, (0, 255, 255), thickness=3)
#    Obj_utilscv.draw_str(imgout, centerreal, bestImage.nameFile.upper())
    # The detected object is displayed in a separate window
    cv2.imshow('ImageDetector', bestImage.imageBinary)