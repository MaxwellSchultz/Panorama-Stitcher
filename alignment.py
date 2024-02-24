import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
            KeyPoint.pt holds a tuple of pixel coordinates (x, y)
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    #BEGIN TODO 2a
    # Construct the A matrix that will be used to compute the homography
    # based on the given set of matches among feature sets f1 and f2.

    raise Exception("TODO 2 in alignment.py not implemented")

    #END TODO

    if A_out is not None:
        A_out[:] = A

    x = minimizeAx(A) # find x that minimizes ||Ax||^2 s.t. ||x|| = 1


    H = np.eye(3) # create the homography matrix

    #BEGIN TODO 3b
    #Fill the homography H with the correct values

    raise Exception("TODO 3b in alignment.py not implemented")

    #END TODO

    return H

def minimizeAx(A):
    """ Given an n-by-m array A, return the 1-by-m vector x that minimizes
    ||Ax||^2 subject to ||x|| = 1.  This turns out to be the right singular
    vector of A corresponding to the smallest singular value."""
    #BEGIN TODO 3a

    raise Exception("TODO 3a in alignment.py not implemented")

    #END TODO 3a

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslate) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call computeHomography.
    #This function should also call getInliers and, at the end,
    #least_squares_fit.
    raise Exception("TODO 4 in alignment.py not implemented")
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        # Determine if the ith matched feature f1[matches[i].queryIdx], when
        # transformed by M, is within RANSACthresh of its match in f2.
        # If so, append i to inliers
        #TODO-BLOCK-BEGIN
        raise Exception("TODO 5 in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        #BEGIN TODO 6 :Compute the average translation vector over all inliers.
        # Fill in the appropriate entries of M to represent the average
        # translation transformation.
        raise Exception("TODO 6 in alignment.py not implemented")
        #END TODO

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers. This should call
        # computeHomography.
        raise Exception("TODO 7 in alignment.py not implemented")
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

