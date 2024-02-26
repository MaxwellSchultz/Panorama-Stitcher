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
    
    # print("f1",f1)
    # print("f2",f2)
    # print("matches", matches)
    # i = 0
    # for match in matches:
    #     print("match",i,"index f1:",match.queryIdx)
    #     print("match",i,"index f2:",match.trainIdx)
    #     print("match",i,"dist:",match.distance)
    #     i += 1
    
    firstPass = True
    A = None
    # i = 0
    for match in matches:
        f1_index = match.queryIdx
        f2_index = match.trainIdx
        
        f1_x, f1_y = f1[f1_index].pt
        f2_x, f2_y = f2[f2_index].pt
        # print("f1_index",f1_index,"f2_index",f2_index)
        # print("f1 (x,y)",f1_x,f1_y)
        # print("f2 (x,y)",f2_x,f2_y)
        
        if firstPass:
            A = np.array([[f1_x,f1_y,1,0,0,0,(-f2_x * f1_x),(-f2_x * f1_y),-f2_x],
                                    [0,0,0,f1_x,f1_y,1,(-f2_y * f1_x), (-f2_y * f1_y), -f2_y]])
            firstPass = False
        else:
            curr_mat = np.array([[f1_x,f1_y,1,0,0,0,(-f2_x * f1_x),(-f2_x * f1_y),-f2_x],
                                    [0,0,0,f1_x,f1_y,1,(-f2_y * f1_x), (-f2_y * f1_y), -f2_y]])
            A = np.vstack((A,curr_mat))
            
        # print("Pass",i,A)
        # i += 1
        
    # print("total_mat",A)

    # Construct the A matrix that will be used to compute the homography
    # based on the given set of matches among feature sets f1 and f2.

    if A_out is not None:
        A_out[:] = A

    x = minimizeAx(A) # find x that minimizes ||Ax||^2 s.t. ||x|| = 1

    H = np.eye(3) # create the homography matrix
    
    # print("x",x)

    #Fill the homography H with the correct values
    H = x.reshape(3,3)
    
    # Normalize homography matrix
    H = H / H[2,2]
    # print(H)

    return H

def minimizeAx(A):
    """ Given an n-by-m array A, return the 1-by-m vector x that minimizes
    ||Ax||^2 subject to ||x|| = 1.  This turns out to be the right singular
    vector of A corresponding to the smallest singular value."""
    
    U, S, x = np.linalg.svd(A)
    x = x[-1]
    return x

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
    
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslate) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call computeHomography.
    #This function should also call getInliers and, at the end,
    #least_squares_fit.
    
    max_numInliers = 0
    best_M = np.eye(3)
    
    for i in range(nRANSAC):
        s = 0
        if (m == 0):
            s = 1
        elif (m == 1):
            s = 4
            
        minimal_set = random.sample(range(len(matches)), s)
        
        M = leastSquaresFit(f1,f2,matches,m,minimal_set)
        inliers = getInliers(f1,f2,matches,M,RANSACthresh)
        numInliers = len(inliers)
        
        if (numInliers > max_numInliers):
            max_numInliers = numInliers
            best_M = M
        
    return best_M

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
        # Determine if the ith matched feature f1[matches[i].queryIdx], when
        # transformed by M, is within RANSACthresh of its match in f2.
        # If so, append i to inliers
        
        f1_index = matches[i].queryIdx
        f2_index = matches[i].trainIdx
        
        f1_x, f1_y = f1[f1_index].pt
        f2_x, f2_y = f2[f2_index].pt
        
        source_vec = np.array([f1_x,f1_y,1])
        
        dest_vec = M.dot(source_vec)
        
        # print("dest vec", dest_vec)
        
        dest_vec = dest_vec / dest_vec[2]
        
        # print("dest vec", dest_vec)
        
        euc_dist = math.sqrt((f2_x - dest_vec[0])**2 + (f2_y - dest_vec[1])**2)
        
        if (euc_dist < RANSACthresh):
            inlier_indices.append(i)

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
        
        for index in inlier_indices:
        #Compute the average translation vector over all inliers.
        # Fill in the appropriate entries of M to represent the average
        # translation transformation.
            f1_index = matches[index].queryIdx
            f2_index = matches[index].trainIdx
        
            f1_x, f1_y = f1[f1_index].pt
            f2_x, f2_y = f2[f2_index].pt
            
            u += math.sqrt((f2_x - f1_x)**2)
            v += math.sqrt((f2_y - f1_y)**2)
            
        u /= len(inlier_indices)
        v /= len(inlier_indices)
        
        M[0][2] = u
        M[1][2] = v

    elif m == eHomography:
        #Compute a homography M using all inliers. This should call
        # computeHomography.
        
        inlier_matches = []
        for index in inlier_indices:
            inlier_matches.append(matches[index])
        
        M = computeHomography(f1,f2,inlier_matches)
    else:
        raise Exception("Error: Invalid motion model.")

    return M

