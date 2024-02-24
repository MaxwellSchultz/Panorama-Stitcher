import os
import cv2
import numpy as np

def warpLocal(src, uv):
    '''
    Input:
        src --    source image in a numpy array with values in [0, 255].
                  The dimensions are (rows, cols, color bands BGR).
        uv --     warped image in terms of addresses of each pixel in the source
                  image in a numpy array.
                  The dimensions are (rows, cols, addresses of pixels [:,:,0]
                  are x (i.e., cols) and [:,,:,1] are y (i.e., rows)).
    Output:
        warped -- resampled image from the source image according to provided
                  addresses in a numpy array with values in [0, 255]. The
                  dimensions are (rows, cols, color bands BGR).
    '''
    width = src.shape[1]
    height  = src.shape[0]
    mask = cv2.inRange(uv[:,:,1],0,height-1)&cv2.inRange(uv[:,:,0],0,width-1)
    warped = cv2.remap(src, uv[:, :, 0].astype(np.float32),\
             uv[:, :, 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    img2_fg = cv2.bitwise_and(warped,warped,mask = mask)
    return img2_fg


def computeSphericalWarpMappings(dstShape, f, k1, k2):
    '''
    Compute the spherical warp. Compute the addresses of each pixel of the
    output image in the source image.

    Input:
        dstShape -- shape of input / output image in a numpy array.
                    [number or rows, number of cols, number of bands]
        f --        focal length in pixel as int
                    See assignment description on how to find the focal length
        k1 --       horizontal distortion as a float
        k2 --       vertical distortion as a float
    Output:
        uvImg --    2-channel image with input coordinates for each output
                    pixel. uvImg[y,x,:] gives the input pixel coordinates
                    output pixel (x,y).
                    The dimensions are (rows, cols, 2); [:,:,0] are x (i.e.,
                    column) coordinates and [:,:,1] are y (i.e., row)
                    coordinates).
    '''

    # calculate minimum y value
    vec = np.zeros(3)
    vec[0] = np.sin(0.0) * np.cos(0.0)
    vec[1] = np.sin(0.0)
    vec[2] = np.cos(0.0) * np.cos(0.0)
    min_y = vec[1]

    # calculate spherical coordinates
    # (xc,yc) is the spherical image coordinates.
    one = np.ones((dstShape[0],dstShape[1]))
    xc = one * np.arange(dstShape[1])
    yc = one.T * np.arange(dstShape[0])
    yc = yc.T

    # BEGIN TODO 1 - 597P or extra credit ONLY
    # remove the return statement in the TODO block below, 
    # then add code to apply the spherical correction, i.e.,
    # 1. convert spherical image pixel coords to angular coordinates 
    #    with (0,0) in the center of the image
    # 2. convert spherical coordinates to Euclidean coordinates on a unit sphere
    # 3. project the point to the z=1 plane at (xt/zt,yt/zt,1),
    # 4. distort with radial distortion coefficients k1 and k2
    # 5. rescale by focal length and translate origin back to the corner
    # Your code should use xc, yc, the spherical image coordinates for each pixel,
    # as input and compute xt, yt, the input pixel coordinates for each pair of
    # spherical coordinates. All of the above should have the shape
    # (img_height, img_width)

    # 497P: leave the return statement; the input image remains unchanged
    # 597P: remove this return statement and fill in the TODO block below
    return np.mgrid[:dstShape[0], :dstShape[1]].transpose((1,2,0))[:,:,::-1]

    # END TODO 1

def warpSpherical(image, focalLength, k1=-0.21, k2=0.26):
    '''
    Input:
        image --       filename of input image as string
        focalLength -- focal length in pixel as int
                       see assignment description on how to find the focal
                       length
        k1, k2 --      Radial distortion parameters
    Output:
        dstImage --    output image in a numpy array with
                       values in [0, 255]. The dimensions are (rows, cols,
                       color bands BGR).
    '''

    # compute spherical warp
    # compute the addresses of each pixel of the output image in the
    # source image
    uv = computeSphericalWarpMappings(np.array(image.shape), focalLength, k1, k2)

    # warp image based on backwards coordinates
    return warpLocal(image, uv)


