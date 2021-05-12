from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm.auto import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import numpy as np
import random
import cv2
from PIL import Image

from tqdm.auto import tqdm
import os 

def loadImages(path):
    def rgb2gray(rgb):
    
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
        return gray
    
    # image
    frame_array = []
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort()
    
    for i in range(len(files)):
        filename = path + files[i]
        
        img = Image.open(filename)
        if img.size == (4032, 3024):
            img = img.resize((1008, 756))
            
        #reading each files
        img = np.array(img)
        img = rgb2gray(img)
        
        # rescale
        img = img / 255
    
        #inserting the frames into an image array
        frame_array.append(img)
    
    return frame_array

def convolution(img: np.array, weights: np.array):
    """
    Convolution with weights filter

    Arguments
    ---------
    - img      : image
    - weights  : filter weights

    Return
    ------
    - filtered : filtered image

    """
    weights = weights.astype(float)
    filtered = np.zeros_like(img)

    width = int((weights.shape[1]-1)/2)
    height = int((weights.shape[0]-1)/2)

    for i in range(height,img.shape[1]-height):
        for j in range(width,img.shape[0]-width):
            filtered[j,i]=np.sum(weights*img[j-width:j+width+1,i-height:i+height+1])

    return filtered

def getFeaturePoints_(image: np.array, winSize: int = 7, threshold: int = 0):
    """
    Detect feature points in an image, using the Harris corner detector
    
    Arguments
    ---------
    - image     : image # we are only doing grayscale images,
    - winSize   : total size of window for summation in pixels,
    - threshold : threshold for best corners 
    
    Return
    ------
    - corners   : detected corners array by Herris
    """
    
    # sobel filter by axis
    filter_sobelx = np.array([
        (-1,0,1),
        (-2,0,2),
        (-1,0,1)
    ])
    filter_sobely = np.array([
        (-1,-2,-1),
        (0,0,0),
        (1,2,1)
    ])
    
    # convolution
    imgdx = convolution(image, 1/8*filter_sobelx)
    imgdy = convolution(image, 1/8*filter_sobely)
    
    # define array
    corner = np.zeros_like(image)
    
    # Ixx, Iyy, Ixy
    ix2 = imgdx*imgdx
    iy2 = imgdy*imgdy
    ixy = imgdx*imgdy
    
    # calculate Herris by window
    for y in range(0+winSize, image.shape[0]-1-winSize):
        for x in range(0+winSize, image.shape[1]-1-winSize):
            sx2 = np.sum(ix2[y-winSize:y+winSize, x-winSize:x+winSize])
            sy2 = np.sum(iy2[y-winSize:y+winSize, x-winSize:x+winSize])
            sxy = np.sum(ixy[y-winSize:y+winSize, x-winSize:x+winSize])
            
            # H
            tmpH = np.array([(sx2,sxy), (sxy,sy2)])

            # Herris
            ttmp = np.trace(tmpH)
            if ttmp != 0:
                corner[y,x] = np.linalg.det(tmpH)/ttmp
                
    # threshold
    best_corner = np.argwhere((corner > threshold) == True)
            
    return best_corner  

def getFeaturePoints(images, winSize: int = 7, threshold: int = 0, workers=None):
    """
    Detect feature points in an image, using the Harris corner detector
    
    Arguments
    ---------
    - images    : images # we are only doing grayscale images,
    - winSize   : total size of window for summation in pixels,
    - threshold : threshold for best corners 
    
    Return
    ------
    - corners   : detected corners array by Herris
    """
    with Pool(workers or cpu_count()) as pool:
        result = list(tqdm(pool.imap(
            func=partial(getFeaturePoints_, winSize=winSize, threshold=threshold),
            iterable=images
        ), total=len(images)))
        pool.close()
        pool.join()
        return result


def getFeatureDesriptors_(points_images, winSize, num_bins=8):
    """
    For a gicen input list of feature points, extract its neighborhood using the pixel values 
    in a small window around each point as a feature vector
    
    Return
    ------
    feature_vectors
    """
    
    points, image = points_images
    
    assert winSize > 4 # 4 is sub-patch size
    
    # Histogram of Gradients
    def HOG_cell_histogram(cell_direction, cell_magnitude, hist_bins):
        HOG_cell_hist = np.zeros(shape=(hist_bins.size))
        cell_size = cell_direction.shape[0]

        for row_idx in range(cell_size):
            for col_idx in range(cell_size):
                curr_direction = cell_direction[row_idx, col_idx]
                curr_magnitude = cell_magnitude[row_idx, col_idx]

                diff = np.abs(curr_direction - hist_bins)

                if curr_direction < hist_bins[0]:
                    first_bin_idx = 0
                    second_bin_idx = hist_bins.size-1
                elif curr_direction > hist_bins[-1]:
                    first_bin_idx = hist_bins.size-1
                    second_bin_idx = 0
                else:
                    first_bin_idx = np.where(diff == np.min(diff))[0][0]
                    temp = hist_bins[[(first_bin_idx-1)%hist_bins.size, (first_bin_idx+1)%hist_bins.size]]
                    temp2 = np.abs(curr_direction - temp)
                    res = np.where(temp2 == np.min(temp2))[0][0]
                    if res == 0 and first_bin_idx != 0:
                        second_bin_idx = first_bin_idx-1
                    else:
                        second_bin_idx = first_bin_idx+1

                first_bin_value = hist_bins[first_bin_idx]
                second_bin_value = hist_bins[second_bin_idx]
                HOG_cell_hist[first_bin_idx] = HOG_cell_hist[first_bin_idx] + (np.abs(curr_direction - first_bin_value)/(360.0/hist_bins.size)) * curr_magnitude
                HOG_cell_hist[second_bin_idx] = HOG_cell_hist[second_bin_idx] + (np.abs(curr_direction - second_bin_value)/(360.0/hist_bins.size)) * curr_magnitude


        return HOG_cell_hist
    
    # define descriptor list
    descriptors = []
    
    # sobel filter by axis
    filter_sobelx = np.array([
        (-1,0,1),
        (-2,0,2),
        (-1,0,1)
    ])
    filter_sobely = np.array([
        (-1,-2,-1),
        (0,0,0),
        (1,2,1)
    ])

    
    num_rows, num_cols = image.shape

    # padding image due to corner points on first and last index of image 
    padding_img = np.zeros((num_rows+winSize, num_cols+winSize))
    halfSize1 = int(winSize/2)
    halfSize2 = winSize - halfSize1
    padding_img[halfSize1:-halfSize2, halfSize1:-halfSize2] = image
    
    # 
    descriptor = [] # TODO 8

    for fp in points:
        paths = np.zeros((winSize, winSize, 2))
        path_img = padding_img[fp[0]:fp[0]+winSize, fp[1]:fp[1]+winSize]
        dev_x = convolution(path_img, 1/8*filter_sobelx)
        dev_y = convolution(path_img, 1/8*filter_sobely)

        orientation = (np.rad2deg(np.arctan2(dev_y, dev_x)) + 360) % 360
        magnitude = np.sqrt(dev_x**2+dev_y**2)

        # histogram bin
        bins_per_degree = 360. / num_bins 
        hist_bins = np.array([bins_per_degree*(i+1) for i in range(num_bins)])

        # descriptor sub patch
        descriptor_4 = []
        
        for r in range(0,winSize,4):
            for c in range(0,winSize,4):
                HOG_cell_hist = HOG_cell_histogram(cell_direction=orientation[c:c+4, r:r+4], 
                                                   cell_magnitude=magnitude[c:c+4, r:r+4], 
                                                   hist_bins=hist_bins)

                descriptor_4.append(HOG_cell_hist)
        descriptor.append(descriptor_4)

    
    return descriptor


def getFeatureDesriptors(points_lst, images, winSize, num_bins=8, workers=None):
    """
    Detect feature points in an image, using the Harris corner detector
    
    Arguments
    ---------
    - images    : images # we are only doing grayscale images,
    - winSize   : total size of window for summation in pixels,
    - threshold : threshold for best corners 
    
    Return
    ------
    - corners   : detected corners array by Herris
    """
    points_images = [[points_lst[i], images[i]] for i in range(len(images))]
    with Pool(workers or cpu_count()) as pool:
        result = list(tqdm(pool.imap(
            func=partial(getFeatureDesriptors_, winSize=winSize, num_bins=num_bins),
            iterable=points_images
        ), total=len(points_images)))
        pool.close()
        pool.join()
        return result


def match2Images_(points_descriptors, threshold):
    """
    Given lists of feature points and feature descriptors from two images, match them
    
    Return
    ------
    matches_idx: the matches as a list of indices in the two images
    """
    points_lst, descriptor_lst = points_descriptors
    
    # normalized cross correlation
    desc1 = np.array(descriptor_lst[0])
    desc1 = desc1.reshape(desc1.shape[0], np.prod(desc1.shape[1:]))

    desc2 = np.array(descriptor_lst[1])
    desc2 = desc2.reshape(desc2.shape[0], np.prod(desc2.shape[1:]))

    corr_desc = np.corrcoef(desc1, desc2)

    # match points
    corr_desc = corr_desc - np.identity(corr_desc.shape[0])

    corr_desc = corr_desc[:desc1.shape[0],desc1.shape[0]:] 
 
    good_matches = np.argsort(corr_desc.reshape(-1))[::-1]
    good_matches_idx = np.unravel_index(good_matches, corr_desc.shape)
    good_matches_idx = [(good_matches_idx[0][i], good_matches_idx[1][i]) for i in range(len(good_matches))]
    
    # threshold and points mapping
    matches = np.zeros((threshold, 4))
    for idx, pair in enumerate(good_matches_idx[:threshold]):
        src_idx, trg_idx = pair
        matches[idx] = np.hstack([points_lst[0][src_idx], points_lst[1][trg_idx]])
    
    return matches


def match2Images(points_lst, descriptor_lst, threshold, workers=None):
    """
    Detect feature points in an image, using the Harris corner detector
    
    Arguments
    ---------
    - images    : images # we are only doing grayscale images,
    - winSize   : total size of window for summation in pixels,
    - threshold : threshold for best corners 
    
    Return
    ------
    - corners   : detected corners array by Herris
    """
    points_descriptors = [[points_lst[i-1:i+1], descriptor_lst[i-1:i+1]] for i in range(1, len(points_lst))]
    with Pool(workers or cpu_count()) as pool:
        result = list(tqdm(pool.imap(
            func=partial(match2Images_, threshold=threshold),
            iterable=points_descriptors
        ), total=len(points_descriptors)))
        pool.close()
        pool.join()
        return result

def refineMatches_(matches, iteration, threshold):
    """
    Given lists of feature points and the match list, implement RANSAC to estimate a homography mapping 
    one image onto the other (so, the full number of 8 parameters). In each iteration of RANSAC, simply 
    select 4 points at random, estimate the homography by finding the nullspace of the transform
    
    
    """
    def calculateHomography(correspondences):
        #loop through the matched points and create assemble matrix
        aList = []
        for corr in correspondences:
            p1 = np.matrix([corr.item(0), corr.item(1), 1])
            p2 = np.matrix([corr.item(2), corr.item(3), 1])

            a1 = [p1.item(0), p1.item(1), 1, 0, 0, 0,
                  -p2.item(0)*p1.item(0), -p2.item(0)*p1.item(1), -p2.item(0)]

            a2 = [0, 0, 0, p1.item(0), p1.item(1), 1,
                  -p2.item(1)*p1.item(0), -p2.item(1)*p1.item(1), -p2.item(1)]

            aList.append(a1)
            aList.append(a2)

        matrixA = np.matrix(aList)

        #svd composition
        u, s, v = np.linalg.svd(matrixA)

        #reshape the min singular value into a 3 by 3 matrix
        h = np.reshape(v[8], (3, 3))

        #normalize and now we have h
        h = (1/h.item(8)) * h

        return h

    def geometricDistance(correspondence, h):

        p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
        estimatep2 = np.dot(h, p1)
        estimatep2 = (1/(estimatep2.item(2)+np.finfo(np.float32).eps.item()))*estimatep2

        p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
        error = p2 - estimatep2

        return np.linalg.norm(error)


    def ransac(points, iteration, threshold):
        maxInliers = []
        finalH = None

        for i in range(iteration):
            #find 4 random points to calculate a homography
            points1 = points[random.randrange(0, len(points))]
            points2 = points[random.randrange(0, len(points))]
            randomFour = np.vstack((points1, points2))
            points3 = points[random.randrange(0, len(points))]
            randomFour = np.vstack((randomFour, points3))
            points4 = points[random.randrange(0, len(points))]
            randomFour = np.vstack((randomFour, points4))

            #call the homography function on those points
            h = calculateHomography(randomFour)
            inliers = []

            for i in range(len(points)):
                d = geometricDistance(points[i], h)
                if d < 5: # this is also threshold, but we define this value as 5
                    inliers.append(points[i])

            if len(inliers) > len(maxInliers):
                maxInliers = inliers
                finalH = h

            if len(maxInliers) > (len(points)*threshold):
                break

        return finalH, maxInliers
    
    # define matrix
    matches_mat = np.matrix(matches)

    # loop RANSAC until Homography matrix passes the criterion
    Done = False

    while Done == False:
        # we use only Homography matrix
        H, _ = ransac(points=matches_mat, iteration=iteration, threshold=threshold)

        D = H[0,0]*H[1,1]-H[0,1]*H[1,0]
        sx = np.sqrt(H[0,0]**2+H[1,0]**2)
        sy = np.sqrt(H[0,1]**2+H[1,1]**2)
        P = np.sqrt(H[2,0]**2+H[2,1]**2)

        # criterion for filtering false matches
        if D<=0 or sx<0.1 or sx>4 or sy<0.1 or sy>4:
            Done = False
        else:
            Done = True
    
    return H

def refineMatches(matches_lst, iteration, threshold, workers=None):
    """
    Detect feature points in an image, using the Harris corner detector
    
    Arguments
    ---------
    - images    : images # we are only doing grayscale images,
    - winSize   : total size of window for summation in pixels,
    - threshold : threshold for best corners 
    
    Return
    ------
    - corners   : detected corners array by Herris
    """
    with Pool(workers or cpu_count()) as pool:
        result = list(tqdm(pool.imap(
            func=partial(refineMatches_, iteration=iteration, threshold=threshold),
            iterable=matches_lst
        ), total=len(matches_lst)))
        pool.close()
        pool.join()
        return result


def warpImages(images, H):
    """
    Given a list of homographies and a list of n images, warp images 2-n onto the image space of image 1
    
    
    """
    trg_image, src_image = images[0], images[1]

    dst = cv2.warpPerspective(src_image.T, H, (trg_image.shape[0], trg_image.shape[1]+src_image.shape[1]))
    
    # make background
    result = np.zeros(dst.T.shape)
    overlap = dst.T!=0

    result[:,:trg_image.shape[1]] += trg_image
    result[overlap] = 0
    result += dst.T
    
    return result

