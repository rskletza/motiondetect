import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import time
from skimage import io, draw

def block_motion_detection(prev_frame, next_frame, errfct, blocksize=16, maxoffset=15):
    """
    find movement of blocks from one frame to the next within a certain range

    returns a matrix containing as many fields as there are blocks in the image, with the movement vector for that block stored at the position of the block. For blocks where no correspondence was found, the value None is inserted
    """
    start = time.time()
    #use only chrominance channel
    prev_frame = prev_frame[:,:,0]
    next_frame = next_frame[:,:,0]
    print(prev_frame.shape)
    blocknum_x = int(prev_frame.shape[1] / blocksize)
    blocknum_y = int(prev_frame.shape[0] / blocksize)

    #crop image to fit multiple of blocksize TODO handle egdes
    prev_frame = prev_frame[0:blocknum_y * blocksize, 0:blocknum_x * blocksize]
    next_frame = next_frame[0:blocknum_y * blocksize, 0:blocknum_x * blocksize]

    #create the result array, the first channel is the error, the second will be the x displacement and the third the y displacement for each block in "prev_frame"
    results_array = np.dstack((np.full((blocknum_y, blocknum_x), np.inf), np.zeros((blocknum_y, blocknum_x, 2))))

    for j in range(-maxoffset, maxoffset+1):
        for i in range(-maxoffset, maxoffset+1):
            rolled = np.roll(next_frame, (i,j), axis=(0,1))
            res = errfct(prev_frame, rolled)

            #calculate sum of differences for each block and update result array given a better result
            for l in range(blocknum_y):
                for k in range(blocknum_x):
                    block_res = np.sum(res[l*blocksize:l*blocksize+blocksize, k*blocksize:k*blocksize+blocksize])
                    if(block_res < results_array[l,k,0]):
                        results_array[l,k,:] = [block_res, j, i]

    #(re)check at offset 0, so that homogenous regions do not report an offset of the first comparison as best (optional, but makes sense logically, because there is no actual movement)
    res = errfct(prev_frame, next_frame)
    for l in range(blocknum_y):
        for k in range(blocknum_x):
            block_res = np.sum(res[l*blocksize:l*blocksize+blocksize, k*blocksize:k*blocksize+blocksize])
            if(block_res <= results_array[l,k,0]):
                results_array[l,k,:] = [block_res, 0, 0]

    #normalize error for visualization
    show_res = results_array[:,:,0]
    show_res = show_res/np.max(show_res)

    results_array[:,:,0] = show_res

    #filter values higher than threshold
    threshold = 1
    not_found_indices = np.argwhere(results_array[:,:,0] > threshold)
    for i in not_found_indices:
        results_array[i[0], i[1], :] = [-1, 0, 0]

    end = time.time()
    print(end - start)

    ################ VISUALIZATION ################
    #circles on the image denote blocks where no corresponding movement was found
    #lines represent the movement vector starting from the center of the block
    show_img = np.copy(prev_frame)
    for l in range(blocknum_y):
        for k in range(blocknum_x):
            orig_x = int(k*blocksize + blocksize/2)
            orig_y = int(l*blocksize + blocksize/2)
            if results_array[l,k,0] >= 0:
                rr, cc = draw.line(orig_y, orig_x, orig_y + int(results_array[l,k,2]), orig_x + int(results_array[l,k,1]))
                try:
                    show_img[rr, cc] = 1
                except IndexError:
                    pass
            else:
                rr, cc = draw.circle_perimeter(orig_y, orig_x, 2)
                show_img[rr, cc] = 1

#    f, axarr = plt.subplots(2,2)
#    axarr[0, 0].set_title('première trame', fontsize=22)
#    axarr[0, 1].set_title('deuxième trame', fontsize=22)
#    axarr[1, 0].set_title('vecteurs de mouvement', fontsize=22)
#    axarr[1, 1].set_title('erreurs normalisées par bloc', fontsize=22)
#    axarr[0, 0].imshow(prev_frame)
#    axarr[0, 1].imshow(next_frame)
#    axarr[1, 0].imshow(show_img)
#    axarr[1, 1].imshow(show_res)
#    plt.show()

    f, axarr = plt.subplots(1,4)
    axarr[0].set_title('première trame', fontsize=22)
    axarr[1].set_title('deuxième trame', fontsize=22)
    axarr[2].set_title('vecteurs de mouvement', fontsize=22)
    axarr[3].set_title('erreurs normalisées par bloc', fontsize=22)
    axarr[0].imshow(prev_frame)
    axarr[1].imshow(next_frame)
    axarr[2].imshow(show_img)
    axarr[3].imshow(show_res)
    plt.show()

    return results_array

def abs_diff(val1, val2):
    """ calculate absolute differences of the values """
    return np.abs(val1 - val2)

def sq_diff(val1, val2):
    """ calculate squared differences of the values """
    return np.power(val1 - val2, 2)
