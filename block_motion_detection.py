import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
from skimage import io, draw

def block_motion_detection(prev_frame, next_frame, blocksize, maxoffset, errfct):
    """
    find movement of blocks from one frame to the next within a certain range

    returns a matrix containing as many fields as there are blocks in the image, with the movement vector for that block stored at the position of the block. For blocks where no correspondence was found, the value None is inserted
    """
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
            print("offset: " + str(i) + ", " + str(j))
            rolled = np.roll(next_frame, (i,j), axis=(0,1))

            res = errfct(prev_frame, rolled)
#            io.imshow(res)
#            io.show()

            for l in range(blocknum_y):
                for k in range(blocknum_x):
                    #mask everything but the block

                    #calculate sum of differences
                    block_res = np.sum(res[l*blocksize:l*blocksize+blocksize, k*blocksize:k*blocksize+blocksize])
#                    if j >= 0 and i >= 0 and j < 5 and i < 5:
#                        print("index: " + str(k) + ", " + str(l) + ", block result: " + str(block_res) + ", best result: " + str(results_array[l,k,0]))
#                        f, axarr = plt.subplots(1,3)
#                        axarr[0].imshow(prev_frame[l:l+blocksize, k:k+blocksize])
#                        axarr[1].imshow(rolled[l:l+blocksize, k:k+blocksize])
#                        axarr[2].imshow(res[l:l+blocksize, k:k+blocksize])
#                        plt.show()
                    if(block_res < results_array[l,k,0]):
                        results_array[l,k,:] = [block_res, j, i]

    #(re)check at offset 0, so that homogenous regions do not report an offset of the first comparison as best (optional, but makes sense logically, because there is no actual movement)
    res = errfct(prev_frame, next_frame)
    for l in range(blocknum_y):
        for k in range(blocknum_x):
            #mask everything but the block
#            f, axarr = plt.subplots(1,3)
#            axarr[0].imshow(prev_frame[l*blocksize:l*blocksize+blocksize, k*blocksize:k*blocksize+blocksize])
#            axarr[1].imshow(next_frame[l*blocksize:l*blocksize+blocksize, k*blocksize:k*blocksize+blocksize])
#            axarr[2].imshow(res[l*blocksize:l*blocksize+blocksize, k*blocksize:k*blocksize+blocksize])
#            plt.show()

            block_res = np.sum(res[l*blocksize:l*blocksize+blocksize, k*blocksize:k*blocksize+blocksize])
            if(block_res <= results_array[l,k,0]):
                results_array[l,k,:] = [block_res, 0, 0]
    #results can be saved for testing the code below
    np.save("results_array.npy", results_array)
#    results_array = np.load("results_array.npy")
    print(results_array)

    #normalize error for visualization
    show_res = results_array[:,:,0]
    show_res = show_res/np.max(show_res)

    results_array[:,:,0] = results_array[:,:,0]/np.max(results_array[:,:,0])
    print(results_array)

    #filter values higher than threshold
    #TODO find a good threshold
    threshold = 0.3
    not_found_indices = np.argwhere(results_array[:,:,0] > threshold)
    for i in not_found_indices:
        results_array[i[0], i[1], :] = [-1, 0, 0]

    #circles on the image denote blocks where no corresponding movement was found
    #lines represent the movement vector starting from the center of the block
    show_img = np.copy(prev_frame)
    for l in range(blocknum_y-1):
        for k in range(blocknum_x-1):
            orig_x = int(k*blocksize + blocksize/2)
            orig_y = int(l*blocksize + blocksize/2)
            if results_array[l,k,0] >= 0:
                rr, cc = draw.line(orig_y, orig_x, orig_y + int(results_array[l,k,2]), orig_x + int(results_array[l,k,1]))
                show_img[rr, cc] = 1
            else:
                rr, cc = draw.circle_perimeter(orig_y, orig_x, 2)
                show_img[rr, cc] = 1

    f, axarr = plt.subplots(1,4)
    axarr[0].imshow(prev_frame)
    axarr[1].imshow(next_frame)
    axarr[2].imshow(show_img)
    axarr[3].imshow(show_res)
    plt.show()

def abs_diff(val1, val2):
    """ calculate absolute differences of the values """
    return np.abs(val1 - val2)

def sq_diff(val1, val2):
    """ calculate squared differences of the values """
    return np.power(val1 - val2, 2)
