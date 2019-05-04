import numpy as np
import skimage as sk
from skimage import io, draw

def block_motion_detection(prev_frame, next_frame, blocksize, maxoffset):
    """
    find movement of blocks from one frame to the next within a certain range

    returns a matrix containing as many fields as there are blocks in the image, with the movement vector for that block stored at the position of the block. For blocks where no correspondence was found, the value None is inserted
    """
    #create results array and initialize
#    array_width = maxoffset * 2 + 1
#    result = np.ones((array_width, array_width))

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
    results_array = np.full((blocknum_y, blocknum_x, 3), 99999)
    print(results_array.shape)

    range_test = []
    for j in range(-maxoffset, maxoffset+1):
        for i in range(-maxoffset, maxoffset+1):
#            print(i,j)
            rolled = np.roll(next_frame, (i,j), axis=(0,1))

            #sum of absolute differences
            res = np.subtract(prev_frame, rolled)
            res = np.absolute(res)
            for l in range(blocknum_y):
                for k in range(blocknum_x):
                    #mask everything but the block
#                    io.imshow(prev_frame[l:l+blocksize, k:k+blocksize])
#                    io.show()
                    block_res = np.sum(res[l:l+blocksize, k:k+blocksize])
                    range_test.append(block_res)
                    print(block_res)
                    if(block_res < results_array[l,k,0]):
                        results_array[l,k,:] = [block_res, j, i]
    #results can be saved for testing the code below
#    np.save("results_array.npy", results_array)
#    results_array = np.load("results_array.npy")

    #filter values higher than threshold
    #TODO find a good threshold
    threshold = np.average(results_array[:,:,0]) + 100
#    range_test = results_array[:,:,0]
#    print(np.min(range_test), np.max(range_test), np.average(range_test), np.median(range_test))
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
                rr, cc = draw.line(orig_y, orig_x, orig_y + results_array[l,k,2], orig_x + results_array[l,k,1])
                show_img[rr, cc] = 1
            else:
                rr, cc = draw.circle_perimeter(orig_y, orig_x, 2)
                show_img[rr, cc] = 1
    io.imshow(show_img)
    io.show()
