import sys
import skimage as sk
from skimage import io, color

import block_motion_detection as bmd

#example call: python3.7 ./testframes/frame_02.jpg ./testframes/frame_03.jpg
if __name__ == "__main__":
    arguments = sys.argv[1:]

    motion_vectors = []
    frames = [ color.convert_colorspace(sk.img_as_float(io.imread(a)), "RGB", "YCbCr") for a in arguments ]

    #loops through the input images and sequentially calculates the motion between each pair 
    for i in range(len(frames)-1):
        vectors = bmd.block_motion_detection(frames[i], frames[i+1], 16, 15, bmd.sq_diff)
        motion_vectors.append(vectors)

