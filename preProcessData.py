import numpy as np
import os
import cv2
from glob import glob

def remove_black(fileNameGT):
    '''
    Returns the ground truth maps for roadArea and the validArea 
    :param fileNameGT:
    '''
    # Read GT
    assert os.path.isfile(fileNameGT), 'Cannot find: %s' % fileNameGT
    full_gt = cv2.imread(fileNameGT, cv2.IMREAD_COLOR)
    new_gt = full_gt[:, :, 0]/255;
    #attention: OpenCV reads in as BGR, so first channel has Blue / road GT

    full_gt[:,:,2] = 255

    # cv2.imshow('new_gt', new_gt)
    # cv2.waitKey(1)

    save_name = fileNameGT.replace('trainBraw', 'trainB')

    cv2.imwrite(save_name, new_gt)
    print('saved ', save_name)



def main():
    gt_paths = glob('./datasets/cam2road/trainBraw/*')

    for path in gt_paths:
        remove_black(path)

if __name__ == "__main__":
    main()
