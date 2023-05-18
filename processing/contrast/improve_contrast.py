import numpy as np
import cv2
from skimage import morphology
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Contrast Enhancement Demo',
                        description='This program runs Contrast Limited Adaptive Histogram Equalization and tophat transforms to improve the contrast and sharpness of the input image',
                        epilog='Distributed as a part of ARCADE 2023 application')

    parser.add_argument('--source', '-s', default="../samples/sample1.png")
    parser.add_argument('--output', '-o', default="out_contrast.png")
    parser.add_argument('--clip_limit', '-cl', default=2.0)
    parser.add_argument('--grid_size', '-gs', default=8)
    args = parser.parse_args()


    img = cv2.imread(args.source, cv2.IMREAD_GRAYSCALE)

    img = np.squeeze(img) 
    clahe = cv2.createCLAHE(clipLimit =float(args.clip_limit), tileGridSize=(int(args.grid_size),int(args.grid_size)))  
    img_not = cv2.bitwise_not(img)  
    se = np.ones((50,50), np.uint8)  
    wth = morphology.white_tophat(img_not, se)  
    raw_minus_topwhite = img.astype(int) - wth 
    raw_minus_topwhite = ((raw_minus_topwhite>0)*raw_minus_topwhite).astype(np.uint8)  
    img = clahe.apply(raw_minus_topwhite) 
    img = img[:,:,np.newaxis] 

    cv2.imwrite(args.output, img)
