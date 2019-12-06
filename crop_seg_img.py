import argparse
import numpy as np
from utils.dataloader.seg_utils import get_pascal_labels
from PIL import Image
import os

def main(args):

    DATA_DIR = args.data
    REF_DIR = args.ref
    SAVE_PATH = args.save

    rgb_list = get_pascal_labels()

    for idx, gray_file in enumerate(os.listdir(REF_DIR)):
        if gray_file == '.DS_Store':
            continue
        img = Image.open(os.path.join(REF_DIR, gray_file))
        original = Image.open(os.path.join(DATA_DIR, gray_file)).convert('RGB')
        white = [255, 255, 255]
        array = np.array(img)
        original_array = np.array(original)
        t = np.reshape(array, (-1))
        unique, counts = np.unique(t, return_counts=True)
        d = dict(zip(unique, counts))
    
        original_array[np.where(array == 0)] = white 
        
        rgb_img = Image.fromarray(original_array.astype('uint8'))
        rgb_img.save(SAVE_PATH+os.sep+gray_file)
        print('cropping img to seg ... {}/{}'.format(idx+1, len(os.listdir(REF_DIR))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gray to rgb')
    parser.add_argument('--data', default='./data', help='gray')
    parser.add_argument('--ref', default='./result/inference_img_gray', help='gray')
    parser.add_argument('--save', default='./result/inference_img_crop', help='save crop')
    args = parser.parse_args()
    main(args)
