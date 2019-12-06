import argparse
import numpy as np
from PIL import Image
from utils.dataloader.seg_utils import get_pascal_labels
import os

def main(args):

    DATA_DIR = args.data
    SAVE_PATH = args.save

    rgb_list = get_pascal_labels()

    for idx, gray_file in enumerate(os.listdir(DATA_DIR)):
        if gray_file == '.DS_Store':
            continue
        img = Image.open(os.path.join(DATA_DIR, gray_file))
        array = np.array(img)
        t = np.reshape(array, (-1))
        unique, counts = np.unique(t, return_counts=True)
        d = dict(zip(unique, counts))
        r = array.copy()
        g = array.copy()
        b = array.copy()
        for gray_label, rgb in enumerate(rgb_list):
            r[array == gray_label] = rgb_list[gray_label][0]
            g[array == gray_label] = rgb_list[gray_label][1]
            b[array == gray_label] = rgb_list[gray_label][2]
        rgb_array = np.dstack((r, g, b))
        rgb_img = Image.fromarray(rgb_array.astype('uint8'))
        rgb_img.save(SAVE_PATH+os.sep+gray_file)
        print('convert gray to rgb ... {}/{}'.format(idx+1, len(os.listdir(DATA_DIR))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gray to rgb')
    parser.add_argument('--data', default='./result/inference_img_gray', help='gray')
    parser.add_argument('--save', default='./result/inference_img_rgb', help='save rgb')
    args = parser.parse_args()
    main(args)
