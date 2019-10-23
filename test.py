import argparse
import os
import numpy as np 
import tqdm
import torch


from PIL import Image
from dataloaders import make_data_loader
from modeling.deeplab import *
from dataloaders.utils import get_pascal_labels
from utils.metrics import Evaluator

class Tester(object):
    def __init__(self, args):
        if not os.path.isfile(args.model):
            raise RuntimeError("no checkpoint found at '{}'".fromat(args.model))
        self.args = args
        self.color_map = get_pascal_labels()
        self.test_loader, self.ids, self.nclass = make_data_loader(args)

        #Define model
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=False,
                        freeze_bn=False)
        
        self.model = model
        device = torch.device('cpu')
        checkpoint = torch.load(args.model, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.evaluator = Evaluator(self.nclass)

    def save_image(self, array, id, op):
        text = 'gt'
        if op == 0:
            text = 'pred'
        file_name = str(id)+'_'+text+'.png'
        r = array.copy()
        g = array.copy()
        b = array.copy()

        for i in range(self.nclass):
            r[array == i] = self.color_map[i][0]
            g[array == i] = self.color_map[i][1]
            b[array == i] = self.color_map[i][2]
    
        rgb = np.dstack((r, g, b))

        save_img = Image.fromarray(rgb.astype('uint8'))
        save_img.save(self.args.save_path+os.sep+file_name)


    def test(self):
        self.model.eval()
        self.evaluator.reset()
        # tbar = tqdm(self.test_loader, desc='\r')
        for i, sample in enumerate(self.test_loader):
            image, target = sample['image'], sample['label']
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.save_image(pred[0], self.ids[i], 0)
            self.save_image(target[0], self.ids[i], 1)
            self.evaluator.add_batch(target, pred)
    
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        print('Acc:{}, Acc_class:{}'.format(Acc, Acc_class))

def main():
    parser = argparse.ArgumentParser(description='Pytorch DeeplabV3Plus Test your data')
    parser.add_argument('--test', action='store_true', default=True, 
                        help='test your data')
    parser.add_argument('--dataset', default='pascal', 
                        help='datset format')
    parser.add_argument('--backbone', default='xception', 
                        help='what is your network backbone')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='output stride')
    parser.add_argument('--crop_size', type=int, default=513,
                        help='image size')
    parser.add_argument('--model', type=str, default='',
                        help='load your model')
    parser.add_argument('--save_path', type=str, default='',
                        help='save your prediction data')

    args = parser.parse_args()
    
    if args.test:
        tester = Tester(args)
        tester.test()

if __name__ == "__main__":
    main()