import os
import time
import torch
import cv2
import sys
import argparse
import numpy as np
from scipy.io import loadmat
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from models import M_SFANet


class M_SFANetModel(torch.nn.Module):
    def __init__(self, model_path='../ShanghaitechWeights/checkpoint_best_MSFANet_B.pth'):
        super(M_SFANetModel, self).__init__()
        
        self.device = torch.device('cuda')
        self.model = M_SFANet.Model().to(self.device) # Build the model.
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.tfms = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            
        self.return_heatmap = False
        
    def forward(self, img):
        """
        the format of input image
        img = Image.open(image_path).convert('RGB')
        """
        
        height, width = img.size[1], img.size[0]
        height = round(height / 16) * 16
        width = round(width / 16) * 16
        
        img = cv2.resize(np.array(img), (width,height), cv2.INTER_CUBIC)
        img = self.tfms(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img)
        
        prediction = round(torch.sum(output[0]).item())
        
        if not self.return_heatmap:
            return prediction
        else:
            return prediction, output[0]
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../ShanghaitechWeights/checkpoint_best_MSFANet_A.pth', help='model.pth path')
    parser.add_argument('--source', type=str, default='images/0.jfif', help='source image')
    parser.add_argument('--output', type=str, default='.', help='output folder')  # output folder
    parser.add_argument('--heatmap', action='store_true', help='save results to output folder') # --heatmap
    opt = parser.parse_args()
    print(opt)
    
    model = M_SFANetModel(opt.weights)
    img = Image.open(opt.source).convert('RGB')
    
    if opt.heatmap:
        model.return_heatmap = True
        prediction, pred_heat_map = model(img)
    else:
        prediction = model(img)
    
    with open(os.path.join(opt.output, 'result.txt'), 'w') as file:
        file.write(prediction)
        
    # TODO SAVE HEATMAP
