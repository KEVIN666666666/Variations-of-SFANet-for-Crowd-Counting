import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
from torchvision import transforms
from models import M_SFANet
import time
import torch
import cv2


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

    model = M_SFANetModel('../ShanghaitechWeights/checkpoint_best_MSFANet_A.pth')
    
    mat_path = "../shanghaiTech_Test_Data/SH_partA_Density_map/test/gtdens/"
    image_path = "../shanghaiTech_Test_Data/SH_partA_Density_map/test/images/"

    image_names = sorted(os.listdir(image_path))
    mat_names = sorted(os.listdir(mat_path))
    
    MAE = []
    bias_rates = []
    start = time.time()

    for index in range(len(image_names)):
        image_name = image_names[index]
        mat_name = mat_names[index]
        
        mat = loadmat(mat_path + mat_name)
        ground_truth = int(mat['all_num'].flatten().tolist()[0])
        
        img = Image.open(image_path + image_name).convert('RGB')
        prediction = model(img) # inference
        
        MAE.append(abs(ground_truth - prediction))
        bias_rates.append(abs(ground_truth - prediction) / ground_truth)

        print(image_name, ground_truth, prediction, MAE[-1], bias_rates[-1])

    end = time.time()

    print("MAE: ", sum(MAE) / len(MAE))
    print("MBR: ", sum(bias_rates) / len(bias_rates))
    print((end - start) / len(MAE))


