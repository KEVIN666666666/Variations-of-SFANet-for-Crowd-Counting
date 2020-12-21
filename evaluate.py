import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
from torchvision import transforms
from models import M_SFANet
import time
import torch
import cv2


mat_path = "../shanghaiTech_Test_Data/SH_partB_Density_map/test/gtdens/"
image_path = "../shanghaiTech_Test_Data/SH_partB_Density_map/test/images/"

image_names = sorted(os.listdir(image_path))
mat_names = sorted(os.listdir(mat_path))
print(len(image_names), len(mat_names))


print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

device = torch.device('cuda')
save_path = '../ShanghaitechWeights/checkpoint_best_MSFANet_B.pth' # Path to the trained weights.
model = M_SFANet.Model().to(device) # Build the model.
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model'])
model.eval()
tfms = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

MAE = []
start = time.time()

for index in range(len(image_names)):
    image_name = image_names[index]
    mat_name = mat_names[index]
    
    mat = loadmat(mat_path + mat_name)
    ground_truth = int(mat['all_num'].flatten().tolist()[0])
    
    img = Image.open(image_path + image_name).convert('RGB')
    height, width = img.size[1], img.size[0]
    height = round(height / 16) * 16
    width = round(width / 16) * 16
    img = cv2.resize(np.array(img), (width,height), cv2.INTER_CUBIC)
    img = tfms(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img)
    prediction = int(torch.sum(output[0]).item())
        
    print(image_name, ground_truth, prediction)

    MAE.append(abs(ground_truth - prediction))

end = time.time()

print(sum(MAE) / len(MAE))
print((end - start) / len(MAE))

