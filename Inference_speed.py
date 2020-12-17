import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models import M_SFANet
import time
from tqdm import tqdm


print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())


device = torch.device('cuda')
save_path = 'checkpoint_best_MSFANet_B.pth' # Path to the trained weights.
model = M_SFANet.Model().to(device) # Build the model.
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model'])


res = [(320, 240),
(640, 480),
(1280, 960),
(1600, 1200)]


for w, h in res:
    input = torch.randn(10, 3, w, h).to(device)
    inference_time = 0
    model.eval()
    for i in tqdm(range(100)):
        start_time = time.time()
        with torch.no_grad():
          output = model(input[i%10].unsqueeze_(0))
        inference_time += (time.time() - start_time)
        del output
        torch.cuda.empty_cache()
    del input
    torch.cuda.empty_cache()
    print(w, h)
    print(inference_time/100)
    print('======')
