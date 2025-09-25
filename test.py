from tqdm import tqdm
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.CRITICAL)
from utils import *
from nets.GDFusion import GaborFusion
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''Select the desired model'''
path_model=r"exp/FMB/model/ckpt_40.pth"
path_img1=r"data/FMB_train/ir"
path_img2=r"data/FMB_train/vi"
path_result=r"test_results/FMB_train"

Fusion_model= GaborFusion().to(device)
Fusion_model.load_state_dict(torch.load(path_model, map_location=device))
Fusion_model.eval()
window_size = 8  

batch_size = 1
img_names = os.listdir(path_img1)
num_imgs = len(img_names)

with torch.no_grad():
    for i in tqdm(range(0, num_imgs, batch_size)):
        batch_imgnames = img_names[i:i+batch_size]
        img1_list, img2_list, CrCb_list, name_list = [], [], [], []
        for imgname in batch_imgnames:
            img1 = image_read(os.path.join(path_img1, imgname))
            img2 = image_read(os.path.join(path_img2, imgname))
            # 检查尺寸一致
            if img1.shape != img2.shape:
                print(f"[跳过] {imgname} ir/vi shape 不一致: {img1.shape} vs {img2.shape}")
                continue
            img1_YCrCb = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
            img2_YCrCb = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
            if is_grayscale(img1):
                if is_grayscale(img2):
                    CrCb = None
                else:
                    CrCb = img2_YCrCb[:,:,1:]
            else:
                if is_grayscale(img2): 
                    CrCb = img1_YCrCb[:,:,1:]
                else:
                    CrCb = fuse_CrCb(img1_YCrCb[:,:,1:], img2_YCrCb[:,:,1:])
            img1_tensor = torch.from_numpy(img1_YCrCb[:,:,0][np.newaxis, ...] / 255).float()
            img2_tensor = torch.from_numpy(img2_YCrCb[:,:,0][np.newaxis, ...] / 255).float()
            img1_list.append(img1_tensor)
            img2_list.append(img2_tensor)
            CrCb_list.append(CrCb)
            name_list.append(imgname)
        if not img1_list:
            continue
        # 拼成batch
        img1_batch = torch.stack(img1_list).to(device)  # (B, 1, H, W)
        img2_batch = torch.stack(img2_list).to(device)
        data_Fuse = Fusion_model(torch.cat((img1_batch, img2_batch), 1))
        data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
        for j in range(data_Fuse.shape[0]):
            fused_image = np.squeeze((data_Fuse[j] * 255).detach().cpu().numpy())
            fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
            img_save(fused_image, name_list[j].split(sep='.')[0], path_result, CrCb_list[j])