import torch

# 载入模型
path = r"D:/Vitis/USERS/10_Zedboard_audio_in/SD_read/tool/bnn_weights_binary_new.pt"
ckpt = torch.load(path, map_location="cpu")

# 如果文件是包含 'model' 键的字典
if isinstance(ckpt, dict) and "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

# 打印所有键名
print("All keys in checkpoint:")
for k in state_dict.keys():
    print(k)
