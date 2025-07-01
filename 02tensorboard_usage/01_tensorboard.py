from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')
image_path = r"data\train\ants_image\Nepenthes_rafflesiana_ant.jpg"
image_pil=Image.open(image_path)
image_array = np.array(image_pil)
writer.add_image("test",image_array,3,dataformats='HWC') #title,inmage,step,dataformats

for i in range(100):
    writer.add_scalar('y=2x',2*i,i) #title,y,x



writer.close()

#uv run tensorboard --logdir=logs --port=6007