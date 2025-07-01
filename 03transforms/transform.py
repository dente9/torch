from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import cv2

#why we need tensor,for convenience

img_path = "../dataset/train/ants/0013035.jpg"
img=Image.open(img_path)


writer = SummaryWriter = SummaryWriter("logs")

#how to use transform
tensor_trans=transforms.ToTensor()
tensor_img = tensor_trans(img )

writer.add_image("tensor_img",tensor_img)
writer.close()

