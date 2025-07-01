from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter(log_dir='./logs')
img = Image.open(r"../dataset/train/ants/0013035.jpg")


#totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('img', img_tensor )

#normalizef
trans_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_normalize = trans_normalize(img_tensor)
writer.add_image('img_normalize', img_normalize)
print(img_normalize)

#resize
trans_resize = transforms.Resize((224, 224))
img_resize = trans_resize(img)
img_resize =trans_totensor(img_resize)
writer.add_image('img_resize', img_resize)

#compose - resize -2
trans_resize_2 =transforms.Resize(224)
tran_compose = transforms.Compose([trans_resize_2, transforms.ToTensor()])
img_resize_2 = trans_totensor(img)
writer.add_image('img_resize', img_resize_2,1)

#randomcrop
trans_random = transforms.RandomCrop(30)
trans_compose_2 = transforms.Compose([trans_random, transforms.ToTensor()])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('img_crop', img_crop,i)
writer.close()

