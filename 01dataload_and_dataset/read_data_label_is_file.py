from torch.utils.data import Dataset
from PIL import Image
import os
#file's content is label
class mydataset(Dataset):
    def __init__(self,image_dir,label_dir):

        self.label_dir = label_dir
        self.image_dir = image_dir
        self.img_path = os.listdir(image_dir)
        self.label_path = os.listdir(label_dir)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.image_dir,img_name)
        img = Image.open(img_item_path)

        label_name = self.label_path[idx]
        label_item_path = os.path.join(self.label_dir,label_name)
        with open(label_item_path,'r') as f:
            label = f.read()
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir = "练手数据集/train"
ant_img_dir = "ants_image"
ant_label_dir = "ants_label"

bees_img_dir = "bees_image"
bees_label_dir = "bees_label"


ants_dataset = mydataset(os.path.join(root_dir,ant_img_dir),os.path.join(root_dir,ant_label_dir))
bees_dataset = mydataset(os.path.join(root_dir,bees_img_dir),os.path.join(root_dir,bees_label_dir))

ants_dataset[0][0].show()
print(ants_dataset[0][1])
train_dataset = ants_dataset + bees_dataset




    