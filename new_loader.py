import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset

root= 'Malnet'
encoding_root= 'Malnet//train'

class ImageFolderDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths, self.labels = self._load_image_paths_and_labels()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def _load_image_paths_and_labels(self):
        image_paths = []
        labels=[]
        class_name= []
        for type in os.listdir(self.image_folder):
            class_name.append(type)

        print(len(class_name))

        for folder_name in os.listdir(self.image_folder): 
            malware_type_path = os.path.join(self.image_folder, folder_name)
            for malware_family in os.listdir(malware_type_path):
                malware_family_path = os.path.join(malware_type_path,malware_family)
                malware_label = folder_name
                for filename in os.listdir(malware_family_path):
                    if filename.endswith(".png"):  # Assuming images are in JPG format
                        image_path = os.path.join(malware_family_path, filename)
                        image_paths.append(image_path)
                        label_filename = malware_label
                        for index, string in enumerate(class_name):
                            if string == label_filename:
                                labels.append(index)

        return image_paths,labels


            


