from torch.utils.data import Dataset
import os
import cv2
import json

def load_dataset(json_file, split="train"):
    with open(json_file, 'r') as f:
        dataset = json.load(f)
    
    data_list = dataset.get(split, [])

    data_list = [item for item in data_list if 'normal' not in item['image']]
    
    return data_list


def get_all_images(train_data):
    all_images = []
    all_masks = []

    all_images.extend([item['image'] for item in train_data])
    all_masks.extend([item['mask'] for item in train_data])

    return all_images, all_masks


class BUSI(Dataset):
    def __init__(
        self,
        base_dir = None,
        split = "train",
        transform = None,
    ):
        self.base_dir = base_dir
        self.data_list = []
        self.transform = transform

        self.data_list = load_dataset(os.path.join(base_dir, "dataset.json"), split=split)
        print(split + f" phase has {len(self.data_list)} samples.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        case = self.data_list[idx]
        img_file =  os.path.join(self.base_dir, case['image'])
        mask_file = os.path.join(self.base_dir, case['mask'])

        image = cv2.imread(img_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        mask = mask.astype('float32') / 255

        sample = {"image": image, "label": mask, "idx": idx, "case_name": case['image'], "mask_name": case['mask']}
        return sample


