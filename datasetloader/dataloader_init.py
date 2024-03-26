import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class Dataset_Mooc(Dataset):
    def __init__(self, Path, flag):
        if flag == "train":
            self.img_path = Path + "train/course_train.npy"
            self.target_path = Path + "train/target_train.npy"
        elif flag == "test":
            self.img_path = Path + "test/course_train.npy"
            self.target_path = Path + "test/target_train.npy"
        else:
            print("flag error!")
            exit(-1)
        self.tf_tensor = transforms.ToTensor()

        self.__read_data__()
        self.__pre_data__()

    def __read_data__(self):
        self.img = np.load(self.img_path)
        self.target = np.load(self.target_path)

    def __pre_data__(self):
        pass

    def __getitem__(self, index):
        return self.tf_tensor(self.img[index].reshape(1, 5, 7).astype(np.uint8)).view(1, 5, 7), self.target[index].astype(np.longlong)

    def __len__(self):
        return self.img.shape[0]