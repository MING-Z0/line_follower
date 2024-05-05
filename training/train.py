import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np


def get_x(path, width):
    """从图像文件名中获取 x 值"""
    return (float(int(path.split("_")[1])) * 224.0 / 640.0 - width / 2) / (width / 2)


def get_y(path, height):
    """从图像文件名中获取 y 值"""
    return ((float(int(path.split("_")[2]))) * 224.0 / 480.0 - height / 2) / (height / 2)


class XYDataset(torch.utils.data.Dataset):

    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = PIL.Image.open(image_path)
        x = float(get_x(os.path.basename(image_path), 224))
        y = float(get_y(os.path.basename(image_path), 224))

        if self.random_hflips:
            if float(np.random.rand(1)) > 0.5:
                image = transforms.functional.hflip(image)
                x = -x

        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy().copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image,
                                                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor([x, y]).float()


def main(args=None):
    # 需要根据自己的环境改为数据集存放位置
    dataset = XYDataset(r'.\data\image_train',
                        random_hflips=False)

    # 创建训练集和测试集
    test_percent = 0.1
    num_test = int(test_percent * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [len(dataset) - num_test, num_test])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    # 创建ResNet18模型，这里选用已经预训练的模型，
    # 更改fc输出为2，即x、y坐标值
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    NUM_EPOCHS = 100
    BEST_MODEL_PATH = r'.\training\model\best_model.pth'
    best_loss = 1e9

    optimizer = optim.Adam(model.parameters())

    for epoch in range(NUM_EPOCHS):

        model.train()
        train_loss = 0.0
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            train_loss += float(loss)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            test_loss += float(loss)
        test_loss /= len(test_loader)

        print('%f, %f' % (train_loss, test_loss))
        if test_loss < best_loss:
            print("save")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_loss


if __name__ == '__main__':
    main()