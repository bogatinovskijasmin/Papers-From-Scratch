from imports import *



class LandscapeData(torch.utils.data.Dataset):
    def __init__(self, dataset_path, image_size, batch_size=64, size=80):
        super().__init__()
        self.batch_size = batch_size
        transformations = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size), 
            torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transformations)
        

    def create_tensor_loader(self, dataset, train):        
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=train)

    def get_train_dataloader(self, ):
        return self.create_tensor_loader(self.dataset, True)

    def get_test_dataloader(self):
        return self.create_tensor_loader(self.dataset, False)
    



class FashionMnist(torch.utils.data.Dataset):
    def __init__(self, batch_size=256, size=(28, 28)):
        super().__init__()
        self.batch_size = batch_size
        transformations = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size), 
            torchvision.transforms.ToTensor()
        ])
        self.train_data = torchvision.datasets.FashionMNIST(root="./FashionMNIST/", train=True, transform=transformations, download=True)
        self.test_data = torchvision.datasets.FashionMNIST(root="./FashionMNIST/", train=False, transform=transformations, download=True)

    def create_tensor_loader(self, dataset, train, indecies=slice(0, None)):
        tensors = tuple(a[indecies] for a in dataset)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=train)

    def get_train_dataloader(self, ):
        return self.create_tensor_loader([self.train_data.train_data, self.train_data.targets], True)

    def get_test_dataloader(self):
        return self.create_tensor_loader([self.test_data.test_data, self.test_data.targets], False)
    