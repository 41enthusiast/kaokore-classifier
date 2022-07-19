import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

transform_train = transforms.Compose([

        # geometric transformation
        # transforms.RandomCrop(32, padding=4),

        # transforms.RandomHorizontalFlip(p=0.5),

        # kernel filters
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),

        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

ds_names = ['fst-kaokore-2-cb-100pct', 'fst-kaokore-cb-100pct']
dataset1 = ImageFolder('../../'+ds_names[0], transform=transform_train)
dataset2 = ImageFolder('../../'+ds_names[1], transform=transform_train)

ds1_sz = len(dataset1)
ds2_sz = len(dataset2)
probs = [0.1, 0.2, 0.5, 0.8, 0.9]
index = 0
bs = 4

concat_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
print(len(concat_dataset), ds1_sz, ds2_sz)
weighted_sampler = torch.utils.data.WeightedRandomSampler([probs[index],]*ds1_sz+[1-probs[index],]*ds2_sz, ds1_sz)
dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size= bs, sampler = weighted_sampler)

print(next(iter(dataloader)))