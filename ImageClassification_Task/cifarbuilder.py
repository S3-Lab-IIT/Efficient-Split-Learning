import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

import numpy as np
import pandas as pd

def setting2_dirch_val(train_full_dataset, test_full_dataset, num_users):
    np.random.seed(42)  # Set the seed for reproducibility
    dict_users, dict_users_test, dict_users_val = {}, {}, {}
    for i in range(num_users):
        dict_users[i] = []
        dict_users_test[i] = []
        dict_users_val[i] = []

    # Create DataFrames
    df = pd.DataFrame(list(zip(train_full_dataset.data, train_full_dataset.targets)), columns=['images', 'labels'])
    df_test = pd.DataFrame(list(zip(test_full_dataset.data, test_full_dataset.targets)), columns=['images', 'labels'])
    num_of_classes = df['labels'].nunique()

    # Initialize class-wise indices
    dict_classwise = {i: df[df['labels'] == i].index.values.astype(int) for i in range(num_of_classes)}
    dict_classwise_test = {i: df_test[df_test['labels'] == i].index.values.astype(int) for i in range(num_of_classes)}

    # Sample sizes per client
    total_train_samples_per_client = 500
    total_test_samples_per_client = 1000
    total_val_samples_per_client = 250

    for i in range(num_users):
        dirichlet_dist = np.random.dirichlet(np.ones(num_of_classes)*0.9)
        num_samples_train = np.round(dirichlet_dist * total_train_samples_per_client).astype(int)
        num_samples_test = np.round(dirichlet_dist * total_test_samples_per_client).astype(int)
        num_samples_val = np.round(dirichlet_dist * total_val_samples_per_client).astype(int)

        for j in range(num_of_classes):
            # Sampling training data
            train_indices = sample_data(dict_classwise, j, num_samples_train[j])
            dict_users[i].extend(train_indices)
            dict_classwise[j] = list(set(dict_classwise[j]) - set(train_indices))

            # Sampling test data
            test_indices = sample_data(dict_classwise_test, j, num_samples_test[j], replace=True)
            dict_users_test[i].extend(test_indices)

            # Sampling validation data
            val_indices = sample_data(dict_classwise, j, num_samples_val[j])
            dict_users_val[i].extend(val_indices)
            dict_classwise[j] = list(set(dict_classwise[j]) - set(val_indices))

    return dict_users, dict_users_test, dict_users_val



def setting2_dirch_val2(train_full_dataset, test_full_dataset, num_users):
    np.random.seed(42)  # Set the seed for reproducibility
    dict_users, dict_users_test, dict_users_val = {}, {}, {}
    for i in range(num_users):
        dict_users[i] = []
        dict_users_test[i] = []
        dict_users_val[i] = []

    # Create DataFrames
    df = pd.DataFrame(list(zip(train_full_dataset.data, train_full_dataset.targets)), columns=['images', 'labels'])
    df_test = pd.DataFrame(list(zip(test_full_dataset.data, test_full_dataset.targets)), columns=['images', 'labels'])
    num_of_classes = df['labels'].nunique()

    # Initialize class-wise indices
    dict_classwise = {i: df[df['labels'] == i].index.values.astype(int) for i in range(num_of_classes)}
    dict_classwise_test = {i: df_test[df_test['labels'] == i].index.values.astype(int) for i in range(num_of_classes)}

    # Sample sizes per client
    total_train_samples_per_client = 500
    total_test_samples_per_client = 1000
    total_val_samples_per_client = 250

    for i in range(num_users):
        dirichlet_dist = np.random.dirichlet(np.ones(num_of_classes))
        num_samples_train = np.round(dirichlet_dist * total_train_samples_per_client).astype(int)
        num_samples_test = np.round(dirichlet_dist * total_test_samples_per_client).astype(int)
        num_samples_val = np.round(dirichlet_dist * total_val_samples_per_client).astype(int)

        for j in range(num_of_classes):
            # Sampling training data
            train_indices = sample_data(dict_classwise, j, num_samples_train[j])
            dict_users[i].extend(train_indices)
            dict_classwise[j] = list(set(dict_classwise[j]) - set(train_indices))

            # Sampling test data
            test_indices = sample_data(dict_classwise_test, j, num_samples_test[j], replace=True)
            dict_users_test[i].extend(test_indices)

            # Sampling validation data
            val_indices = sample_data(dict_classwise, j, num_samples_val[j])
            dict_users_val[i].extend(val_indices)
            dict_classwise[j] = list(set(dict_classwise[j]) - set(val_indices))

    return dict_users, dict_users_test, dict_users_val





def sample_data(data_dict, class_index, num_samples, replace=False):
    population_size = len(data_dict[class_index])
    if num_samples > population_size:
        num_samples = population_size
    return list(np.random.choice(data_dict[class_index], num_samples, replace=replace))

'''
def setting2_dirch_val(train_full_dataset, test_full_dataset, num_users):
    np.random.seed(42)  # Set the seed for reproducibility
    dict_users, dict_users_test, dict_users_val = {}, {}, {}
    for i in range(num_users):
        dict_users[i] = []
        dict_users_test[i] = []
        dict_users_val[i] = []

    df = pd.DataFrame(list(zip(train_full_dataset.data, train_full_dataset.targets)), columns=['images', 'labels'])
    df_test = pd.DataFrame(list(zip(test_full_dataset.data, test_full_dataset.targets)), columns=['images', 'labels'])
    num_of_classes = len(df['labels'].unique())

    dict_classwise = {}
    dict_classwise_test = {}

    total_train_samples_per_client = 500
    total_test_samples_per_client = 1000
    total_val_samples_per_client = 250

    for i in range(num_of_classes):
        dict_classwise[i] = df[df['labels'] == i].index.values.astype(int)

    for i in range(num_of_classes):
        dict_classwise_test[i] = df_test[df_test['labels'] == i].index.values.astype(int)

    for i in range(num_users):
        dirichlet_dist = np.random.dirichlet(np.ones(num_of_classes))
        #print(f'for {i} dirchlet dist {dirichlet_dist}')
        num_samples_train = np.round(dirichlet_dist * total_train_samples_per_client).astype(int)
        num_samples_test = np.round(dirichlet_dist * total_test_samples_per_client).astype(int)
        num_samples_val = np.round(dirichlet_dist * total_val_samples_per_client).astype(int)

        for j in range(num_of_classes):
            population_size_train = len(dict_classwise[j])
            population_size_test = len(dict_classwise_test[j])

            if num_samples_train[j] > population_size_train:
                num_samples_train[j] = population_size_train
            if num_samples_test[j] > population_size_test:
                num_samples_test[j] = population_size_test

            temp = list(np.random.choice(dict_classwise[j], num_samples_train[j], replace=False))
            dict_users[i].extend(temp)
            dict_classwise[j] = list(set(dict_classwise[j]) - set(temp))

            temp_test = list(np.random.choice(dict_classwise_test[j], num_samples_test[j], replace=True))
            dict_users_test[i].extend(temp_test)
            dict_classwise_test[j] = list(set(dict_classwise_test[j]) - set(temp_test))

            population_size_val = len(dict_classwise[j])
            if num_samples_val[j] > population_size_val:
                num_samples_val[j] = population_size_val

            temp_val = list(np.random.choice(dict_classwise[j], num_samples_val[j], replace=False))
            dict_users_val[i].extend(temp_val)
            dict_classwise[j] = list(set(dict_classwise[j]) - set(temp_val))

    return dict_users, dict_users_test, dict_users_val'''

class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, client_id, dataset_type, tfms):
        self.images = images
        self.labels = labels
        self.client_id = client_id
        self.dataset_type = dataset_type  # 'train=0, 'val=1', or 'test=2'
        self.tfms = tfms

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        unique_id = f"{self.client_id}-{self.dataset_type}-{idx}"  # Append dataset_type to the unique ID

        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)
        if self.tfms:
            image = self.tfms(image)
        label = torch.tensor(label).long()
        return {
            'image': image,
            'label': label,
            'id': unique_id
        }

    def __len__(self):
        return len(self.images)  # Return the length of the images (number of samples)


class CIFAR10DataBuilder:
    def __init__(self, img_size=32, num_clients=10):
        self.img_size = img_size
        self.num_clients = num_clients

    def download_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        return train_dataset, test_dataset

    def get_default_transforms(self):
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        
        transform_test =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
        
        return transform_train, transform_test

    def get_datasets(self, client_id, transform_train=None, transform_test=None, pool=False):
        train_dataset, test_dataset = self.download_data()

        if transform_train is None:
            transform_train, transform_test = self.get_default_transforms()

        dict_users, dict_users_test, dict_users_val = setting2_dirch_val2(train_dataset, test_dataset, self.num_clients)

        train_indices = dict_users[client_id]
        val_indices = dict_users_val[client_id]
        test_indices = dict_users_test[client_id]

        train_images = train_dataset.data[train_indices]
        train_labels = np.array(train_dataset.targets)[train_indices]

        val_images = train_dataset.data[val_indices]
        val_labels = np.array(train_dataset.targets)[val_indices]

        test_images = test_dataset.data[test_indices]
        test_labels = np.array(test_dataset.targets)[test_indices]
        print("Training")
        train_ds = CIFAR10Dataset(train_images, train_labels, client_id, 0, transform_train)
        print("Validation")
        val_ds = CIFAR10Dataset(val_images, val_labels, client_id, 1,  transform_test)
        print("Testing")
        test_ds = CIFAR10Dataset(test_images, test_labels, client_id, 2, transform_test)

        return train_ds, val_ds, test_ds

if __name__ == '__main__':
    cifar_builder = CIFAR10DataBuilder()
    client_id = 0
    train_ds, val_ds, test_ds = cifar_builder.get_datasets(client_id)
    
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Example of fetching a batch
    item = next(iter(train_loader))
    im, lb = item['image'], item['label']
    print(f"Batch image shape: {im.shape}, Batch label shape: {lb.shape}")
