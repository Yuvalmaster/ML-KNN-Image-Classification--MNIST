import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================================================== MODEL ===================================================== #
# Quick summary: The data is split to 70% Train dataset, and 30% Test dataset. 
# The model is fitted to the train dataset, meaning, the algorithm knows the labels of the images of the 
# train dataset. When trying to predict the test images' labels, the model goes through each image, calculate
# The euclidean distance of the image compare to all the Train dataset, and chooses the k best closest 
# (=smallest distance). Than, the model chooses the most fitting label by counting appearnces of each label
# from the k chosen labels, choosing the most frequent.

class EuclideanKNN:
    def __init__(self, k):
        self.k = k

    # Fit the model to the train data set
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Send data to device
        X       = X.to(device)
        X_train = self.X_train.to(device)
        y_train = self.y_train.to(device)

        for x in tqdm(X):
            # Measures the euclidean distance of each image from test data set to the train data set
            distances = self.euclid(X_train, x)
            
            # Choose k lowest absolute distances and best labels
            _, nearest_indices = torch.topk(distances, k=self.k, largest=False)
            nearest_labels = y_train[nearest_indices]
            
            # Predict label based on best's most frequent
            unique_labels, label_counts = torch.unique(nearest_labels, return_counts=True)
            y_pred.append(unique_labels[label_counts.argmax()].item())
        
        return torch.tensor(y_pred)

    @staticmethod
    def euclid(x1, x2):
        return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))

# ================================================== LOAD DATA ================================================== #
def load_data(split_ratio, batch_size):
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('../Dataset/', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST('../Dataset/', train=False, transform=transform)

    # Concatenate train and test datasets
    X = torch.cat([train_dataset.data, test_dataset.data], dim=0).view(-1, 28 * 28).float()
    y = torch.cat([train_dataset.targets, test_dataset.targets], dim=0)

    # Split the complete dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

    # Move data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    train_loader = DataLoader(dataset=list(zip(X_train, y_train)),
                              batch_size=batch_size,
                              shuffle=True)
    
    test_loader  = DataLoader(dataset=list(zip(X_test, y_test)),
                              batch_size=batch_size,
                              shuffle=False)

    return train_loader, test_loader, X_train, y_train, X_test, y_test

# Load data with defined split ratio and batch size
batch_size = 64
train_loader, test_loader, X_train, y_train, X_test, y_test = load_data(split_ratio=0.3, batch_size=batch_size)

# Print datasets sizes
print(f'Dataset size:  {len(train_loader.dataset) + len(test_loader.dataset)}\n'
      f'Trainset size: {len(train_loader.dataset)}                           \n'
      f'Testset size:  {len(test_loader.dataset)}                            \n')


# ==================================================== TRAIN ==================================================== #
max_k      = 19
test_range = range(1, max_k + 2, 2)
k_accuracy = []

for k in test_range:
    print(f'\033[1mK = {k}\033[0m')
    # Set the current model
    model = EuclideanKNN(k)
    model.train(X_train, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test.cpu().numpy(), y_pred.cpu().numpy())*100
    print(f"Accuracy: {accuracy: .3f}")
    k_accuracy.append(accuracy)
    
# Visualize accuracy vs. k
fig, ax = plt.subplots()
ax.plot(test_range, k_accuracy)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="K-NN Performance over K value")
plt.xticks(test_range)
plt.show()