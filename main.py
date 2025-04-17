import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import gym

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def preprocess_frame(frame):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((84, 84)),
        T.ToTensor()
    ])
    return transform(frame).unsqueeze(0)

# Initialize the environment
env = gym.make('ALE/Assault-v5')

# Initialize the model
model = Model()

# Example of running one step in the environment
state = env.reset()
state = preprocess_frame(state)

# Ensure the state is a FloatTensor and move it to the appropriate device (CPU or GPU)
state = state.float()

# Pass the preprocessed state through the model
action = model(state)

# Print the action
