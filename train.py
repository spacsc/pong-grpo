import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, in_channels=4, action_space=4):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)   # expects in_channels=4
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(16 * 18 * 18, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, action_space)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def preprocess_frame(frame):
    transform = T.Compose([
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize((84, 84)),
        T.ToTensor()
    ])
    return transform(frame)

gamma      = 0.99
alpha      = 0.0001
epsilon    = 0.1
epochs     = 1000
num_frames = 4

def select_action(model, state, epsilon, action_space):
    if random.random() < epsilon:
        return random.randint(0, action_space - 1)
    with torch.no_grad():
        q_values = model(state)
        return torch.argmax(q_values, dim=1).item()

def runloop(env, model, optimizer, criterion, epochs, num_frames, gamma, epsilon):
    frame_stack   = []
    total_rewards = []

    state, _ = env.reset()
    for epoch in range(epochs):
        state_tensor = preprocess_frame(state).unsqueeze(0).to(device)
        frame_stack.append(state_tensor)
        if len(frame_stack) > num_frames:
            frame_stack.pop(0)

        if len(frame_stack) == num_frames:
            stacked_state = torch.cat(frame_stack, dim=1).to(device)
            action = select_action(model, stacked_state, epsilon, env.action_space.n)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_tensor = preprocess_frame(next_state).unsqueeze(0).to(device)
            frame_stack.append(next_state_tensor)
            if len(frame_stack) > num_frames:
                frame_stack.pop(0)

            next_stacked_state = torch.cat(frame_stack, dim=1).to(device)

            with torch.no_grad():
                next_q_values     = model(next_stacked_state)
                max_next_q_value = torch.max(next_q_values, dim=1)[0]
            target = reward + gamma * max_next_q_value

            current_q_values = model(stacked_state)
            current_q_value  = current_q_values[0, action]

            loss = criterion(current_q_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_rewards.append(reward)

            if terminated or truncated:
                state, _ = env.reset()
                frame_stack.clear()
            else:
                state = next_state

        if epoch % 100 == 0:
            avg = np.mean(total_rewards[-100:]) if total_rewards else 0.0
            print(f"Epoch {epoch}, Average Reward: {avg:.2f}")

        env.render()

    env.close()

env       = gym.make("ALE/Assault-v5", render_mode="human")
model     = Model(in_channels=num_frames, action_space=env.action_space.n).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
criterion = nn.MSELoss()

runloop(env, model, optimizer, criterion, epochs, num_frames, gamma, epsilon)
