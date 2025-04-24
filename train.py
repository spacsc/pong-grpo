import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyModel(nn.Module):
    def __init__(self, in_channels=4, action_space=4):
        super(PolicyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 18 * 18, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, action_space)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

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
epochs     = 1000
num_frames = 4

def runloop(env, model, optimizer, epochs, num_frames, gamma):
    frame_stack   = []
    total_rewards = []

    state, _ = env.reset()
    for epoch in range(epochs):
        log_probs = []
        rewards   = []
        done      = False

        while not done:
            state_tensor = preprocess_frame(state).unsqueeze(0).to(device)
            frame_stack.append(state_tensor)
            if len(frame_stack) > num_frames:
                frame_stack.pop(0)

            if len(frame_stack) < num_frames:
                action = env.action_space.sample()
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                continue

            stacked_state = torch.cat(frame_stack, dim=1).to(device)
            probs = model(stacked_state)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)

            if terminated or truncated:
                done = True
                state, _ = env.reset()
                frame_stack.clear()
            else:
                state = next_state

        # Compute discounted returns
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        log_probs = torch.stack(log_probs)
        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_rewards.append(sum(rewards))
        if epoch % 10 == 0:
            avg = np.mean(total_rewards[-10:])
            print(f"Epoch {epoch}, Avg Reward (last 10): {avg:.2f}")

        env.render()

    env.close()

env       = gym.make("ALE/Assault-v5", render_mode="human")
model     = PolicyModel(in_channels=num_frames, action_space=env.action_space.n).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

runloop(env, model, optimizer, epochs, num_frames, gamma)
