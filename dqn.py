from libs import *
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def size(self):
        return len(self.buffer)

def compute_loss(batch, model, target_model, gamma):
    states, actions, rewards, next_states, dones = batch

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = model(states)
    next_q_values = target_model(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()
    return loss

def train_dqn(env, model, target_model, optimizer, replay_buffer, batch_size, gamma):
    if replay_buffer.size() < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    loss = compute_loss(batch, model, target_model, gamma)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
