from libs import *
from model import *
from train import *
from dqn import *
from pruning_env import *
from evaluate import *


train_loader, test_loader = get_data_loaders()

model = SimpleNN()
model = train_model(model, train_loader)

print("Model evaluation before pruning:")
model_evaluation(model, train_loader, test_loader)

state_dim = 1  # Độ chính xác ban đầu của mô hình
action_dim = 1  # Chúng ta có một hành động là pruning
dqn_model = DQN(state_dim, action_dim)
target_model = DQN(state_dim, action_dim)
target_model.load_state_dict(dqn_model.state_dict())

optimizer = optim.Adam(dqn_model.parameters(), lr=0.001)
replay_buffer = ReplayBuffer(1000)
gamma = 0.99
batch_size = 32

env = PruningEnv(model, test_loader)
state = env.evaluate()

num_episodes = 100
for episode in range(num_episodes):
    state = np.array([state])
    action = dqn_model(torch.FloatTensor(state)).argmax().item()
    reward, next_state = env.step(action)
    replay_buffer.push(state, action, reward, next_state, 0)
    state = next_state
    
    train_dqn(env, dqn_model, target_model, optimizer, replay_buffer, batch_size, gamma)
    
    if episode % 10 == 0:
        target_model.load_state_dict(dqn_model.state_dict())
        print(f'Episode {episode}, Accuracy: {next_state:.2f}%')

# Đánh giá hiệu suất của mô hình sau pruning
print("Model evaluation after pruning:")
final_accuracy = env.evaluate()
print(f'Final Accuracy of the pruned model: {final_accuracy:.2f}%')
model_evaluation(model, train_loader, test_loader)
