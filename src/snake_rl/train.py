import torch
import torch.nn.functional as F


def train_step(agent, optimizer, batch, gamma: float):
    states = torch.tensor(batch["state"], dtype=torch.float32, device=agent.device)
    actions = torch.tensor(batch["action"], dtype=torch.int64, device=agent.device).unsqueeze(1)
    rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=agent.device).unsqueeze(1)
    next_states = torch.tensor(batch["next_state"], dtype=torch.float32, device=agent.device)
    dones = torch.tensor(batch["done"], dtype=torch.float32, device=agent.device).unsqueeze(1)

    q_values = agent.q_network(states).gather(1, actions)
    with torch.no_grad():
        next_q = agent.q_network(next_states).max(1, keepdim=True)[0]
        target = rewards + gamma * (1 - dones) * next_q

    loss = F.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
