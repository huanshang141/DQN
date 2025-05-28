import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
from collections import deque
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clean(self):
        self.buffer.clear()


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        
        # 环境参数
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        # 超参数 - 调整为更稳定的设置
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # 提高最小epsilon，保持一定探索
        self.epsilon_decay = getattr(args, 'epsilon_decay', 0.9995)  # 支持动态配置
        self.batch_size = getattr(args, 'batch_size', 64)  # 支持动态配置
        self.memory_size = getattr(args, 'memory_size', 50000)  # 支持动态配置
        self.update_target_freq = getattr(args, 'update_target_freq', 500)  # 支持动态配置
        self.train_freq = getattr(args, 'train_freq', 2)  # 训练频率
        self.hidden_size = args.hidden_size
        self.n_frames = args.n_frames
        
        # 学习率衰减参数 - 从命令行参数获取
        self.lr_decay = args.lr_decay
        self.lr_min = args.lr_min
        self.lr_patience = args.lr_patience
        self.lr_factor = args.lr_factor
        self.initial_lr = self.lr
        self.best_avg_reward = -float('inf')
        self.lr_stagnant_count = 0
        
        # 收敛判断参数 - 从命令行参数获取
        self.convergence_window = args.convergence_window
        self.convergence_threshold = args.convergence_threshold
        self.convergence_check_interval = args.convergence_check_interval
        
        # 梯度裁剪
        self.grad_clip = args.grad_norm_clip
        
        # 设备
        self.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
        
        # 网络
        self.q_network = QNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        # 使用简单的指数衰减调度器
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
        # 经验回放
        self.memory = ReplayBuffer(self.memory_size)
        
        # 更新目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 训练统计
        self.frame_count = 0
        self.episode_count = 0
        
        # TensorBoard
        self.writer = SummaryWriter('logs/dqn')
        
        # 绘图相关
        self.episode_rewards = []
        self.avg_rewards = []
        self.episodes = []
        
        # 创建图形目录
        os.makedirs('plots', exist_ok=True)
        
        # 设置matplotlib为非交互模式
        plt.ioff()
    
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        self.epsilon = 0.01  # 测试时使用较小的epsilon

    def train(self):
        """
        Implement your training algorithm here
        """
        if len(self.memory) < self.batch_size:
            return
        
        # GPU内存预分配以提高性能
        if torch.cuda.is_available() and self.batch_size >= 1024:
            torch.cuda.empty_cache()
        
        # 从经验回放中采样
        batch = self.memory.sample(self.batch_size)
        
        # 批量创建tensor以提高GPU利用率
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device, non_blocking=True)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device, non_blocking=True)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device, non_blocking=True)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device, non_blocking=True)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device, non_blocking=True)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: 使用主网络选择动作，目标网络评估Q值
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        # 更新epsilon - 更慢的衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 每1000步进行一次学习率衰减
        if self.frame_count % 1000 == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr > self.lr_min:
                self.scheduler.step()
        
        # 定期清理GPU缓存
        if self.frame_count % 10000 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def check_convergence(self, reward_history):
        """
        检查是否收敛：判断最近N个回合的平均奖励是否大于阈值
        """
        if len(reward_history) < self.convergence_window:
            return False, 0
            
        # 获取最近N个回合的奖励
        recent_rewards = list(reward_history)[-self.convergence_window:]
        mean_reward = np.mean(recent_rewards)
        
        # 判断是否收敛：平均奖励大于阈值
        is_converged = mean_reward >= self.convergence_threshold
        
        return is_converged, mean_reward

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        if not test and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return q_values.max(1)[1].item()

    def plot_rewards(self, save_path='plots/reward_curve.png'):
        """
        绘制reward曲线图
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制每回合奖励
        plt.plot(self.episodes, self.episode_rewards, 'b-', alpha=0.6, label='Episode Reward')
        plt.axhline(y=self.convergence_threshold, color='g', linestyle='--', label=f'Target ({self.convergence_threshold})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Reward curve saved to {save_path}")

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        reward_history = deque(maxlen=100)
        all_rewards = []  # 存储所有回合的奖励用于收敛判断
        
        print(f"开始DQN训练...")
        print(f"收敛条件：最近{self.convergence_window}回合平均奖励≥{self.convergence_threshold}")
        print(f"稳定性改进：Double DQN + 梯度裁剪 + 更大batch_size + 更慢epsilon衰减")
        print("-" * 70)
        
        while self.frame_count < self.n_frames:
            state = self.env.reset()[0]
            episode_reward = 0
            done = False
            
            while not done:
                action = self.make_action(state, test=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 存储经验
                self.memory.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                self.frame_count += 1
                
                # 训练 - 使用可配置的训练频率
                if self.frame_count % self.train_freq == 0 and len(self.memory) >= self.batch_size:
                    self.train()
                
                # 更新目标网络
                if self.frame_count % self.update_target_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                
                if done:
                    break
            
            reward_history.append(episode_reward)
            all_rewards.append(episode_reward)
            self.episode_count += 1
            
            # 记录奖励数据
            self.episode_rewards.append(episode_reward)
            self.episodes.append(self.episode_count)
            avg_reward = np.mean(reward_history)
            self.avg_rewards.append(avg_reward)
            
            # 记录日志
            if self.episode_count % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Episode: {self.episode_count}, Frames: {self.frame_count}, "
                      f"Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}, LR: {current_lr:.6f}")
                
                self.writer.add_scalar('Reward/Episode', episode_reward, self.episode_count)
                self.writer.add_scalar('Reward/Average', avg_reward, self.episode_count)
                self.writer.add_scalar('Epsilon', self.epsilon, self.episode_count)
                self.writer.add_scalar('Learning_Rate', current_lr, self.episode_count)
                
                # 每50回合保存一次图
                if self.episode_count % 50 == 0:
                    self.plot_rewards()
                
                # 检查收敛
                if self.episode_count % self.convergence_check_interval == 0:
                    is_converged, mean_reward = self.check_convergence(all_rewards)
                    
                    if is_converged:
                        print("-" * 70)
                        print(f"🎉 算法收敛！在第{self.episode_count}回合达到稳定状态")
                        print(f"📊 最近{self.convergence_window}回合平均奖励: {mean_reward:.2f}")
                        print(f"🎯 使用样本数: {self.frame_count}")
                        self.plot_rewards('plots/final_reward_curve.png')
                        break
                    elif len(all_rewards) >= self.convergence_window:
                        # 输出收敛状态信息
                        print(f"   收敛检查: 平均={mean_reward:.2f}")
                
                # 动态调整学习率
                if self.episode_count % 50 == 0:  # 每50回合检查一次性能
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if avg_reward > self.best_avg_reward:
                        self.best_avg_reward = avg_reward
                        self.lr_stagnant_count = 0
                    else:
                        self.lr_stagnant_count += 1
                    
                    # 如果长时间没有改善且学习率还不是最小值，手动降低学习率
                    if self.lr_stagnant_count >= 10 and current_lr > self.lr_min:
                        new_lr = max(current_lr * self.lr_factor, self.lr_min)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f"   性能停滞{self.lr_stagnant_count}次，降低学习率至: {new_lr:.6f}")
                        self.lr_stagnant_count = 0
        else:
            # 达到最大帧数但未收敛
            print("-" * 70)
            print(f"训练结束：达到最大帧数{self.n_frames}")
            is_converged, mean_reward = self.check_convergence(all_rewards)
            if is_converged:
                print(f"✅ 已收敛: 平均奖励{mean_reward:.2f}")
            else:
                print(f"⚠️  未收敛: 平均奖励{mean_reward:.2f}")
                print(f"💡 建议：增加训练时间或调整超参数以提高稳定性")
        
        # 训练结束后绘制最终曲线
        self.plot_rewards('plots/final_reward_curve.png')
        self.writer.close()
        print(f"Training completed. Final average reward: {np.mean(reward_history):.2f}")
