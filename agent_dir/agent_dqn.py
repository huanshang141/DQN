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
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(QNetwork, self).__init__()
        self.num_layers = num_layers
        
        # 构建多层网络 - 修复BatchNorm问题
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            # 使用LayerNorm替代BatchNorm，避免小批次问题
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.Dropout(0.1))
            current_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


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
        
        # 超参数 - GPU优化版本
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = getattr(args, 'epsilon_decay', 0.9995)
        self.batch_size = getattr(args, 'batch_size', 64)  # 确保最小批次大小
        self.memory_size = getattr(args, 'memory_size', 100000)
        self.update_target_freq = getattr(args, 'update_target_freq', 500)
        self.hidden_size = args.hidden_size
        self.n_frames = args.n_frames
        self.train_freq = getattr(args, 'train_freq', 1)
        self.num_layers = getattr(args, 'num_layers', 3)
        
        # 确保批次大小至少为2，避免BatchNorm问题
        self.batch_size = max(self.batch_size, 2)
        
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
        print(f"使用设备: {self.device}")
        
        # GPU优化设置
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # 优化cuDNN性能
            torch.backends.cudnn.deterministic = False  # 允许非确定性算法以提高性能
        
        # 网络 - 支持多层
        self.q_network = QNetwork(self.state_size, self.hidden_size, self.action_size, self.num_layers).to(self.device)
        self.target_network = QNetwork(self.state_size, self.hidden_size, self.action_size, self.num_layers).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # 使用更aggressive的学习率调度
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
        
        # 批处理优化
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_state_buffer = []
        self.done_buffer = []
        
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

    def rebuild_network(self, hidden_size=None, num_layers=None):
        """重新构建网络以支持不同的架构"""
        if hidden_size:
            self.hidden_size = hidden_size
        if num_layers:
            self.num_layers = num_layers
        
        # 重新创建网络
        self.q_network = QNetwork(self.state_size, self.hidden_size, self.action_size, self.num_layers).to(self.device)
        self.target_network = QNetwork(self.state_size, self.hidden_size, self.action_size, self.num_layers).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
        # 更新目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self):
        """
        Implement your training algorithm here - GPU优化版本
        """
        if len(self.memory) < self.batch_size:
            return
        
        # 使用更大的批次进行训练以提高GPU利用率
        effective_batch_size = min(self.batch_size, len(self.memory))
        
        # 确保批次大小至少为2
        if effective_batch_size < 2:
            return
        
        # 从经验回放中采样
        batch = self.memory.sample(effective_batch_size)
        
        # 预分配tensor以提高内存效率
        states = torch.empty((effective_batch_size, self.state_size), dtype=torch.float32, device=self.device)
        actions = torch.empty(effective_batch_size, dtype=torch.long, device=self.device)
        rewards = torch.empty(effective_batch_size, dtype=torch.float32, device=self.device)
        next_states = torch.empty((effective_batch_size, self.state_size), dtype=torch.float32, device=self.device)
        dones = torch.empty(effective_batch_size, dtype=torch.bool, device=self.device)
        
        # 填充数据
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            states[i] = torch.from_numpy(np.array(state))
            actions[i] = action
            rewards[i] = reward
            next_states[i] = torch.from_numpy(np.array(next_state))
            dones[i] = done
        
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
        
        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 学习率衰减
        if self.frame_count % 1000 == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr > self.lr_min:
                self.scheduler.step()

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
        
        # 确保输入是正确的形状
        if isinstance(observation, np.ndarray):
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        else:
            state = torch.FloatTensor([observation]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
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
        Implement the interaction between agent and environment here - GPU优化版本
        """
        reward_history = deque(maxlen=100)
        all_rewards = []
        
        print(f"开始DQN训练... (GPU优化版本)")
        print(f"批次大小: {self.batch_size}, 网络层数: {self.num_layers}, 隐藏层大小: {self.hidden_size}")
        print(f"收敛条件：最近{self.convergence_window}回合平均奖励≥{self.convergence_threshold}")
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
                
                # 更频繁的训练以提高GPU利用率
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
