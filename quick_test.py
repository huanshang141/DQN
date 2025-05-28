import json
import gymnasium as gym
from argparse import Namespace
from agent_dir.agent_dqn import AgentDQN


def load_best_config():
    """加载最佳配置"""
    try:
        with open('optimization_results/best_config.json', 'r', encoding='utf-8') as f:
            best_config = json.load(f)
        return best_config['params']
    except FileNotFoundError:
        print("未找到最佳配置文件，使用默认优化参数")
        # 基于经验的优化参数
        return {
            'lr': 0.0025,
            'lr_decay': 0.9996,
            'epsilon_decay': 0.9997,
            'batch_size': 96,
            'memory_size': 25000,
            'update_target_freq': 250,
            'hidden_size': 192,
            'convergence_window': 30,
            'convergence_threshold': 195,
            'convergence_check_interval': 20
        }


def create_optimized_args(**override_params):
    """创建优化的参数配置"""
    # 加载最佳参数
    best_params = load_best_config()
    
    # 基础参数
    default_args = {
        'env_name': 'CartPole-v0',
        'seed': 11037,
        'grad_norm_clip': 1.0,
        'lr_min': 0.00005,
        'lr_patience': 50,
        'lr_factor': 0.5,
        'test': False,
        'use_cuda': True,
        'n_frames': 60000,
        'gamma': 0.99
    }
    
    # 合并参数
    default_args.update(best_params)
    default_args.update(override_params)
    
    return Namespace(**default_args)


def test_optimized_config():
    """测试优化配置"""
    print("使用优化参数进行快速测试...")
    
    # 创建环境和代理
    env = gym.make('CartPole-v0')
    args = create_optimized_args()
    
    print(f"优化参数:")
    print(f"  学习率: {args.lr}")
    print(f"  隐藏层大小: {args.hidden_size}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  经验池大小: {args.memory_size}")
    print(f"  目标网络更新频率: {args.update_target_freq}")
    print(f"  收敛窗口: {args.convergence_window}")
    print(f"  收敛阈值: {args.convergence_threshold}")
    
    agent = AgentDQN(env, args)
    agent.run()
    
    env.close()


if __name__ == '__main__':
    test_optimized_config()
