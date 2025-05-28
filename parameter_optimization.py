import json
import time
import numpy as np
import gymnasium as gym
from argparse import Namespace
from agent_dir.agent_dqn import AgentDQN
import matplotlib.pyplot as plt
import os
import itertools
import random as python_random
from gpu_monitor import GPUMonitor


class ParameterOptimizer:
    def __init__(self):
        self.results = []
        self.best_config = None
        self.best_frames = float('inf')
        
        # 定义参数搜索空间 - 针对GPU优化，避免小批次问题
        self.param_ranges = {
            'lr': [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01],
            'lr_decay': [0.995, 0.997, 0.999, 0.9995, 0.9997, 0.9999],
            'epsilon_decay': [0.995, 0.997, 0.999, 0.9995, 0.9997, 0.9999],
            'batch_size': [64, 128, 256, 512, 1024],  # 确保最小批次为64
            'memory_size': [50000, 100000, 200000, 300000, 500000],
            'update_target_freq': [100, 200, 300, 500, 750, 1000],
            'hidden_size': [256, 512, 1024, 2048],
            'grad_norm_clip': [0.5, 1.0, 2.0, 5.0, 10.0],
            'train_freq': [1, 2, 4],
            'num_layers': [2, 3, 4, 5]
        }
        
        # 随机搜索的参数范围（连续值）
        self.continuous_ranges = {
            'lr': (0.0001, 0.01),
            'lr_decay': (0.99, 0.9999),
            'epsilon_decay': (0.99, 0.9999),
            'grad_norm_clip': (0.1, 10.0)
        }
        
        # 离散值范围 - GPU优化版本，确保批次大小足够
        self.discrete_ranges = {
            'batch_size': [64, 128, 256, 512, 1024, 1536, 2048],  # 移除小批次
            'memory_size': list(range(50000, 500001, 50000)),
            'update_target_freq': list(range(50, 1001, 50)),
            'hidden_size': [128, 256, 384, 512, 768, 1024, 1536, 2048],
            'train_freq': [1, 2, 4, 8],
            'num_layers': [2, 3, 4, 5, 6]
        }
        
        # GPU监控器
        self.gpu_monitor = None

    def create_args(self, **kwargs):
        """创建参数配置"""
        default_args = {
            'env_name': 'CartPole-v1',  # 使用v1版本
            'seed': 11037,
            'hidden_size': 256,  # 增加默认网络大小
            'lr': 0.001,
            'gamma': 0.99,
            'grad_norm_clip': 1.0,
            'lr_decay': 0.9995,
            'lr_min': 0.00005,
            'lr_patience': 50,
            'lr_factor': 0.5,
            'convergence_window': 30,
            'convergence_threshold': 195,
            'convergence_check_interval': 20,
            'test': False,
            'use_cuda': True,
            'n_frames': 60000,
            'train_freq': 1,
            'num_layers': 3,
            'batch_size': 64,  # 确保默认批次足够大
        }
        
        # 更新参数
        default_args.update(kwargs)
        
        # 确保批次大小至少为64
        if 'batch_size' in default_args:
            default_args['batch_size'] = max(default_args['batch_size'], 64)
        
        return Namespace(**default_args)

    def generate_random_params(self):
        """生成随机参数组合"""
        params = {}
        
        # 连续参数
        for param, (min_val, max_val) in self.continuous_ranges.items():
            if param in ['lr_decay', 'epsilon_decay']:
                # 对于衰减参数，使用对数空间采样
                log_min = np.log10(1 - max_val)
                log_max = np.log10(1 - min_val)
                decay_factor = 1 - 10**(np.random.uniform(log_min, log_max))
                params[param] = decay_factor
            else:
                # 对于学习率，使用对数空间
                if param == 'lr':
                    params[param] = 10**(np.random.uniform(np.log10(min_val), np.log10(max_val)))
                else:
                    params[param] = np.random.uniform(min_val, max_val)
        
        # 离散参数
        for param, values in self.discrete_ranges.items():
            params[param] = python_random.choice(values)
        
        return params
    
    def grid_search_small(self):
        """小规模网格搜索 - 修复批次大小问题"""
        print("执行小规模网格搜索...")
        
        # 选择较少的参数值进行网格搜索，确保批次大小足够
        small_ranges = {
            'lr': [0.001, 0.003, 0.005],
            'batch_size': [64, 128, 256],  # 移除32
            'hidden_size': [128, 256, 512],  # 调整范围
            'update_target_freq': [200, 400, 600]
        }
        
        # 生成所有组合
        param_names = list(small_ranges.keys())
        param_values = list(small_ranges.values())
        
        total_combinations = np.prod([len(values) for values in param_values])
        print(f"总共 {total_combinations} 个参数组合")
        
        for i, combination in enumerate(itertools.product(*param_values)):
            params = dict(zip(param_names, combination))
            config_name = f"Grid_{i+1:03d}"
            
            print(f"\n进度: {i+1}/{total_combinations}")
            self.test_configuration(config_name, **params)
    
    def random_search(self, n_trials=50):
        """随机搜索"""
        print(f"执行随机搜索 ({n_trials} 次试验)...")
        
        for i in range(n_trials):
            params = self.generate_random_params()
            config_name = f"Random_{i+1:03d}"
            
            print(f"\n进度: {i+1}/{n_trials}")
            self.test_configuration(config_name, **params)
    
    def adaptive_search(self, n_trials=30):
        """自适应搜索：基于当前最佳结果进行局部搜索"""
        print(f"执行自适应搜索 ({n_trials} 次试验)...")
        
        if not self.best_config:
            print("没有基础配置，先执行随机搜索...")
            self.random_search(10)
        
        for i in range(n_trials):
            if self.best_config:
                # 基于最佳配置进行扰动
                params = self.perturb_best_params()
            else:
                # 如果还没有最佳配置，使用随机搜索
                params = self.generate_random_params()
            
            config_name = f"Adaptive_{i+1:03d}"
            print(f"\n进度: {i+1}/{n_trials}")
            self.test_configuration(config_name, **params)
    
    def perturb_best_params(self):
        """基于最佳参数进行扰动"""
        best_params = self.best_config['params'].copy()
        params = {}
        
        for param_name, current_value in best_params.items():
            if param_name in self.continuous_ranges:
                min_val, max_val = self.continuous_ranges[param_name]
                # 在当前值附近进行扰动
                noise_factor = 0.2  # 20%的扰动
                if param_name == 'lr':
                    # 对数空间扰动
                    log_current = np.log10(current_value)
                    log_noise = np.random.normal(0, noise_factor)
                    new_value = 10**(log_current + log_noise)
                    params[param_name] = np.clip(new_value, min_val, max_val)
                else:
                    noise = np.random.normal(0, (max_val - min_val) * noise_factor)
                    params[param_name] = np.clip(current_value + noise, min_val, max_val)
            
            elif param_name in self.discrete_ranges:
                values = self.discrete_ranges[param_name]
                current_idx = values.index(current_value) if current_value in values else 0
                # 在附近索引中选择
                nearby_range = 3
                start_idx = max(0, current_idx - nearby_range)
                end_idx = min(len(values), current_idx + nearby_range + 1)
                params[param_name] = python_random.choice(values[start_idx:end_idx])
            else:
                params[param_name] = current_value
        
        return params
    
    def test_configuration(self, config_name, **params):
        """测试单个参数配置"""
        print(f"测试配置: {config_name}")
        print(f"参数: {params}")
        
        # 启动GPU监控
        self.gpu_monitor = GPUMonitor()
        self.gpu_monitor.start_monitoring()
        
        try:
            # 创建环境和代理
            env = gym.make('CartPole-v1')  # 使用v1版本
            args = self.create_args(**params)
            
            agent = AgentDQN(env, args)
            
            # 设置所有参数到agent中
            for param_name, param_value in params.items():
                if hasattr(agent, param_name):
                    setattr(agent, param_name, param_value)
            
            # 特殊处理需要重新初始化的参数
            if 'batch_size' in params:
                agent.batch_size = max(params['batch_size'], 64)  # 确保最小64
            if 'memory_size' in params:
                agent.memory_size = params['memory_size']
                agent.memory = agent.ReplayBuffer(params['memory_size'])
            if 'update_target_freq' in params:
                agent.update_target_freq = params['update_target_freq']
            if 'train_freq' in params:
                agent.train_freq = params['train_freq']
            if 'num_layers' in params or 'hidden_size' in params:
                # 重新构建网络以支持更多层
                agent.rebuild_network(
                    hidden_size=params.get('hidden_size', agent.hidden_size),
                    num_layers=params.get('num_layers', 3)
                )
            
            # 记录开始时间
            start_time = time.time()
            
            # 运行训练
            converged, frames_used = self.run_training(agent)
            
            # 记录结束时间
            end_time = time.time()
            training_time = end_time - start_time
            
            # 停止监控并获取统计
            self.gpu_monitor.stop_monitoring()
            gpu_stats = self.gpu_monitor.get_current_stats()
            
            # 保存结果
            result = {
                'config_name': config_name,
                'params': params,
                'converged': converged,
                'frames_used': frames_used,
                'training_time': training_time,
                'success': converged and frames_used < self.best_frames,
                'gpu_stats': gpu_stats  # 添加GPU统计信息
            }
            
            self.results.append(result)
            
            # 打印GPU利用率信息
            if gpu_stats:
                print(f"GPU利用率: 平均{gpu_stats['avg_gpu_usage']:.1f}%, 最大{gpu_stats['max_gpu_usage']:.1f}%")
                print(f"GPU显存: 平均{gpu_stats['avg_gpu_memory']:.1f}%, 最大{gpu_stats['max_gpu_memory']:.1f}%")
            
            if converged and frames_used < self.best_frames:
                self.best_frames = frames_used
                self.best_config = result
                print(f"🎉 新的最佳配置！使用帧数: {frames_used}")
                # 保存当前最佳配置
                self.save_best_config_immediately()
            
            # 清理
            agent.writer.close()
            env.close()
            
            return result
            
        except Exception as e:
            if self.gpu_monitor:
                self.gpu_monitor.stop_monitoring()
            print(f"❌ 配置测试失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'config_name': config_name,
                'params': params,
                'converged': False,
                'frames_used': float('inf'),
                'training_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def save_best_config_immediately(self):
        """立即保存当前最佳配置"""
        if self.best_config:
            os.makedirs('optimization_results', exist_ok=True)
            with open('optimization_results/current_best.json', 'w', encoding='utf-8') as f:
                json.dump(self.best_config, f, indent=2, ensure_ascii=False)
    
    def run_training(self, agent):
        """运行训练并检测收敛"""
        all_rewards = []
        
        while agent.frame_count < agent.n_frames:
            state = agent.env.reset()[0]
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.make_action(state, test=False)
                next_state, reward, terminated, truncated, _ = agent.env.step(action)
                done = terminated or truncated
                
                agent.memory.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                agent.frame_count += 1
                
                if agent.frame_count % 2 == 0 and len(agent.memory) >= agent.batch_size:
                    agent.train()
                
                if agent.frame_count % agent.update_target_freq == 0:
                    agent.target_network.load_state_dict(agent.q_network.state_dict())
                
                if done:
                    break
            
            all_rewards.append(episode_reward)
            agent.episode_count += 1
            
            # 检查收敛
            if agent.episode_count % agent.convergence_check_interval == 0:
                is_converged, mean_reward = agent.check_convergence(all_rewards)
                if is_converged:
                    print(f"✅ 收敛！回合: {agent.episode_count}, 帧数: {agent.frame_count}, 平均奖励: {mean_reward:.2f}")
                    return True, agent.frame_count
        
        # 未收敛
        print(f"❌ 未收敛，达到最大帧数: {agent.frame_count}")
        return False, agent.frame_count
    
    def optimize(self, method='all'):
        """执行参数优化"""
        print("开始系统化参数优化...")
        
        if method == 'all' or method == 'grid':
            self.grid_search_small()
        
        if method == 'all' or method == 'random':
            self.random_search(n_trials=30)
        
        if method == 'all' or method == 'adaptive':
            self.adaptive_search(n_trials=20)
    
    def save_results(self):
        """保存优化结果"""
        os.makedirs('optimization_results', exist_ok=True)
        
        # 保存详细结果
        with open('optimization_results/detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 保存最佳配置
        if self.best_config:
            with open('optimization_results/best_config.json', 'w', encoding='utf-8') as f:
                json.dump(self.best_config, f, indent=2, ensure_ascii=False)
        
        # 生成报告
        self.generate_report()
    
    def generate_report(self):
        """生成优化报告"""
        report = []
        report.append("参数优化报告")
        report.append("="*50)
        report.append(f"总测试配置数: {len(self.results)}")
        
        # 统计成功率
        successful_configs = [r for r in self.results if r.get('converged', False)]
        success_rate = len(successful_configs) / len(self.results) * 100
        report.append(f"收敛成功率: {success_rate:.1f}%")
        
        if successful_configs:
            # 最佳配置
            report.append(f"\n最佳配置 (最少帧数: {self.best_frames}):")
            report.append(f"配置名称: {self.best_config['config_name']}")
            report.append(f"训练时间: {self.best_config['training_time']:.2f}秒")
            report.append("参数设置:")
            for key, value in self.best_config['params'].items():
                report.append(f"  {key}: {value}")
            
            # 所有成功配置排序
            successful_configs.sort(key=lambda x: x['frames_used'])
            report.append(f"\n所有成功配置 (按帧数排序):")
            for i, config in enumerate(successful_configs, 1):
                report.append(f"{i}. {config['config_name']}: {config['frames_used']} 帧 "
                            f"({config['training_time']:.1f}秒)")
        
        # 失败配置
        failed_configs = [r for r in self.results if not r.get('converged', False)]
        if failed_configs:
            report.append(f"\n失败配置:")
            for config in failed_configs:
                reason = config.get('error', '未收敛')
                report.append(f"- {config['config_name']}: {reason}")
        
        # 保存报告
        with open('optimization_results/optimization_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # 打印报告
        print('\n'.join(report))
    
    def plot_results(self):
        """绘制优化结果图表"""
        if not self.results:
            return
        
        successful_configs = [r for r in self.results if r.get('converged', False)]
        if not successful_configs:
            return
        
        plt.figure(figsize=(12, 8))
        
        # 帧数对比
        plt.subplot(2, 2, 1)
        names = [r['config_name'] for r in successful_configs]
        frames = [r['frames_used'] for r in successful_configs]
        plt.bar(range(len(names)), frames)
        plt.xticks(range(len(names)), names, rotation=45)
        plt.ylabel('使用帧数')
        plt.title('各配置收敛所需帧数')
        
        # 训练时间对比
        plt.subplot(2, 2, 2)
        times = [r['training_time'] for r in successful_configs]
        plt.bar(range(len(names)), times)
        plt.xticks(range(len(names)), names, rotation=45)
        plt.ylabel('训练时间 (秒)')
        plt.title('各配置训练时间')
        
        # 效率对比 (帧数/时间)
        plt.subplot(2, 2, 3)
        efficiency = [f/t for f, t in zip(frames, times)]
        plt.bar(range(len(names)), efficiency)
        plt.xticks(range(len(names)), names, rotation=45)
        plt.ylabel('帧数/秒')
        plt.title('训练效率对比')
        
        # 学习率对比
        plt.subplot(2, 2, 4)
        lrs = [r['params'].get('lr', 0) for r in successful_configs]
        plt.scatter(lrs, frames)
        plt.xlabel('学习率')
        plt.ylabel('使用帧数')
        plt.title('学习率 vs 收敛帧数')
        
        plt.tight_layout()
        plt.savefig('optimization_results/optimization_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_parameter_importance(self):
        """分析参数重要性"""
        if not self.results:
            return
        
        successful_configs = [r for r in self.results if r.get('converged', False)]
        if len(successful_configs) < 3:
            return
        
        # 分析每个参数与性能的关系
        param_analysis = {}
        
        for param_name in self.param_ranges.keys():
            param_values = []
            frame_counts = []
            
            for config in successful_configs:
                if param_name in config['params']:
                    param_values.append(config['params'][param_name])
                    frame_counts.append(config['frames_used'])
            
            if len(param_values) > 2:
                correlation = np.corrcoef(param_values, frame_counts)[0, 1]
                param_analysis[param_name] = {
                    'correlation': correlation,
                    'best_value': param_values[np.argmin(frame_counts)],
                    'avg_value': np.mean(param_values),
                    'std_value': np.std(param_values)
                }
        
        # 保存分析结果
        os.makedirs('optimization_results', exist_ok=True)
        with open('optimization_results/parameter_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(param_analysis, f, indent=2, ensure_ascii=False)
        
        return param_analysis


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='参数优化工具')
    parser.add_argument('--method', choices=['all', 'grid', 'random', 'adaptive'], 
                       default='all', help='优化方法')
    parser.add_argument('--trials', type=int, default=30, help='随机搜索试验次数')
    args = parser.parse_args()
    
    optimizer = ParameterOptimizer()
    
    try:
        if args.method == 'random':
            optimizer.random_search(n_trials=args.trials)
        elif args.method == 'adaptive':
            optimizer.adaptive_search(n_trials=args.trials)
        else:
            optimizer.optimize(method=args.method)
        
        optimizer.save_results()
        optimizer.plot_results()
        optimizer.analyze_parameter_importance()
        
        print(f"\n🎉 参数优化完成！")
        if optimizer.best_config:
            print(f"最佳配置使用帧数: {optimizer.best_frames}")
            print(f"详细结果保存在: optimization_results/")
        else:
            print("❌ 没有找到收敛的配置")
            
    except KeyboardInterrupt:
        print("\n⚠️ 优化被用户中断")
        optimizer.save_results()
    except Exception as e:
        print(f"❌ 优化过程出错: {e}")
        optimizer.save_results()


if __name__ == '__main__':
    main()
