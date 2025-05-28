import time
import psutil
import threading
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("警告: GPUtil未安装，GPU监控功能受限")

class GPUMonitor:
    def __init__(self, monitor_interval=1.0):
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.gpu_usage = deque(maxlen=1000)
        self.gpu_memory = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        self.start_time = None
        self.has_gpu = HAS_GPUTIL
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        if self.has_gpu:
            print("GPU监控已启动...")
        else:
            print("CPU监控已启动...")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        if self.has_gpu:
            print("GPU监控已停止")
        else:
            print("CPU监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 获取GPU信息
                if self.has_gpu:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # 使用第一个GPU
                        self.gpu_usage.append(gpu.load * 100)
                        self.gpu_memory.append(gpu.memoryUsed / gpu.memoryTotal * 100)
                    else:
                        self.gpu_usage.append(0)
                        self.gpu_memory.append(0)
                else:
                    self.gpu_usage.append(0)
                    self.gpu_memory.append(0)
                
                # 获取CPU信息
                self.cpu_usage.append(psutil.cpu_percent())
                
                # 记录时间戳
                self.timestamps.append(time.time() - self.start_time)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(self.monitor_interval)
    
    def get_current_stats(self):
        """获取当前统计信息"""
        if not self.gpu_usage:
            return None
        
        return {
            'gpu_usage': self.gpu_usage[-1] if self.gpu_usage else 0,
            'gpu_memory': self.gpu_memory[-1] if self.gpu_memory else 0,
            'cpu_usage': self.cpu_usage[-1] if self.cpu_usage else 0,
            'avg_gpu_usage': np.mean(list(self.gpu_usage)) if self.gpu_usage else 0,
            'avg_gpu_memory': np.mean(list(self.gpu_memory)) if self.gpu_memory else 0,
            'max_gpu_usage': max(self.gpu_usage) if self.gpu_usage else 0,
            'max_gpu_memory': max(self.gpu_memory) if self.gpu_memory else 0
        }
    
    def plot_stats(self, save_path='gpu_monitor.png'):
        """绘制监控统计图"""
        if not self.timestamps:
            print("没有监控数据可绘制")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        times = list(self.timestamps)
        
        # GPU利用率
        axes[0].plot(times, list(self.gpu_usage), 'r-', label='GPU Usage')
        axes[0].set_ylabel('GPU Usage (%)')
        axes[0].set_title('GPU Utilization Over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)
        
        # GPU显存使用
        axes[1].plot(times, list(self.gpu_memory), 'b-', label='GPU Memory')
        axes[1].set_ylabel('GPU Memory (%)')
        axes[1].set_title('GPU Memory Usage Over Time')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        # CPU使用率
        axes[2].plot(times, list(self.cpu_usage), 'g-', label='CPU Usage')
        axes[2].set_ylabel('CPU Usage (%)')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_title('CPU Usage Over Time')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"监控图表已保存到: {save_path}")
    
    def print_summary(self):
        """打印监控摘要"""
        stats = self.get_current_stats()
        if stats:
            print("\n" + "="*50)
            print("GPU监控摘要")
            print("="*50)
            print(f"平均GPU利用率: {stats['avg_gpu_usage']:.1f}%")
            print(f"最大GPU利用率: {stats['max_gpu_usage']:.1f}%")
            print(f"平均GPU显存使用: {stats['avg_gpu_memory']:.1f}%")
            print(f"最大GPU显存使用: {stats['max_gpu_memory']:.1f}%")
            print(f"监控时长: {self.timestamps[-1]:.1f}秒")
            print("="*50)


# 使用示例
if __name__ == '__main__':
    monitor = GPUMonitor()
    monitor.start_monitoring()
    
    try:
        # 模拟训练过程
        time.sleep(60)  # 监控60秒
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop_monitoring()
        monitor.plot_stats()
        monitor.print_summary()
