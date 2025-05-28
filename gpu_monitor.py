import time
import torch
import psutil
import threading
from collections import deque


class GPUMonitor:
    def __init__(self, log_interval=5):
        self.log_interval = log_interval
        self.monitoring = False
        self.gpu_usage_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        
    def start_monitoring(self):
        """开始监控GPU使用情况"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        print("GPU监控已启动...")
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            if torch.cuda.is_available():
                # GPU利用率和内存使用
                gpu_util = torch.cuda.utilization()
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                memory_percent = (memory_used / memory_total) * 100
                
                self.gpu_usage_history.append(gpu_util)
                self.memory_usage_history.append(memory_percent)
                
                # CPU使用率
                cpu_percent = psutil.cpu_percent()
                
                print(f"GPU利用率: {gpu_util:5.1f}% | "
                      f"GPU内存: {memory_used:5.2f}/{memory_total:5.2f}GB ({memory_percent:5.1f}%) | "
                      f"CPU: {cpu_percent:5.1f}%")
                      
                # 如果GPU利用率持续过低，给出建议
                if len(self.gpu_usage_history) >= 10:
                    avg_gpu_util = sum(list(self.gpu_usage_history)[-10:]) / 10
                    if avg_gpu_util < 30:
                        print(f"⚠️  GPU利用率过低 (平均{avg_gpu_util:.1f}%)，建议:")
                        print("   - 增大batch_size")
                        print("   - 增大网络规模(hidden_size)")
                        print("   - 提高训练频率(降低train_freq)")
                        print("   - 增大经验池大小")
            
            time.sleep(self.log_interval)
    
    def get_stats(self):
        """获取统计信息"""
        if self.gpu_usage_history:
            avg_gpu = sum(self.gpu_usage_history) / len(self.gpu_usage_history)
            max_gpu = max(self.gpu_usage_history)
            avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history)
            
            return {
                'avg_gpu_utilization': avg_gpu,
                'max_gpu_utilization': max_gpu,
                'avg_memory_usage': avg_memory
            }
        return None


def main():
    """独立运行GPU监控"""
    monitor = GPUMonitor(log_interval=2)
    monitor.start_monitoring()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        stats = monitor.get_stats()
        if stats:
            print(f"\n监控结束，统计信息:")
            print(f"平均GPU利用率: {stats['avg_gpu_utilization']:.1f}%")
            print(f"最大GPU利用率: {stats['max_gpu_utilization']:.1f}%")
            print(f"平均内存使用: {stats['avg_memory_usage']:.1f}%")


if __name__ == '__main__':
    main()
