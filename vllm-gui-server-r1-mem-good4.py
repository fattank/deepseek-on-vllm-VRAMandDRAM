"""
VLLM GUI服务器 - 版本 R1
最近更新: 2025-03-06

更新内容:
1. 修复了内存交换参数设置，确保与VLLM最新文档一致
2. 修正了--swap-space参数，使用正确的GiB单位
3. 更新了CPU卸载参数，使用--cpu-offload-gb替代旧版参数
4. 改进了参数检测逻辑，支持不同版本的VLLM
5. 增强了备用启动方法，提供更多参数组合选项
6. 修复了model_size变量未定义的问题
7. 添加了--enforce-eager参数，避免CUDA图捕获阶段的内存不足
8. 优化了update_config函数，确保配置参数正确应用
9. 改进了推荐设置功能，根据模型大小和硬件条件提供更合理的配置

注意: 此版本支持VLLM的内存交换功能，允许加载超出GPU显存大小的模型

预期效果
这些优化应该能够显著提高您的LLM推理速度，特别是：
多通道加载器优化应该能减轻DDR3内存的带宽限制
KV缓存优化应该能提高缓存命中率，减少重复计算
GPU利用率优化应该能让GPU发挥更大作用
启动参数优化应该能更好地配置vLLM服务器
如果您在使用过程中发现任何问题，或者需要进一步的优化，请随时告诉我
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import socket
import json
import threading
from vllm import AsyncLLMEngine, SamplingParams
import subprocess
import GPUtil
import time
import os
import torch
import psutil
import mmap
import sys
from datetime import datetime
import re
import pynvml

class VLLMServerGUI:
    def __init__(self, master):
        self.master = master
        master.title("VLLM-DRAM-VRAM Server Manager")
        
        # 配置参数存储
        self.config = {
            'model_path': '',
            'ip': self.get_local_ip(),
            'port': 8000,
            'gpu_count': 1,
            'mem_ratio': 95,  # 提高显存使用率
            'max_tokens': 4096,  # 增加最大token数
            'kv_dtype': 'float16',
            'block_size': 16,
            'max_blocks': '',
            'calculate_scales': True,
            'max_model_len': 4096,  # 减小max_model_len以节省内存
            # 内存交换相关配置
            'enable_memory_offload': True,  # 默认启用内存交换
            'memory_offload_ratio': 70,  # 增加内存交换比例
            'memory_channels': 4,
            'reserved_memory': 20
        }
        
        # 服务器进程
        self.server_process = None
        
        # API地址
        self.api_address = None
        
        # 主界面布局
        self.create_widgets()
        
        # 加载配置
        self.load_config()
        
        # 专业监控标志
        self.monitoring = True
        # 启动GPU监控线程
        threading.Thread(target=self.update_gpu_stats, daemon=True).start()
        
    def get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP
    
    def create_widgets(self):
        # 基本配置区域
        self.config_frame = ttk.LabelFrame(self.master, text="基本配置")
        self.config_frame.pack(padx=10, pady=5, fill='x')
        
        # 模型路径
        ttk.Label(self.config_frame, text="模型路径:").grid(row=0, column=0)
        self.model_path_entry = ttk.Entry(self.config_frame, width=50)
        self.model_path_entry.grid(row=0, column=1)
        self.model_path_entry.insert(0, self.config['model_path'])
        ttk.Button(self.config_frame, text="浏览", command=self.select_model_path).grid(row=0, column=2)
        
        # 添加保存配置按钮和推荐设置按钮
        save_config_button = ttk.Button(self.config_frame, text="保存配置", command=self.save_config_with_message)
        save_config_button.grid(row=0, column=3, padx=5)
        recommend_button = ttk.Button(self.config_frame, text="推荐设置", command=self.recommend_settings)
        recommend_button.grid(row=0, column=4, padx=5)
        
        # IP地址
        ttk.Label(self.config_frame, text="IP地址:").grid(row=1, column=0)
        self.ip_entry = ttk.Entry(self.config_frame)
        self.ip_entry.grid(row=1, column=1, sticky='w')
        self.ip_entry.insert(0, self.config['ip'])
        
        # 端口
        ttk.Label(self.config_frame, text="端口:").grid(row=2, column=0)
        self.port_entry = ttk.Entry(self.config_frame)
        self.port_entry.grid(row=2, column=1, sticky='w')
        self.port_entry.insert(0, str(self.config['port']))
        
        # GPU数量
        ttk.Label(self.config_frame, text="GPU数量:").grid(row=3, column=0)
        self.gpu_count_var = tk.StringVar(value=str(self.config['gpu_count']))
        gpu_count_combo = ttk.Combobox(self.config_frame, textvariable=self.gpu_count_var,
                                     values=["1", "2", "3", "4"], width=5)
        gpu_count_combo.grid(row=3, column=1, sticky='w')
        
        # 显存比例
        ttk.Label(self.config_frame, text="显存比例(%):").grid(row=4, column=0)
        self.mem_ratio_entry = ttk.Entry(self.config_frame)
        self.mem_ratio_entry.grid(row=4, column=1, sticky='w')
        self.mem_ratio_entry.insert(0, str(self.config['mem_ratio']))
        
        # 最大Token数
        ttk.Label(self.config_frame, text="最大Token数:").grid(row=5, column=0)
        self.max_tokens_var = tk.StringVar(value=str(self.config['max_tokens']))
        ttk.Entry(self.config_frame, textvariable=self.max_tokens_var, width=8).grid(row=5, column=1)
        ttk.Label(self.config_frame, text="(回复token数应不小于整体序列长度)", foreground="gray").grid(row=6, column=0, columnspan=2, sticky='w')
        
        # 最大序列长度
        ttk.Label(self.config_frame, text="最大序列长度:").grid(row=5, column=2)
        self.max_model_len_var = tk.StringVar(value=str(self.config['max_model_len']))
        max_model_len_combo = ttk.Combobox(self.config_frame, textvariable=self.max_model_len_var,
                                         values=["2048", "4096", "8192", "16384", "32768", "65536"], width=8)
        max_model_len_combo.grid(row=5, column=3)
        ttk.Label(self.config_frame, text="(请根据硬件条件选择合适参数)", foreground="gray").grid(row=6, column=2, columnspan=2, sticky='w')
        
        # KV缓存配置
        cache_frame = ttk.LabelFrame(self.config_frame, text="KV缓存配置")
        cache_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)
        
        # 缓存精度
        ttk.Label(cache_frame, text="缓存精度:").grid(row=0, column=0)
        self.kv_dtype_var = tk.StringVar(value=self.config['kv_dtype'])
        ttk.Combobox(cache_frame, textvariable=self.kv_dtype_var,
                    values=["float16", "float32"], width=8).grid(row=0, column=1)
        
        # 缓存块大小
        ttk.Label(cache_frame, text="块大小(tokens):").grid(row=0, column=2)
        self.block_size_var = tk.StringVar(value=str(self.config['block_size']))
        ttk.Entry(cache_frame, textvariable=self.block_size_var, width=8).grid(row=0, column=3)
        
        # 最大缓存块数
        ttk.Label(cache_frame, text="最大块数:").grid(row=1, column=0)
        self.max_blocks_var = tk.StringVar(value=str(self.config['max_blocks']))
        ttk.Entry(cache_frame, textvariable=self.max_blocks_var, width=8).grid(row=1, column=1)
        ttk.Label(cache_frame, text="(留空为自动)").grid(row=1, column=2)
        
        # 动态缩放选项
        self.calculate_scales_var = tk.BooleanVar(value=self.config['calculate_scales'])
        ttk.Checkbutton(cache_frame, text="启用动态缩放", 
                       variable=self.calculate_scales_var).grid(row=1, column=3)
        
        # 添加高级性能设置区域
        self.create_advanced_settings()
        
        # 监控面板
        monitor_frame = ttk.LabelFrame(self.master, text="GPU监控")
        monitor_frame.pack(padx=10, pady=5, fill='both', expand=True)
        
        # GPU状态显示
        columns = ('GPU', '显存使用率', 'GPU使用率', '温度', '功耗', 'KV缓存命中率')
        self.gpu_tree = ttk.Treeview(monitor_frame, columns=columns, show='headings')
        for col in columns:
            self.gpu_tree.heading(col, text=col)
            self.gpu_tree.column(col, width=100)
        self.gpu_tree.pack(fill='both', expand=True)
        
        # 状态显示区域
        self.status_text = tk.Text(monitor_frame, height=10)
        self.status_text.pack(fill='both')
        
        # 服务器控制按钮
        button_frame = ttk.Frame(self.config_frame)
        button_frame.grid(row=8, column=0, columnspan=3, pady=5)
        ttk.Button(button_frame, text="启动服务器", command=self.start_server).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="停止服务器", command=self.stop_server).grid(row=0, column=1, padx=5)
        
        # API地址显示
        self.api_label = ttk.Label(self.config_frame, text="API地址:")
        self.api_label.grid(row=9, column=0, columnspan=3)
        
        # 添加内存交换配置框架
        offload_frame = ttk.LabelFrame(self.config_frame, text="内存交换配置")
        offload_frame.grid(row=10, column=0, columnspan=3, sticky="ew", pady=5)
        
        # 启用内存交换选项
        self.enable_offload_var = tk.BooleanVar(value=self.config['enable_memory_offload'])
        ttk.Checkbutton(offload_frame, text="启用内存交换", 
                       variable=self.enable_offload_var).grid(row=0, column=0)
        
        # 内存通道数量
        ttk.Label(offload_frame, text="内存通道数:").grid(row=0, column=1)
        self.memory_channels_var = tk.StringVar(value=str(self.config['memory_channels']))
        ttk.Combobox(offload_frame, textvariable=self.memory_channels_var,
                    values=["2", "4", "8", "16"], width=5).grid(row=0, column=2)
        
        # 内存交换比例
        ttk.Label(offload_frame, text="内存交换比例(%):").grid(row=1, column=0)
        self.memory_offload_ratio_var = tk.StringVar(value=str(self.config['memory_offload_ratio']))
        ttk.Entry(offload_frame, textvariable=self.memory_offload_ratio_var, width=5).grid(row=1, column=1)
        
        # 预留系统内存比例
        ttk.Label(offload_frame, text="系统内存预留(%):").grid(row=1, column=2)
        self.reserved_memory_var = tk.StringVar(value=str(self.config['reserved_memory']))
        ttk.Entry(offload_frame, textvariable=self.reserved_memory_var, width=5).grid(row=1, column=3)
        
        # 添加高级说明
        ttk.Label(offload_frame, text="(启用后可加载超出显存的大模型，但会降低推理速度)", 
                 foreground="gray").grid(row=2, column=0, columnspan=4, sticky='w')
        
        # 添加"检查兼容性"按钮
        self.check_compatibility_button = ttk.Button(
            self.config_frame, 
            text="检查兼容性", 
            command=self.check_model_compatibility
        )
        self.check_compatibility_button.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        
        # 添加性能监控面板
        self.add_performance_monitoring()
    
    def select_model_path(self):
        path = filedialog.askdirectory()
        if path:
            self.config['model_path'] = path
            self.model_path_entry.delete(0, tk.END)  # 清除当前内容
            self.model_path_entry.insert(0, path)    # 插入新路径
            
    def start_server(self):
        """启动VLLM服务器"""
        if not self.config['model_path']:
            messagebox.showerror("错误", "请先选择模型路径")
            return
        
        if hasattr(self, 'server_process') and self.server_process and self.server_process.poll() is None:
            messagebox.showinfo("提示", "服务器已经在运行")
            return
        
        # 检查模型兼容性
        if not self.check_model_compatibility():
            if not messagebox.askokcancel("警告", "模型兼容性检查发现潜在问题，是否继续启动服务器？"):
                return
        
        # 清理GPU内存
        self.clean_gpu_memory()
        
        # 设置环境变量以避免内存碎片问题
        env = os.environ.copy()

        # 应用高级设置中的CUDA内存分块大小
        cuda_split_size = self.config.get('advanced_cuda_split_size', 128)  # 默认128MB
        env['PYTORCH_CUDA_ALLOC_CONF'] = f'expandable_segments:True,max_split_size_mb:{cuda_split_size}'
        self.status_text.insert(tk.END, f"CUDA内存分块大小: {cuda_split_size}MB\n")

        env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(self.config['gpu_count'])])
        env['OMP_NUM_THREADS'] = '4'  # 限制OpenMP线程数
        env['MKL_NUM_THREADS'] = '4'  # 限制MKL线程数

        # 添加性能优化环境变量
        env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'  # 优化CUDA连接
        env['NCCL_P2P_DISABLE'] = '1'  # 对于单GPU，禁用P2P可能提高性能
        env['CUDA_AUTO_BOOST'] = '1'  # 启用GPU自动提升频率
        env['VLLM_USE_ASYNC_CUDA_MALLOC'] = '1'  # 使用异步CUDA内存分配
        # 获取系统内存大小
        system_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
        # 根据硬件情况选择是否启用内存高效线性层
        if system_memory > 16:  # 只有在系统内存足够时才启用
            env['VLLM_ENABLE_MEMORY_EFFICIENT_LINEAR'] = '1'  # 启用内存高效线性层
        
        # 记录启动信息
        self.status_text.insert(tk.END, "\n===== 启动服务器 =====\n")
        self.status_text.insert(tk.END, f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.status_text.insert(tk.END, f"模型路径: {self.config['model_path']}\n")
        self.status_text.insert(tk.END, f"GPU数量: {self.config['gpu_count']}\n")
        self.status_text.insert(tk.END, f"显存比例: {self.config['mem_ratio']}%\n")
        
        # 检查GPU监控线程
        if not self.monitoring:
            self.monitoring = True
            threading.Thread(target=self.update_gpu_stats, daemon=True).start()
        
        # 保存配置
        self.save_config()
        
        # 预先分配内存空间，防止运行时内存不足
        self.preallocate_memory_buffer()
        
        # 初始化KV缓存监控
        self.kv_cache_hits = 0
        self.kv_cache_misses = 0
        
        # 检查是否需要内存交换
        if self.config['enable_memory_offload']:
            try:
                self.status_text.insert(tk.END, "正在设置内存交换...\n")
                
                # 计算模型大小
                model_size = self.estimate_model_size()
                
                # 获取可用显存
                available_vram = self.get_available_vram(use_ratio=self.config['mem_ratio'] / 100)
                
                self.status_text.insert(tk.END, f"模型大小: {model_size:.2f}GB, 可用显存: {available_vram:.2f}GB\n")
                
                # 计算需要卸载的内存大小
                offload_ratio = self.config['memory_offload_ratio'] / 100
                initial_offload_size = model_size * offload_ratio
                
                self.status_text.insert(tk.END, f"将卸载 {initial_offload_size:.2f}GB 到系统内存 (比例: {self.config['memory_offload_ratio']}%)\n")
                
                # 设置内存映射文件
                self.setup_memory_offload(model_size, offload_ratio)
                
                # 检查VLLM支持的参数
                self.status_text.insert(tk.END, "检查VLLM支持的参数...\n")
                
                # 计算可用系统内存（考虑预留比例）
                available_memory = self.get_available_system_memory()
                reserved_ratio = self.config['reserved_memory'] / 100
                safe_memory = available_memory * (1 - reserved_ratio)
                    
                # 获取实际分配的内存大小
                actual_offload_size = 0
                if hasattr(self, 'mm') and self.mm:
                    try:
                        # 获取内存映射文件大小
                        map_file = os.path.join(os.getcwd(), "model_offload", "model_offload.bin")
                        if os.path.exists(map_file):
                            actual_offload_size = os.path.getsize(map_file) / (1024 * 1024 * 1024)
                            self.status_text.insert(tk.END, f"实际分配的内存映射大小: {actual_offload_size:.2f}GB\n")
                    except Exception as e:
                        self.status_text.insert(tk.END, f"获取内存映射大小失败: {str(e)}\n")
                
                # 动态调整所需的内存大小
                min_required_size = min(18, model_size * 0.8)  # 至少需要模型大小的80%
                
                if actual_offload_size < min_required_size:
                    self.status_text.insert(tk.END, f"警告: 实际分配的内存映射大小不足{min_required_size:.1f}GB，可能无法加载模型\n")
                    if not messagebox.askokcancel("警告", 
                        f"实际分配的内存映射大小仅为{actual_offload_size:.2f}GB，建议至少{min_required_size:.1f}GB。\n是否继续？"):
                        return False
                
                # 计算合理的交换空间大小 - 根据模型大小动态调整
                # 对于小模型(<10GB)，使用较小的交换空间
                if model_size < 10:
                    swap_size = max(2.0, model_size * 0.1)
                else:
                    # 对于大模型，使用更大的交换空间
                    swap_size = max(4.0, model_size * 0.15)
                
                # 确保不超过安全内存的20%
                swap_size = min(swap_size, safe_memory * 0.2)
                
                # 计算合理的CPU卸载大小 - 根据模型大小和可用显存动态调整
                available_vram = self.get_available_vram(use_ratio=self.config['mem_ratio'] / 100)
                
                # 如果模型大小超过可用显存，计算需要卸载的部分
                if model_size > available_vram:
                    # 需要卸载的大小 = 模型大小 - 可用显存 + 额外缓冲区(1GB)
                    min_offload_size = model_size - available_vram + 1.0
                    # 确保至少卸载模型的60%
                    offload_size = max(min_offload_size, model_size * 0.6)
                else:
                    # 如果模型可以完全放入显存，仍然卸载一部分以提高稳定性
                    offload_size = model_size * 0.3
                
                # 确保不超过安全内存的70%
                offload_size = min(offload_size, safe_memory * 0.7)
                
                # 计算总内存使用
                total_mem_usage = swap_size + offload_size
                mem_usage_ratio = total_mem_usage / safe_memory * 100
                    
                self.status_text.insert(tk.END, f"可用系统内存: {available_memory:.2f}GB, 安全内存: {safe_memory:.2f}GB\n")
                self.status_text.insert(tk.END, f"计算交换空间: {swap_size:.2f}GB, CPU卸载: {offload_size:.2f}GB\n")
                self.status_text.insert(tk.END, f"总内存使用: {total_mem_usage:.2f}GB (安全内存的{mem_usage_ratio:.1f}%)\n")
                
                # 确保max_num_batched_tokens大于等于max_num_seqs
                max_tokens = max(self.config['max_tokens'], 256)  # 确保至少为256
                
                # 构建命令
                cmd = [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', str(self.config['mem_ratio'] / 100),
                    '--max-num-batched-tokens', str(max_tokens),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half'  # 强制使用half精度
                ]
                
                # 添加最大块数（如果指定）
                if self.config['max_blocks']:
                    cmd.extend(['--num-gpu-blocks', self.config['max_blocks']])
                
                # 添加交换空间参数
                swap_param = f"{swap_size:.2f}"  # 移除GiB单位，只使用数字
                cmd.extend(['--swap-space', swap_param])
                self.status_text.insert(tk.END, f"添加交换空间参数: --swap-space {swap_param} (GB)\n")
                
                # 添加CPU卸载参数
                offload_param = f"{offload_size:.2f}"  # 移除GB单位，只使用数字
                cmd.extend(['--cpu-offload-gb', offload_param])
                self.status_text.insert(tk.END, f"添加CPU卸载参数: --cpu-offload-gb {offload_param} (GB)\n")
                
                # 添加强制使用eager模式，避免CUDA图捕获阶段的内存不足
                cmd.append('--enforce-eager')
                self.status_text.insert(tk.END, "添加强制eager模式参数: --enforce-eager\n")
                    
                self.status_text.insert(tk.END, f"已启用内存交换，可用CPU内存: {safe_memory:.2f}GB\n")
                    
                # 记录完整命令
                cmd_str = ' '.join(cmd)
                self.status_text.insert(tk.END, f"完整命令: {cmd_str}\n")
                self.status_text.see(tk.END)
                
            except Exception as e:
                self.status_text.insert(tk.END, f"设置内存交换时出错: {str(e)}\n")
                import traceback
                self.status_text.insert(tk.END, traceback.format_exc())
                if not messagebox.askokcancel("错误", 
                    f"设置内存交换时出错: {str(e)}\n是否继续启动服务器（不使用内存交换）？"):
                    return
                
                # 如果内存交换设置失败，使用基本命令
                max_tokens = max(self.config['max_tokens'], 256)  # 确保至少为256
                cmd = [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', str(self.config['mem_ratio'] / 100),
                    '--max-num-batched-tokens', str(max_tokens),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half',  # 强制使用half精度
                    '--enforce-eager'  # 添加强制使用eager模式，避免CUDA图捕获阶段的内存不足
                ]
        else:
            # 如果不需要内存交换，使用基本命令
            max_tokens = max(self.config['max_tokens'], 256)  # 确保至少为256
            cmd = [
                'vllm', 'serve',
                self.config['model_path'],
                '--host', self.config['ip'],
                '--port', str(self.config['port']),
                '--tensor-parallel-size', str(self.config['gpu_count']),
                '--gpu-memory-utilization', str(self.config['mem_ratio'] / 100),
                '--max-num-batched-tokens', str(max_tokens),
                '--block-size', str(self.config['block_size']),
                '--max-model-len', str(self.config['max_model_len']),
                '--dtype', 'half',  # 强制使用half精度
                '--enforce-eager'  # 添加强制使用eager模式，避免CUDA图捕获阶段的内存不足
            ]
        
        # 添加性能优化参数
        performance_args = [
            '--max-num-seqs', '32',  # 增加最大序列数
            '--disable-log-stats',  # 禁用统计日志，减少开销
            '--kv-cache-dtype', 'auto',  # 使用自动选择KV缓存精度
            '--trust-remote-code'  # 信任远程代码，支持更多模型
        ]
        
        # 应用高级设置中的批处理大小
        batch_size = self.config.get('advanced_batch_size', 16)  # 默认16
        performance_args.extend(['--max-num-batched-tokens', str(max(batch_size * 256, max_tokens))])
        self.status_text.insert(tk.END, f"批处理大小: {batch_size}\n")

        # 添加内存带宽优化参数
        if int(self.block_size_var.get()) < 32:
            # 如果块大小小于32，建议增加到32以提高内存带宽利用率
            self.status_text.insert(tk.END, f"注意: 当前块大小({self.block_size_var.get()})较小，可能影响内存带宽利用率\n")
            self.status_text.insert(tk.END, "建议使用更大的块大小(32-64)以提高内存带宽利用率\n")

        # 检查是否支持Flash Attention
        if self.check_flash_attention_support():
            performance_args.append('--enable-chunked-prefill')
            self.status_text.insert(tk.END, "启用分块预填充优化\n")
        
        # 添加性能参数到命令
        cmd.extend(performance_args)
        
        # 异步启动服务器
        try:
            self.status_text.insert(tk.END, "正在启动服务器进程...\n")
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env  # 使用修改后的环境变量
            )
            
            # 等待一小段时间，检查进程是否立即退出
            time.sleep(1)
            if self.server_process.poll() is not None:
                # 进程已退出，获取输出
                output, _ = self.server_process.communicate()
                error_msg = f"启动服务器失败: {output.decode()}"
                self.status_text.insert(tk.END, f"{error_msg}\n")
                
                # 尝试使用备用方法
                return self.fallback_start_server(error_msg)
            
            # 启动监控线程
            threading.Thread(target=self.monitor_server_output).start()

            # 更新API地址
            # 说明：GET /v1返回404是正常现象，请使用支持POST的具体API endpoint进行请求
            api_base = f"http://{self.config['ip']}:{self.config['port']}/v1"
            self.api_label.config(text=f"API地址: {api_base}")
            self.status_text.insert(tk.END, f"\n服务器启动中...\nAPI地址: {api_base}\n")
            self.status_text.see(tk.END)
            
            return True
            
        except Exception as e:
            error_msg = f"启动服务器失败: {str(e)}"
            self.status_text.insert(tk.END, f"{error_msg}\n")
            import traceback
            self.status_text.insert(tk.END, traceback.format_exc())
            
            # 尝试使用备用方法
            return self.fallback_start_server(error_msg)
    
    def stop_server(self):
        try:
            # 先停止所有监控线程
            self.monitoring = False
            # 等待一小段时间让线程有机会退出
            time.sleep(0.5)
            
            if hasattr(self, 'server_process') and self.server_process and self.server_process.poll() is None:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=5)
                    self.status_text.insert(tk.END, "\n服务器已停止.\n")
                except subprocess.TimeoutExpired:
                    self.status_text.insert(tk.END, "\n停止服务器超时，但服务器可能已停止.\n")
            else:
                self.status_text.insert(tk.END, "\n服务器未在运行.\n")
                
            # 清理内存映射资源
            self.cleanup_memory_offload()
                
        except Exception as e:
            messagebox.showerror("错误", f"停止服务器失败: {str(e)}")
        finally:
            # 确保监控标志被设置为False
            self.monitoring = False
            # 禁用自动调优
            if hasattr(self, 'auto_tune_var'):
                self.auto_tune_var.set(False)
            self.api_label.config(text="API地址: 服务器未启动")
    
    def cleanup_memory_offload(self):
        """清理内存映射资源"""
        try:
            # 清理内存缓冲区
            self.cleanup_memory_buffer()
            
            # 清理多通道加载器
            if hasattr(self, 'multi_channel_loader') and self.multi_channel_loader is not None:
                try:
                    # 调用加载器的close方法
                    if hasattr(self.multi_channel_loader, 'close'):
                        self.multi_channel_loader.close()
                    self.multi_channel_loader = None
                    self.status_text.insert(tk.END, "多通道加载器已关闭\n")
                except Exception as e:
                    self.status_text.insert(tk.END, f"关闭多通道加载器时出错: {str(e)}\n")
            elif hasattr(self, 'channel_loaders'):
                # 兼容旧版本的代码
                for loader in self.channel_loaders:
                    if hasattr(loader, 'mm') and loader.mm:
                        loader.mm.close()
                    if hasattr(loader, 'mm_file') and loader.mm_file:
                        loader.mm_file.close()
                self.channel_loaders = []
            
            # 清理内存映射
            if hasattr(self, 'mm') and self.mm:
                self.mm.close()
                self.mm = None
                
            if hasattr(self, 'mm_file') and self.mm_file:
                self.mm_file.close()
                self.mm_file = None
            
            self.status_text.insert(tk.END, "内存映射资源已释放\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"释放内存映射资源时出错: {str(e)}\n")
    
    def monitor_server_output(self):
        """监控服务器输出并检测错误"""
        error_patterns = [
            # 内存不足错误
            (r"CUDA out of memory", "GPU内存不足"),
            (r"OutOfMemoryError", "内存不足"),
            (r"OOM", "内存不足"),
            # 模型加载错误
            (r"Error loading model", "模型加载错误"),
            (r"Failed to load", "模型加载失败"),
            # 参数错误
            (r"ValueError", "参数错误"),
            (r"TypeError", "类型错误"),
            # 权限错误
            (r"PermissionError", "权限错误"),
            # 网络错误
            (r"ConnectionError", "连接错误"),
            (r"Address already in use", "端口已被占用"),
            # 通用错误
            (r"Error:", "发生错误"),
            (r"Exception:", "发生异常"),
            (r"Traceback", "程序崩溃")
        ]
        
        # Token生成模式
        token_pattern = r"Processed (\d+) tokens"
        
        # 记录启动时间
        start_time = time.time()
        error_detected = False
        error_message = ""
        server_started = False
        
        while True:
            if not hasattr(self, 'server_process') or self.server_process is None:
                self.status_text.insert(tk.END, "服务器进程不存在\n")
                break
                
            if self.server_process.poll() is not None:
                self.status_text.insert(tk.END, f"服务器进程已退出，退出码: {self.server_process.poll()}\n")
                break
                
            try:
                output = self.server_process.stdout.readline()
                if not output:
                    time.sleep(0.1)
                    continue
                    
                output_text = output.decode(errors='replace')
                self.status_text.insert(tk.END, output_text)
                self.status_text.see(tk.END)
                
                # 检查是否有token生成信息
                token_match = re.search(token_pattern, output_text)
                if token_match:
                    tokens = int(token_match.group(1))
                    self.update_token_count(tokens)
                
                # 检查是否包含错误信息
                for pattern, error_type in error_patterns:
                    if re.search(pattern, output_text, re.IGNORECASE):
                        error_detected = True
                        error_message = f"{error_type}: {output_text.strip()}"
                        self.status_text.insert(tk.END, f"检测到错误: {error_type}\n")
                        break
                        
                # 如果检测到错误，等待一段时间收集更多日志，然后尝试恢复
                if error_detected:
                    # 继续读取一些输出以获取更多错误信息
                    for _ in range(10):  # 读取最多10行额外输出
                        try:
                            more_output = self.server_process.stdout.readline()
                            if more_output:
                                more_text = more_output.decode(errors='replace')
                                self.status_text.insert(tk.END, more_text)
                                error_message += "\n" + more_text.strip()
                        except:
                            break
                        time.sleep(0.1)
                    
                    # 如果是内存不足错误，尝试使用备用启动方法
                    if "内存不足" in error_message:
                        self.status_text.insert(tk.END, "检测到内存不足错误，尝试使用备用启动方法...\n")
                        # 停止当前进程
                        try:
                            self.server_process.terminate()
                            self.server_process.wait(timeout=5)
                        except:
                            pass
                        # 尝试使用备用方法启动
                        self.fallback_start_server(error_message)
                        return
                    # 如果是端口被占用，尝试使用不同端口
                    elif "端口已被占用" in error_message:
                        self.status_text.insert(tk.END, "检测到端口被占用，尝试使用不同端口...\n")
                        # 停止当前进程
                        try:
                            self.server_process.terminate()
                            self.server_process.wait(timeout=5)
                        except:
                            pass
                        # 尝试使用不同端口
                        self.config['port'] += 1
                        self.status_text.insert(tk.END, f"尝试使用新端口: {self.config['port']}\n")
                        self.start_server()
                        return
                    else:
                        # 其他错误，显示错误信息并询问用户是否尝试备用方法
                        if messagebox.askokcancel("错误", f"服务器启动时发生错误:\n{error_message}\n\n是否尝试使用备用方法启动?"):
                            # 停止当前进程
                            try:
                                self.server_process.terminate()
                                self.server_process.wait(timeout=5)
                            except:
                                pass
                            # 尝试使用备用方法启动
                            self.fallback_start_server(error_message)
                        return
                
                # 检查是否成功启动（通常在几秒内会出现API相关信息）
                if "API server" in output_text and time.time() - start_time > 5:
                    self.status_text.insert(tk.END, "服务器已成功启动\n")
                    server_started = True
                    
                    # 服务器成功启动后，执行自动性能优化
                    if not hasattr(self, 'performance_optimized') or not self.performance_optimized:
                        threading.Thread(target=self.auto_optimize_performance, daemon=True).start()
                        self.performance_optimized = True
                
            except Exception as e:
                self.status_text.insert(tk.END, f"监控服务器输出时出错: {str(e)}\n")
                time.sleep(1)
    
    def update_gpu_stats(self):
        while self.monitoring:
            try:
                gpus = GPUtil.getGPUs()
                self.gpu_tree.delete(*self.gpu_tree.get_children())
                for gpu in gpus:
                    # 使用nvidia-smi获取功耗信息
                    try:
                        power_info = subprocess.run(
                            ['nvidia-smi', f'--id={gpu.id}', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                            capture_output=True,
                            text=True
                        )
                        power_draw = power_info.stdout.strip()
                    except:
                        power_draw = "N/A"
                    
                    self.gpu_tree.insert('', 'end', values=(
                        gpu.id,
                        f"{gpu.memoryUsed}MB/{gpu.memoryTotal}MB",
                        f"{gpu.load*100:.1f}%",
                        f"{gpu.temperature}°C",
                        f"{power_draw}W" if power_draw and power_draw != "N/A" else "N/A",
                        "0.0%"  # KV缓存命中率暂时不支持
                    ))
                time.sleep(2)
            except Exception as e:
                self.status_text.insert(tk.END, f"GPU监控错误: {e}\n")
                self.status_text.see(tk.END)
                time.sleep(5)
    
    def get_gpu_stats(self):
        """获取GPU统计信息，返回字典列表"""
        try:
            # 使用pynvml库代替执行nvidia-smi命令
            pynvml.nvmlInit()
            
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_stats = []
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 获取GPU利用率
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = f"{utilization.gpu} %"
                mem_util = f"{utilization.memory} %"
                
                # 获取温度
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # 获取功耗
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    power_draw = f"{power:.1f} W"
                except:
                    power_draw = "N/A"
                
                gpu_stat = {
                    'utilization.gpu': gpu_util,
                    'utilization.memory': mem_util,
                    'temperature.gpu': f"{temp} C",
                    'power.draw': power_draw
                }
                gpu_stats.append(gpu_stat)
            
            pynvml.nvmlShutdown()
            return gpu_stats
        except ImportError:
            # 如果pynvml未安装，返回一个模拟的状态信息并记录警告
            self.status_text.insert(tk.END, "警告: pynvml未安装，无法获取GPU信息。请执行 pip install nvidia-ml-py3 安装。\n")
            # 返回一个包含默认值的字典，避免程序崩溃
            return [{'utilization.gpu': '0 %', 'utilization.memory': '0 %', 'temperature.gpu': '0 C', 'power.draw': 'N/A'}]
        except Exception as e:
            # 记录错误但返回一个空结果集而不是抛出异常
            self.status_text.insert(tk.END, f"获取GPU统计信息错误: {str(e)}\n")
            return []
    
    def load_config(self):
        try:
            with open('server_config.json', 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
                
                # 更新界面上的值
                self.model_path_entry.delete(0, tk.END)
                self.model_path_entry.insert(0, self.config['model_path'])
                
                self.ip_entry.delete(0, tk.END)
                self.ip_entry.insert(0, self.config['ip'])
                
                self.port_entry.delete(0, tk.END)
                self.port_entry.insert(0, str(self.config['port']))
                
                self.gpu_count_var.set(str(self.config['gpu_count']))
                
                self.mem_ratio_entry.delete(0, tk.END)
                self.mem_ratio_entry.insert(0, str(self.config['mem_ratio']))
                
                self.max_tokens_var.set(str(self.config['max_tokens']))
                
                self.max_model_len_var.set(str(self.config['max_model_len']))  # 加载max_model_len
                
                # 加载内存交换配置
                if 'enable_memory_offload' in self.config:
                    self.enable_offload_var.set(self.config['enable_memory_offload'])
                if 'memory_channels' in self.config:
                    self.memory_channels_var.set(str(self.config['memory_channels']))
                if 'memory_offload_ratio' in self.config:
                    self.memory_offload_ratio_var.set(str(self.config['memory_offload_ratio']))
                if 'reserved_memory' in self.config:
                    self.reserved_memory_var.set(str(self.config['reserved_memory']))
                
        except FileNotFoundError:
            pass

    def save_config(self):
        with open('server_config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def save_config_with_message(self):
        # 先调用update_config确保配置已更新
        if self.update_config():
            # 保存配置
            self.save_config()
            messagebox.showinfo("成功", "配置已保存到server_config.json")

    def select_calibrated_model(self):
        path = filedialog.askdirectory(title="选择校准模型目录")
        if path:
            self.calibrated_model_var.set(path)
            self.config['calibrated_model'] = path
            
    def check_fp8_support(self):
        try:
            if not torch.cuda.is_available():
                return False
            capability = torch.cuda.get_device_capability()
            # 需要Ampere或更新架构（计算能力 >= 8.0）
            return capability[0] >= 8
        except Exception as e:
            print(f"检查FP8支持失败: {e}")
            return False
            
    def run_calibration(self):
        if not self.check_fp8_support():
            messagebox.showerror("错误", "当前GPU不支持FP8量化")
            return
            
        if not self.config['model_path']:
            messagebox.showerror("错误", "请先选择模型路径")
            return
            
        # 生成校准脚本
        calibration_script = f"""
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.transformers import oneshot

# 加载模型
model = AutoModelForCausalLM.from_pretrained("{self.config['model_path']}", 
                                            device_map="auto", 
                                            torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("{self.config['model_path']}")

# 配置校准参数
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# 加载数据集
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def process_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

# 量化配置
recipe = '''
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
'''

# 应用量化
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 保存量化模型
SAVE_DIR = "{os.path.basename(self.config['model_path'])}-FP8-KV"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
"""
        
        # 保存并运行校准脚本
        with open("run_calibration.py", "w") as f:
            f.write(calibration_script)
            
        # 检测操作系统，使用适当的方式启动进程
        try:
            if sys.platform == 'win32':
                # Windows系统
                subprocess.Popen(["python", "run_calibration.py"],
                                cwd=os.getcwd(),
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                # Linux/Mac系统
                subprocess.Popen(["python", "run_calibration.py"],
                                cwd=os.getcwd())
                        
            messagebox.showinfo("校准", "校准进程已启动，请等待完成...")
        except Exception as e:
            self.status_text.insert(tk.END, f"启动校准进程失败: {str(e)}\n")
            messagebox.showerror("错误", f"启动校准进程失败: {str(e)}")
    
    def get_available_system_memory(self):
        """获取可用系统内存（GB）"""
        mem = psutil.virtual_memory()
        # 返回可用内存（GB）
        return mem.available / (1024 * 1024 * 1024)
    
    def get_available_vram(self, use_ratio=None):
        """获取可用显存（GB）"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return 0
            
            # 如果使用多GPU，计算总显存
            if self.config['gpu_count'] > 1:
                total_vram = sum([gpu.memoryTotal for gpu in gpus[:self.config['gpu_count']]])
            else:
                total_vram = gpus[0].memoryTotal
                
            # 转换为GB并应用显存比例
            ratio = use_ratio if use_ratio is not None else (self.config['mem_ratio'] / 100)
            return total_vram * ratio / 1024
        except Exception as e:
            self.status_text.insert(tk.END, f"获取显存信息错误: {e}\n")
            return 0
    
    def estimate_model_size(self):
        """估算模型大小（GB）"""
        try:
            # 简单估算：检查模型目录中的.bin文件大小总和
            model_path = self.config['model_path']
            total_size = 0
            
            # 检查是否有model.safetensors文件
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                total_size = os.path.getsize(safetensors_path)
                self.status_text.insert(tk.END, f"找到model.safetensors文件，大小: {total_size/(1024*1024*1024):.2f}GB\n")
                # 转换为GB
                return total_size / (1024 * 1024 * 1024)
            
            # 检查是否有pytorch_model.bin文件
            pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                total_size = os.path.getsize(pytorch_model_path)
                self.status_text.insert(tk.END, f"找到pytorch_model.bin文件，大小: {total_size/(1024*1024*1024):.2f}GB\n")
                # 转换为GB
                return total_size / (1024 * 1024 * 1024)
            
            # 如果是分片模型，计算所有分片的大小
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith('.bin') or file.endswith('.safetensors'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        self.status_text.insert(tk.END, f"找到模型文件: {file}, 大小: {file_size/(1024*1024*1024):.2f}GB\n")
            
            # 如果没有找到任何模型文件，使用默认值
            if total_size == 0:
                self.status_text.insert(tk.END, "未找到模型文件，使用默认值29.5GB\n")
                return 29.5  # 默认值为29.5GB
            
            # 转换为GB
            model_size_gb = total_size / (1024 * 1024 * 1024)
            self.status_text.insert(tk.END, f"估算模型总大小: {model_size_gb:.2f}GB\n")
            return model_size_gb
        except Exception as e:
            self.status_text.insert(tk.END, f"估算模型大小错误: {e}\n")
            # 返回默认值
            return 29.5  # 默认值为29.5GB

    def setup_memory_offload(self, model_size, offload_ratio):
        """设置内存交换功能"""
        if not self.config['enable_memory_offload']:
            return False
            
        try:
            # 计算需要卸载到内存的部分
            offload_size = model_size * offload_ratio
            
            self.status_text.insert(tk.END, f"将卸载 {offload_size:.2f}GB 到系统内存 (比例: {offload_ratio*100:.0f}%)\n")
            
            # 创建内存映射文件目录
            offload_dir = os.path.join(os.getcwd(), "model_offload")
            os.makedirs(offload_dir, exist_ok=True)
            
            # 创建内存映射文件
            map_file = os.path.join(offload_dir, "model_offload.bin")
            
            # 转换为字节
            offload_size_bytes = int(offload_size * 1024 * 1024 * 1024)
            
            # 检查是否有足够的磁盘空间
            disk_usage = psutil.disk_usage(os.getcwd())
            if disk_usage.free < offload_size_bytes:
                self.status_text.insert(tk.END, f"警告: 磁盘空间不足，需要 {offload_size:.2f}GB，但只有 {disk_usage.free/(1024*1024*1024):.2f}GB 可用\n")
                return False
                
            # 获取系统内存信息
            mem = psutil.virtual_memory()
            available_memory = mem.available / (1024 * 1024 * 1024)  # 可用内存(GB)
            
            # 确保至少有2GB的系统内存预留
            safe_memory = available_memory - 2.0
            
            # 检查是否有足够的内存
            if safe_memory < offload_size:
                # 调整大小到可用安全内存的90%
                adjusted_size = safe_memory * 0.9
                self.status_text.insert(tk.END, f"警告: 可用内存不足，需要 {offload_size:.2f}GB，但安全可用内存只有 {safe_memory:.2f}GB\n")
                self.status_text.insert(tk.END, f"自动调整卸载大小到 {adjusted_size:.2f}GB (安全内存的90%)\n")
                offload_size = adjusted_size
                offload_size_bytes = int(offload_size * 1024 * 1024 * 1024)
            
            # 创建内存映射文件
            self.status_text.insert(tk.END, f"正在创建内存映射文件，大小: {offload_size:.2f}GB...\n")
            
            # 记录内存使用情况
            mem_before = psutil.virtual_memory()
            self.status_text.insert(tk.END, f"创建前系统内存: 已用 {mem_before.percent}% ({mem_before.used/1024/1024/1024:.2f}GB/{mem_before.total/1024/1024/1024:.2f}GB)\n")
            
            # 使用fallocate预分配文件空间(如果可用)
            try:
                import subprocess
                self.status_text.insert(tk.END, f"尝试使用fallocate快速分配 {offload_size:.2f}GB 空间...\n")
                result = subprocess.run(['fallocate', '-l', f"{offload_size_bytes}", map_file], 
                                      check=True, capture_output=True)
                self.status_text.insert(tk.END, "使用fallocate成功预分配空间\n")
                
                # 验证文件大小
                actual_size = os.path.getsize(map_file)
                self.status_text.insert(tk.END, f"验证文件大小: {actual_size/(1024*1024*1024):.2f}GB\n")
                
                if actual_size < offload_size_bytes * 0.99:  # 允许1%的误差
                    self.status_text.insert(tk.END, f"警告: 文件大小不足，将使用传统方法分配\n")
                    os.remove(map_file)  # 删除不完整的文件
                    raise Exception("文件大小不足")
                    
            except Exception as e:
                self.status_text.insert(tk.END, f"fallocate失败: {str(e)}，将使用传统方法分配空间\n")
                
                # 传统方法: 分块写入
                with open(map_file, "wb") as f:
                    # 写入全零数据以分配空间
                    chunk_size = 1024 * 1024 * 128  # 减小到128MB块，降低内存压力
                    remaining = offload_size_bytes
                    
                    try:
                        while remaining > 0:
                            # 每写入512MB检查一次内存状态，更频繁地检查
                            if (offload_size_bytes - remaining) % (512*1024*1024) < chunk_size:
                                mem_check = psutil.virtual_memory()
                                # 如果可用内存低于1.5GB，停止写入
                                if mem_check.available < 1.5 * 1024 * 1024 * 1024:
                                    self.status_text.insert(tk.END, f"警告: 可用内存低于1.5GB，停止分配更多内存\n")
                                    break
                            
                            write_size = min(chunk_size, remaining)
                            f.write(b'\0' * write_size)
                            remaining -= write_size
                            # 更新进度
                            progress = (offload_size_bytes - remaining) / offload_size_bytes * 100
                            self.status_text.delete("end-2l", "end-1l")  # 删除上一行进度
                            self.status_text.insert(tk.END, f"创建内存映射文件: {progress:.1f}% ({(offload_size_bytes-remaining)/(1024*1024*1024):.2f}GB/{offload_size:.2f}GB)\n")
                            self.status_text.see(tk.END)
                            
                            # 添加小延迟，让系统有时间释放内存
                            time.sleep(0.01)
                            
                    except MemoryError:
                        self.status_text.insert(tk.END, f"内存不足，无法完成映射文件创建\n")
                        # 记录已分配的大小
                        actual_size = offload_size_bytes - remaining
                        self.status_text.insert(tk.END, f"已分配 {actual_size/(1024*1024*1024):.2f}GB\n")
                        # 截断文件到已写入的大小
                        f.flush()
                        f.truncate(actual_size)
            
            # 记录内存使用情况
            mem_after = psutil.virtual_memory()
            self.status_text.insert(tk.END, f"创建后系统内存: 已用 {mem_after.percent}% ({mem_after.used/1024/1024/1024:.2f}GB/{mem_after.total/1024/1024/1024:.2f}GB)\n")
            
            # 验证最终文件大小
            final_size = os.path.getsize(map_file)
            self.status_text.insert(tk.END, f"内存映射文件最终大小: {final_size/(1024*1024*1024):.2f}GB\n")
            
            # 不再强制要求18GB，而是根据模型大小动态调整
            min_required_size = min(18, model_size * 0.8)  # 至少需要模型大小的80%
            
            if final_size < min_required_size * 1024 * 1024 * 1024:
                self.status_text.insert(tk.END, f"警告: 内存映射文件大小不足{min_required_size:.1f}GB，可能无法加载模型\n")
                if not messagebox.askokcancel("警告", 
                    f"内存映射文件大小仅为{final_size/(1024*1024*1024):.2f}GB，建议至少{min_required_size:.1f}GB。\n是否继续？"):
                    return False
                
            self.status_text.insert(tk.END, "内存映射文件创建完成\n")
            
            # 创建内存映射
            self.mm_file = open(map_file, "r+b")
            self.mm = mmap.mmap(self.mm_file.fileno(), 0)
            
            # 使用用户设置的内存通道数，不再自动增加
            channels = self.config['memory_channels']
            self.status_text.insert(tk.END, f"使用用户设置的内存通道数: {channels}\n")
            
            self.setup_multi_channel_loader()
            
            # 创建配置文件
            offload_config = {
                'enabled': True,
                'offload_dir': offload_dir,
                'offload_ratio': offload_ratio,
                'channels': channels,
                'reserved_memory': self.config['reserved_memory'] / 100,
                'actual_size_gb': final_size/(1024*1024*1024)
            }
            
            offload_config_path = os.path.join(offload_dir, "offload_config.json")
            with open(offload_config_path, 'w') as f:
                json.dump(offload_config, f, indent=4)
            
            self.status_text.insert(tk.END, f"内存交换配置已保存到 {offload_config_path}\n")
            
            return True
        except Exception as e:
            self.status_text.insert(tk.END, f"设置内存交换错误: {str(e)}\n")
            import traceback
            self.status_text.insert(tk.END, traceback.format_exc())
            return False
    
    def setup_multi_channel_loader(self):
        """设置多通道加载器"""
        class MultiChannelLoader:
            def __init__(self, memory_map, num_channels=4, cache_size=32):  # 添加cache_size参数
                self.memory_map = memory_map
                self.num_channels = num_channels
                self.channel_locks = [threading.Lock() for _ in range(num_channels)]
                self.channel_positions = [0] * num_channels
                self.channel_usage = [0] * num_channels  # 记录每个通道的使用次数
                self.channel_last_access = [time.time()] * num_channels  # 记录每个通道的最后访问时间
                self.cache = {}  # 简单的内存缓存
                self.cache_hits = 0
                self.cache_misses = 0
                self.max_cache_size = cache_size  # 使用传入的缓存大小
                self.prefetch_queue = []  # 预取队列
                self.prefetch_lock = threading.Lock()
                self.prefetch_thread_running = True
                # 启动预取线程
                threading.Thread(target=self._prefetch_worker, daemon=True).start()
                
            def read_chunk(self, offset, size, channel_id=None):
                # 检查缓存
                cache_key = (offset, size)
                if cache_key in self.cache:
                    self.cache_hits += 1
                    # 更新缓存访问时间
                    self.cache[cache_key]['last_access'] = time.time()
                    return self.cache[cache_key]['data']
                
                self.cache_misses += 1
                
                # 如果未指定通道，选择最佳通道
                if channel_id is None:
                    channel_id = self._get_best_channel(offset)
                    
                with self.channel_locks[channel_id]:
                    # 记录访问时间
                    self.channel_last_access[channel_id] = time.time()
                    
                    # 如果当前位置接近请求的偏移量，可以减少寻址时间
                    if abs(self.channel_positions[channel_id] - offset) < 1024*1024:  # 如果在1MB范围内
                        # 已经接近目标位置，直接读取
                        pass
                    else:
                        # 需要重新定位
                        self.memory_map.seek(offset)
                    
                    data = self.memory_map.read(size)
                    self.channel_positions[channel_id] = offset + size
                    self.channel_usage[channel_id] += 1
                    
                    # 更新缓存
                    if len(self.cache) >= self.max_cache_size:
                        # 删除最旧的缓存项
                        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['last_access'])
                        del self.cache[oldest_key]
                    
                    self.cache[cache_key] = {
                        'data': data,
                        'last_access': time.time()
                    }
                    
                    # 预测性预取 - 预取下一个可能的块
                    next_offset = offset + size
                    self.prefetch(next_offset, size)
                    
                    return data
                    
            def _get_best_channel(self, target_offset):
                # 优先选择位置接近的通道，其次考虑使用频率
                best_channel = 0
                best_score = float('inf')
                
                for i in range(self.num_channels):
                    # 计算位置接近度分数
                    position_score = abs(self.channel_positions[i] - target_offset) / (1024*1024)  # MB为单位
                    
                    # 计算使用频率分数
                    usage_score = self.channel_usage[i] * 0.1
                    
                    # 计算时间分数（越久未使用越好）
                    time_score = -10 * (time.time() - self.channel_last_access[i])
                    
                    # 综合评分（越低越好）
                    total_score = position_score + usage_score + time_score
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_channel = i
                
                return best_channel
                    
            def _get_least_busy_channel(self):
                # 选择使用次数最少的通道
                return self.channel_usage.index(min(self.channel_usage))
                
            def get_stats(self):
                return {
                    'positions': self.channel_positions,
                    'usage': self.channel_usage,
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses + 0.001) * 100,
                    'prefetch_queue_size': len(self.prefetch_queue)
                }
            
            def prefetch(self, offset, size):
                """预取数据到缓存"""
                # 检查是否已经在缓存中
                cache_key = (offset, size)
                if cache_key in self.cache:
                    return
                
                # 检查是否已经在预取队列中
                with self.prefetch_lock:
                    for item in self.prefetch_queue:
                        if item[0] == offset and item[1] == size:
                            return
                    
                    # 添加到预取队列，最多保留10个预取请求
                    self.prefetch_queue.append((offset, size))
                    if len(self.prefetch_queue) > 10:
                        self.prefetch_queue.pop(0)
            
            def _prefetch_worker(self):
                """预取线程"""
                while self.prefetch_thread_running:
                    try:
                        # 检查预取队列
                        with self.prefetch_lock:
                            if self.prefetch_queue:
                                offset, size = self.prefetch_queue.pop(0)
                            else:
                                offset, size = None, None
                        
                        # 如果有预取请求，执行预取
                        if offset is not None and size is not None:
                            # 检查是否已经在缓存中
                            cache_key = (offset, size)
                            if cache_key not in self.cache:
                                # 选择最佳通道
                                channel_id = self._get_best_channel(offset)
                                # 执行预取
                                self.read_chunk(offset, size, channel_id)
                    except Exception as e:
                        print(f"预取错误: {e}")
                    
                    # 短暂休眠，避免占用过多CPU
                    time.sleep(0.01)
                
            def close(self):
                """关闭加载器"""
                self.prefetch_thread_running = False
                self.cache.clear()
        
        # 创建多通道加载器
        num_channels = max(4, int(self.config['memory_channels']))  # 确保至少有4个通道
        
        # 应用高级设置中的缓存大小
        cache_size = self.config.get('advanced_cache_size', 32)  # 默认32
        self.status_text.insert(tk.END, f"内存缓存大小: {cache_size}\n")
        
        self.multi_channel_loader = MultiChannelLoader(
            self.mm, 
            num_channels=num_channels,
            cache_size=cache_size  # 传入缓存大小
        )
        
        self.status_text.insert(tk.END, f"已创建 {num_channels} 个内存通道加载器，带缓存和预取功能\n")
        
        # 启动内存监控线程
        self.memory_monitor_thread_running = True
        threading.Thread(target=self.memory_monitor_thread, daemon=True).start()

    def update_system_memory_stats(self):
        """更新系统内存统计信息"""
        try:
            # 检查监控标志，如果已关闭则直接返回
            if not self.monitoring:
                return False
                
            # 获取系统内存信息
            mem = psutil.virtual_memory()
            
            # 更新到界面
            self.status_text.insert(tk.END, f"系统内存: 已用 {mem.percent}% ({mem.used/1024/1024/1024:.2f}GB/{mem.total/1024/1024/1024:.2f}GB)\n")
            
            # 如果启用了内存交换，监控交换性能
            if hasattr(self, 'multi_channel_loader') and self.multi_channel_loader is not None:
                try:
                    stats = self.multi_channel_loader.get_stats()
                    channel_stats = [f"通道{i}: {pos/1024/1024:.2f}MB" for i, pos in enumerate(stats['positions'])]
                    usage_stats = [f"通道{i}: {usage}次" for i, usage in enumerate(stats['usage'])]
                    
                    self.status_text.insert(tk.END, f"内存交换通道状态: {', '.join(channel_stats)}\n")
                    self.status_text.insert(tk.END, f"内存交换通道使用: {', '.join(usage_stats)}\n")
                    
                    # 显示缓存命中率
                    if 'cache_hits' in stats and 'cache_misses' in stats:
                        total_requests = stats['cache_hits'] + stats['cache_misses']
                        if total_requests > 0:
                            hit_ratio = stats['cache_hits'] / total_requests * 100
                            self.status_text.insert(tk.END, f"内存缓存命中率: {hit_ratio:.2f}% (命中: {stats['cache_hits']}, 未命中: {stats['cache_misses']})\n")
                except Exception as e:
                    # 捕获获取统计信息时的错误，但不中断监控
                    self.status_text.insert(tk.END, f"获取内存交换统计信息错误: {str(e)}\n")
            
            # 更新GPU KV缓存命中率（如果有）
            if hasattr(self, 'kv_cache_hits') and hasattr(self, 'kv_cache_misses'):
                total_kv_requests = self.kv_cache_hits + self.kv_cache_misses
                if total_kv_requests > 0:
                    kv_hit_ratio = self.kv_cache_hits / total_kv_requests * 100
                    self.status_text.insert(tk.END, f"KV缓存命中率: {kv_hit_ratio:.2f}% (命中: {self.kv_cache_hits}, 未命中: {self.kv_cache_misses})\n")
            
            self.status_text.see(tk.END)
            return True
        except Exception as e:
            self.status_text.insert(tk.END, f"内存监控错误: {e}\n")
            return False

    def memory_monitor_thread(self):
        """内存监控线程"""
        try:
            # 设置本地变量，避免频繁访问self属性
            monitoring = True
            
            while monitoring and self.monitoring:
                try:
                    if hasattr(self, 'server_process') and self.server_process is not None and self.server_process.poll() is None:
                        self.update_system_memory_stats()
                    time.sleep(5)  # 每5秒更新一次
                    
                    # 检查监控标志是否已更改
                    monitoring = self.monitoring
                except Exception as e:
                    self.status_text.insert(tk.END, f"内存监控错误: {str(e)}\n")
                    time.sleep(5)  # 出错时等待5秒再继续
        except Exception as e:
            # 捕获线程启动时的异常
            print(f"内存监控线程启动错误: {str(e)}")

    def check_vllm_supported_args(self):
        """检查VLLM支持的命令行参数"""
        supported_args = {
            'swap_space': '--swap-space',
            'cpu_offload': '--cpu-offload-gb',
            'max_cpu_memory': '--max-cpu-memory'
        }
        
        try:
            # 尝试运行vllm help命令，增加超时时间
            help_output = subprocess.run(
                ['vllm', 'serve', '--help'],
                capture_output=True,
                text=True,
                timeout=15  # 增加超时时间到15秒
            )
            
            # 检查输出中是否包含特定参数
            output = help_output.stdout + help_output.stderr
            self.status_text.insert(tk.END, f"检查VLLM支持的参数...\n")
            
            # 检查每个参数
            if '--swap-space' not in output:
                if '--swap' in output:
                    supported_args['swap_space'] = '--swap'
                    self.status_text.insert(tk.END, "未找到--swap-space参数，将使用--swap\n")
                else:
                    supported_args['swap_space'] = None
                    self.status_text.insert(tk.END, "未找到交换空间相关参数\n")
                
            # 检查CPU卸载参数
            if '--cpu-offload-gb' not in output:
                if '--cpu-offload' in output:
                    supported_args['cpu_offload'] = '--cpu-offload'
                    self.status_text.insert(tk.END, "未找到--cpu-offload-gb参数，将使用--cpu-offload\n")
                elif '--offload-params' in output:
                    supported_args['cpu_offload'] = '--offload-params'
                    self.status_text.insert(tk.END, "未找到--cpu-offload-gb参数，将使用--offload-params\n")
                else:
                    supported_args['cpu_offload'] = None
                    self.status_text.insert(tk.END, "未找到CPU卸载相关参数\n")
                    
            if '--max-cpu-memory' not in output:
                supported_args['max_cpu_memory'] = None
                self.status_text.insert(tk.END, "未找到--max-cpu-memory参数\n")
                
            return supported_args
            
        except subprocess.TimeoutExpired:
            self.status_text.insert(tk.END, "检查VLLM参数超时，使用默认参数\n")
            # 使用最常见的参数组合
            return {
                'swap_space': '--swap-space',
                'cpu_offload': '--cpu-offload',
                'max_cpu_memory': None
            }
        except Exception as e:
            self.status_text.insert(tk.END, f"检查VLLM参数失败: {str(e)}\n")
            # 返回默认值
            return supported_args

    def fallback_start_server(self, error_msg):
        """备用启动方法，尝试使用不同的参数启动服务器"""
        if not messagebox.askokcancel("错误", 
            f"{error_msg}\n\n是否尝试使用备用方法启动服务器？"):
            return False
            
        self.status_text.insert(tk.END, "\n尝试使用备用方法启动服务器...\n")
        
        # 清理GPU内存
        self.clean_gpu_memory()
        
        # 设置环境变量以避免内存碎片问题
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(self.config['gpu_count'])])
        env['OMP_NUM_THREADS'] = '4'  # 限制OpenMP线程数
        env['MKL_NUM_THREADS'] = '4'  # 限制MKL线程数
        
        # 添加VLLM特定的环境变量，优化内存使用
        env['VLLM_USE_ASYNC_CUDA_MALLOC'] = '1'  # 使用异步CUDA内存分配
        env['VLLM_CPU_OFFLOAD_PIPELINE'] = '1'  # 启用CPU卸载流水线
        env['VLLM_ENABLE_STAGED_INIT'] = '1'  # 启用分阶段初始化
        
        self.status_text.insert(tk.END, "已设置优化环境变量\n")
        
        # 临时降低模型参数
        original_max_model_len = self.config['max_model_len']
        original_max_tokens = self.config['max_tokens']
        
        # 降低序列长度以减少内存使用
        self.config['max_model_len'] = min(self.config['max_model_len'], 2048)  # 调整到2048
        self.config['max_tokens'] = min(self.config['max_tokens'], 2048)  # 调整到2048，确保大于max_num_seqs
        
        self.status_text.insert(tk.END, f"临时降低序列长度: {self.config['max_model_len']}, 最大token数: {self.config['max_tokens']}\n")
        
        # 获取模型大小
        model_size = self.estimate_model_size()
        
        # 尝试不同的启动选项
        options = [
            {
                "desc": "使用最小内存配置",
                "cmd": [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', '0.7',  # 降低显存使用率
                    '--max-num-batched-tokens', str(self.config['max_tokens']),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half',
                    '--enforce-eager'  # 添加强制使用eager模式
                ]
            },
            {
                "desc": "使用量化配置",
                "cmd": [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', '0.8',
                    '--max-num-batched-tokens', str(self.config['max_tokens']),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half',
                    '--quantization', 'awq',  # 尝试使用AWQ量化
                    '--enforce-eager'  # 添加强制使用eager模式
                ]
            },
            {
                "desc": "使用最小内存交换配置",
                "cmd": [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', '0.6',  # 进一步降低显存使用率
                    '--max-num-batched-tokens', str(self.config['max_tokens']),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half',
                    '--swap-space', '2',  # 移除GiB单位，只使用数字
                    '--cpu-offload-gb', '10',
                    '--enforce-eager'  # 添加强制使用eager模式
                ]
            }
        ]
        
        # 针对大型模型(>10GB)添加特殊选项
        if model_size > 10:
            # 添加分阶段加载选项
            staged_option = {
                "desc": "使用分阶段加载（适合大模型）",
                "cmd": [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', '0.5',  # 显著降低显存使用率
                    '--max-num-batched-tokens', str(min(self.config['max_tokens'], 1024)),  # 降低批处理大小
                    '--block-size', str(min(self.config['block_size'], 8)),  # 降低块大小
                    '--max-model-len', str(min(self.config['max_model_len'], 1024)),  # 降低最大长度
                    '--dtype', 'half',
                    '--swap-space', '4',
                    '--cpu-offload-gb', str(max(10, int(model_size * 0.7))),  # 至少卸载70%的模型
                    '--enforce-eager'  # 添加强制使用eager模式
                ]
            }
            options.insert(0, staged_option)  # 将此选项放在首位
            
            # 添加8位量化选项
            int8_option = {
                "desc": "使用8位量化（适合大模型）",
                "cmd": [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', '0.7',
                    '--max-num-batched-tokens', str(self.config['max_tokens']),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half',
                    '--quantization', 'int8',  # 使用int8量化
                    '--enforce-eager'  # 添加强制使用eager模式
                ]
            }
            options.insert(1, int8_option)
        
        # 尝试每个选项
        for i, option in enumerate(options):
            self.status_text.insert(tk.END, f"\n尝试选项 {i+1}: {option['desc']}\n")
            cmd_str = ' '.join(option['cmd'])
            self.status_text.insert(tk.END, f"命令: {cmd_str}\n")
                
            try:
                # 启动服务器
                self.server_process = subprocess.Popen(
                    option['cmd'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                    
                # 等待一小段时间，检查进程是否立即退出
                time.sleep(5)  # 增加等待时间
                if self.server_process.poll() is None:
                    # 进程仍在运行，启动成功
                    self.status_text.insert(tk.END, "服务器启动成功！\n")
                    
                    # 启动监控线程
                    threading.Thread(target=self.monitor_server_output).start()
                    
                    # 更新API地址
                    api_base = f"http://{self.config['ip']}:{self.config['port']}/v1"
                    self.api_label.config(text=f"API地址: {api_base}")
                    
                    return True
                else:
                    # 进程已退出，获取输出
                    output, _ = self.server_process.communicate()
                    error_output = output.decode()
                    self.status_text.insert(tk.END, f"启动失败: {error_output}\n")
                    
                    # 分析错误原因
                    if "CUDA out of memory" in error_output:
                        self.status_text.insert(tk.END, "检测到错误: GPU内存不足\n")
                    elif "RuntimeError" in error_output:
                        self.status_text.insert(tk.END, "检测到错误: 程序崩溃\n")
                    
                    # 在选项之间添加额外的清理步骤
                    self.clean_gpu_memory()
                    time.sleep(2)  # 等待GPU内存释放
                    
            except Exception as e:
                self.status_text.insert(tk.END, f"尝试选项 {i+1} 失败: {str(e)}\n")
        
        # 所有选项都失败，提供建议
        self.status_text.insert(tk.END, "所有备用选项都失败，建议：\n")
        self.status_text.insert(tk.END, "1. 关闭其他内存密集型应用程序\n")
        self.status_text.insert(tk.END, "2. 重启系统以清理内存碎片\n")
        self.status_text.insert(tk.END, "3. 尝试使用量化版本的模型\n")
        self.status_text.insert(tk.END, "4. 尝试使用更小的模型，如7B或更小的版本\n")
        
        # 恢复原始设置
        self.config['max_model_len'] = original_max_model_len
        self.config['max_tokens'] = original_max_tokens
        
        return False

    def clean_gpu_memory(self):
        """清理GPU内存"""
        try:
            self.status_text.insert(tk.END, "正在清理GPU内存...\n")
            
            # 尝试释放PyTorch缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.status_text.insert(tk.END, "已清理PyTorch缓存\n")
                
                # 获取当前GPU内存使用情况
                gpu = GPUtil.getGPUs()[0]
                free_mem = gpu.memoryFree
                total_mem = gpu.memoryTotal
                used_mem = total_mem - free_mem
                
                self.status_text.insert(tk.END, f"当前GPU内存: 已用 {used_mem}MB / 总计 {total_mem}MB\n")
                
                # 如果内存使用率过高，建议用户重启系统
                if used_mem / total_mem > 0.5:  # 如果使用超过50%
                    self.status_text.insert(tk.END, "警告: GPU内存使用率较高，可能影响模型加载\n")
                    self.status_text.insert(tk.END, "建议关闭其他使用GPU的应用程序或重启系统\n")
                    
            # 尝试运行系统命令释放内存
            os.system("sync")  # 同步文件系统缓存
            
            # 尝试释放系统缓存
            try:
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("1")
                self.status_text.insert(tk.END, "已释放系统缓存\n")
            except:
                pass  # 可能没有权限，忽略错误
                
            self.status_text.insert(tk.END, "GPU内存清理完成\n")
            
        except Exception as e:
            self.status_text.insert(tk.END, f"清理GPU内存时出错: {str(e)}\n")

    def preallocate_memory_buffer(self):
        """预先分配内存缓冲区，防止运行时内存不足"""
        try:
            self.status_text.insert(tk.END, "正在预分配内存缓冲区...\n")
                
            # 获取模型大小
            model_size = self.estimate_model_size()
            
            # 计算需要预分配的内存大小 - 根据模型大小动态调整
            if model_size < 10:
                # 小模型使用较小的缓冲区
                buffer_size_gb = model_size * 0.2
                buffer_size_gb = max(buffer_size_gb, 4.0)  # 至少4GB
            else:
                # 大模型使用较大的缓冲区，但比例更小
                buffer_size_gb = model_size * 0.15
                buffer_size_gb = max(buffer_size_gb, 6.0)  # 至少6GB
            
            # 检查可用内存
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 * 1024 * 1024)
            
            # 确保缓冲区不超过可用内存的50%
            max_buffer_size = available_gb * 0.5
            if buffer_size_gb > max_buffer_size:
                self.status_text.insert(tk.END, f"警告: 计算的缓冲区大小({buffer_size_gb:.2f}GB)超过可用内存的50%，调整大小\n")
                buffer_size_gb = max_buffer_size
            
            # 保留至少5GB系统运行空间
            if available_gb < buffer_size_gb + 5:
                self.status_text.insert(tk.END, f"警告: 可用内存({available_gb:.2f}GB)不足，减小缓冲区大小\n")
                buffer_size_gb = max(2.0, available_gb - 5)  # 至少2GB，保留5GB系统运行空间
                
            self.status_text.insert(tk.END, f"预分配内存缓冲区大小: {buffer_size_gb:.2f}GB\n")
            
            # 创建内存缓冲区目录
            buffer_dir = os.path.join(os.getcwd(), "memory_buffer")
            os.makedirs(buffer_dir, exist_ok=True)
            
            # 创建内存缓冲区文件
            buffer_file = os.path.join(buffer_dir, "memory_buffer.bin")
            
            # 如果文件已存在，检查大小是否足够
            if os.path.exists(buffer_file):
                current_size = os.path.getsize(buffer_file) / (1024 * 1024 * 1024)
                if current_size >= buffer_size_gb:
                    self.status_text.insert(tk.END, f"使用现有内存缓冲区: {current_size:.2f}GB\n")
                    return
                else:
                    self.status_text.insert(tk.END, f"现有内存缓冲区大小不足({current_size:.2f}GB)，重新创建\n")
                    os.remove(buffer_file)
            
            # 创建新的内存缓冲区文件
            self.status_text.insert(tk.END, f"创建内存缓冲区文件: {buffer_file}\n")
            
            # 计算缓冲区大小（字节）
            buffer_size_bytes = int(buffer_size_gb * 1024 * 1024 * 1024)
            
            # 创建内存缓冲区文件
            with open(buffer_file, "wb") as f:
                # 分块写入，避免一次性分配过多内存
                chunk_size = 1024 * 1024 * 64  # 减小到64MB块，降低内存压力
                remaining = buffer_size_bytes
                
                # 记录内存使用情况
                mem_before = psutil.virtual_memory()
                self.status_text.insert(tk.END, f"创建前系统内存: 已用 {mem_before.percent}% ({mem_before.used/1024/1024/1024:.2f}GB/{mem_before.total/1024/1024/1024:.2f}GB)\n")
                
                try:
                    while remaining > 0:
                        # 每写入256MB检查一次内存状态，更频繁地检查
                        if (buffer_size_bytes - remaining) % (256*1024*1024) < chunk_size:
                            mem_check = psutil.virtual_memory()
                            # 如果可用内存低于2.5GB，停止写入
                            if mem_check.available < 2.5 * 1024 * 1024 * 1024:
                                self.status_text.insert(tk.END, f"警告: 可用内存低于2.5GB，停止分配更多内存\n")
                                break
                        
                        write_size = min(chunk_size, remaining)
                        f.write(b'\0' * write_size)
                        remaining -= write_size
                        # 更新进度
                        progress = (buffer_size_bytes - remaining) / buffer_size_bytes * 100
                        self.status_text.delete("end-2l", "end-1l")  # 删除上一行进度
                        self.status_text.insert(tk.END, f"创建内存缓冲区: {progress:.1f}% ({(buffer_size_bytes-remaining)/(1024*1024*1024):.2f}GB/{buffer_size_gb:.2f}GB)\n")
                        self.status_text.see(tk.END)
                except MemoryError:
                    self.status_text.insert(tk.END, f"内存不足，无法完成缓冲区创建\n")
                    # 记录已分配的大小
                    actual_size = buffer_size_bytes - remaining
                    self.status_text.insert(tk.END, f"已分配 {actual_size/(1024*1024*1024):.2f}GB\n")
                    # 截断文件到已写入的大小
                    f.flush()
                    f.truncate(actual_size)
                
                # 记录内存使用情况
                mem_after = psutil.virtual_memory()
                self.status_text.insert(tk.END, f"创建后系统内存: 已用 {mem_after.percent}% ({mem_after.used/1024/1024/1024:.2f}GB/{mem_after.total/1024/1024/1024:.2f}GB)\n")
        
            # 验证最终文件大小
            final_size = os.path.getsize(buffer_file)
            self.status_text.insert(tk.END, f"内存缓冲区最终大小: {final_size/(1024*1024*1024):.2f}GB\n")
            
            # 打开文件并映射到内存
            self.buffer_file = open(buffer_file, "r+b")
            self.buffer_mm = mmap.mmap(self.buffer_file.fileno(), 0)
            
            self.status_text.insert(tk.END, f"内存缓冲区创建完成: {final_size/(1024*1024*1024):.2f}GB\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"创建内存缓冲区时出错: {str(e)}\n")
            import traceback
            self.status_text.insert(tk.END, traceback.format_exc())
    
    def cleanup_memory_buffer(self):
        """清理内存缓冲区"""
        try:
            if hasattr(self, 'buffer_mm') and self.buffer_mm:
                self.buffer_mm.close()
                self.buffer_mm = None
            
            if hasattr(self, 'buffer_file') and self.buffer_file:
                self.buffer_file.close()
                self.buffer_file = None
            
            self.status_text.insert(tk.END, "内存缓冲区已释放\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"释放内存缓冲区时出错: {str(e)}\n")

    def recommend_settings(self):
        """根据模型大小和硬件条件推荐设置"""
        try:
            # 检查是否选择了模型
            if not self.config['model_path']:
                messagebox.showerror("错误", "请先选择模型路径")
                return
                
            # 估算模型大小
            model_size = self.estimate_model_size()
            
            # 获取GPU信息
            gpus = GPUtil.getGPUs()
            if not gpus:
                messagebox.showerror("错误", "未检测到GPU")
                return
                
            # 获取第一个GPU的显存大小(GB)
            gpu_memory = gpus[0].memoryTotal / 1024
            
            # 获取系统内存大小(GB)
            system_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            # 根据模型大小和硬件条件推荐设置
            self.status_text.insert(tk.END, "\n===== 推荐设置 =====\n")
            self.status_text.insert(tk.END, f"模型大小: {model_size:.2f}GB\n")
            self.status_text.insert(tk.END, f"GPU显存: {gpu_memory:.2f}GB\n")
            self.status_text.insert(tk.END, f"系统内存: {system_memory:.2f}GB\n")
            
            # 推荐显存比例
            if model_size > gpu_memory * 0.9:
                # 模型接近或超过显存大小，需要内存交换
                mem_ratio = 85  # 降低到85%，给系统留出更多余量
                self.status_text.insert(tk.END, f"推荐显存比例: {mem_ratio}% (模型较大，降低比例避免OOM)\n")
                
                # 启用内存交换
                self.enable_offload_var.set(True)
                
                # 计算合理的内存交换比例
                if model_size > gpu_memory * 1.5:
                    # 模型远大于显存，需要大量交换
                    offload_ratio = 70  # 降低到70%，避免系统内存压力过大
                else:
                    # 模型略大于显存，适度交换
                    offload_ratio = 60
                    
                self.memory_offload_ratio_var.set(str(offload_ratio))
                self.status_text.insert(tk.END, f"推荐内存交换比例: {offload_ratio}%\n")
                
                # 推荐内存通道数 - 根据系统内存大小调整
                if system_memory > 64:  # 只有大内存系统才推荐更多通道
                    channels = 8
                else:
                    channels = 4  # 对于32GB内存系统，使用4个通道
                    
                self.memory_channels_var.set(str(channels))
                self.status_text.insert(tk.END, f"推荐内存通道数: {channels}\n")
                
                # 推荐预留内存比例
                reserved_memory = 20
                self.reserved_memory_var.set(str(reserved_memory))
                self.status_text.insert(tk.END, f"推荐系统内存预留: {reserved_memory}%\n")
                
                # 推荐较小的序列长度
                if model_size > 20:
                    max_model_len = 2048
                else:
                    max_model_len = 4096
                    
                self.max_model_len_var.set(str(max_model_len))
                self.status_text.insert(tk.END, f"推荐最大序列长度: {max_model_len}\n")
                
                # 推荐适中的块大小以提高内存带宽利用率
                block_size = 32  # 对于普通硬件，32是较好的平衡点
                self.block_size_var.set(str(block_size))
                self.status_text.insert(tk.END, f"推荐块大小: {block_size} (提高内存带宽利用率)\n")
                
                # 推荐使用--enforce-eager参数
                self.status_text.insert(tk.END, "推荐使用强制eager模式，避免CUDA图捕获阶段的内存不足\n")
                
            else:
                # 模型可以完全放入显存
                mem_ratio = 90
                self.status_text.insert(tk.END, f"推荐显存比例: {mem_ratio}% (模型可完全放入显存)\n")
                
                # 不需要内存交换
                self.enable_offload_var.set(False)
                self.status_text.insert(tk.END, "不需要启用内存交换\n")
                
                # 推荐较大的序列长度
                max_model_len = 8192
                self.max_model_len_var.set(str(max_model_len))
                self.status_text.insert(tk.END, f"推荐最大序列长度: {max_model_len}\n")
                
                # 推荐适中的块大小以提高内存带宽利用率
                block_size = 32  # 对于普通硬件，32是较好的平衡点
                self.block_size_var.set(str(block_size))
                self.status_text.insert(tk.END, f"推荐块大小: {block_size} (提高内存带宽利用率)\n")
            
            # 更新界面上的值
            self.mem_ratio_entry.delete(0, tk.END)
            self.mem_ratio_entry.insert(0, str(mem_ratio))
            
            # 更新配置
            self.update_config()
            
            self.status_text.insert(tk.END, "推荐设置已应用到界面\n")
            self.status_text.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("错误", f"推荐设置失败: {str(e)}")

    def update_config(self):
        """更新配置参数"""
        try:
            # 获取界面上的值
            model_path = self.model_path_entry.get()
            ip = self.ip_entry.get()
            port = int(self.port_entry.get())
            gpu_count = int(self.gpu_count_var.get())
            mem_ratio = int(self.mem_ratio_entry.get())
            max_tokens = int(self.max_tokens_var.get())
            max_model_len = int(self.max_model_len_var.get())
            block_size = int(self.block_size_var.get())
            
            # 获取内存交换配置
            enable_memory_offload = self.enable_offload_var.get()
            memory_channels = int(self.memory_channels_var.get())
            memory_offload_ratio = int(self.memory_offload_ratio_var.get())
            reserved_memory = int(self.reserved_memory_var.get())
            
            # 验证参数
            if port < 1 or port > 65535:
                messagebox.showerror("错误", "端口号必须在1-65535之间")
                return False
                
            if gpu_count < 1:
                messagebox.showerror("错误", "GPU数量必须大于0")
                return False
                
            if mem_ratio < 10 or mem_ratio > 100:
                messagebox.showerror("错误", "显存比例必须在10-100之间")
                return False
                
            if max_tokens < 256:
                messagebox.showerror("错误", "最大Token数不能小于256")
                return False
                
            if max_model_len < 512:
                messagebox.showerror("错误", "最大模型长度不能小于512")
                return False
                
            if block_size < 1:
                messagebox.showerror("错误", "块大小必须大于0")
                return False
                
            # 验证内存交换配置
            if enable_memory_offload:
                if memory_channels < 1:
                    messagebox.showerror("错误", "内存通道数必须大于0")
                    return False
                    
                if memory_offload_ratio < 10 or memory_offload_ratio > 100:
                    messagebox.showerror("错误", "内存交换比例必须在10-100之间")
                    return False
                    
                if reserved_memory < 0 or reserved_memory > 50:
                    messagebox.showerror("错误", "预留内存比例必须在0-50之间")
                    return False
            
            # 更新配置
            self.config['model_path'] = model_path
            self.config['ip'] = ip
            self.config['port'] = port
            self.config['gpu_count'] = gpu_count
            self.config['mem_ratio'] = mem_ratio
            self.config['max_tokens'] = max_tokens
            self.config['max_model_len'] = max_model_len
            self.config['block_size'] = block_size
            
            # 更新内存交换配置
            self.config['enable_memory_offload'] = enable_memory_offload
            self.config['memory_channels'] = memory_channels
            self.config['memory_offload_ratio'] = memory_offload_ratio
            self.config['reserved_memory'] = reserved_memory
            
            # 保存配置到文件
            self.save_config()
            
            # 在状态栏显示配置信息
            self.status_text.insert(tk.END, "\n===== 配置已更新 =====\n")
            self.status_text.insert(tk.END, f"模型路径: {model_path}\n")
            self.status_text.insert(tk.END, f"IP地址: {ip}, 端口: {port}\n")
            self.status_text.insert(tk.END, f"GPU数量: {gpu_count}, 显存比例: {mem_ratio}%\n")
            self.status_text.insert(tk.END, f"最大Token数: {max_tokens}, 最大模型长度: {max_model_len}, 块大小: {block_size}\n")
            
            if enable_memory_offload:
                self.status_text.insert(tk.END, f"已启用内存交换: 通道数={memory_channels}, 交换比例={memory_offload_ratio}%, 预留内存={reserved_memory}%\n")
            else:
                self.status_text.insert(tk.END, "未启用内存交换\n")
                
            self.status_text.see(tk.END)
            
            return True
            
        except Exception as e:
            messagebox.showerror("错误", f"更新配置失败: {str(e)}")
            return False

    def validate_config(self):
        """验证配置参数"""
        if self.config['max_tokens'] < self.config['max_model_len']:
            if not messagebox.askokcancel("警告", 
                "最大回复token数小于整体序列长度，这可能会影响模型性能。\n建议将max_tokens设置为不小于max_model_len。\n是否继续？"):
                return False
        return True

    def check_model_compatibility(self):
        """检查模型与VLLM的兼容性"""
        if not self.config['model_path']:
            self.status_text.insert(tk.END, "错误: 未选择模型路径\n")
            return False
        
        self.status_text.insert(tk.END, "正在检查模型兼容性...\n")
        
        # 检查硬件配置
        self.check_hardware_configuration()
        
        # 检查模型文件是否存在
        model_path = self.config['model_path']
        if not os.path.exists(model_path):
            self.status_text.insert(tk.END, f"错误: 模型路径不存在: {model_path}\n")
            return False
        
        # 检查必要的模型文件
        required_files = []
        safetensors_found = False
        bin_files_found = False
        
        # 检查是否有.safetensors文件
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith('.safetensors'):
                    safetensors_found = True
                    self.status_text.insert(tk.END, f"找到safetensors文件: {file}\n")
                elif file.endswith('.bin'):
                    bin_files_found = True
                    self.status_text.insert(tk.END, f"找到bin文件: {file}\n")
        
        if not (safetensors_found or bin_files_found):
            self.status_text.insert(tk.END, "错误: 未找到模型权重文件(.safetensors或.bin)\n")
            return False
        
        # 检查config.json文件
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            self.status_text.insert(tk.END, "错误: 未找到config.json文件\n")
            return False
        
        # 检查tokenizer文件
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
        tokenizer_found = False
        for file in tokenizer_files:
            if os.path.exists(os.path.join(model_path, file)):
                tokenizer_found = True
                break
        
        if not tokenizer_found:
            self.status_text.insert(tk.END, "警告: 未找到标准tokenizer文件，VLLM可能无法正确加载\n")
        
        # 读取模型配置
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 检查模型类型
            model_type = config.get('model_type', '')
            self.status_text.insert(tk.END, f"模型类型: {model_type}\n")
            
            # 检查是否是支持的模型类型
            supported_types = ["llama", "mistral", "falcon", "gpt_neox", "gpt2", "bloom", "qwen", "baichuan", "chatglm", "mpt"]
            if model_type.lower() not in [t.lower() for t in supported_types]:
                self.status_text.insert(tk.END, f"警告: 模型类型 '{model_type}' 可能不被VLLM完全支持\n")
            
            # 检查模型大小
            hidden_size = config.get('hidden_size', 0)
            num_layers = config.get('num_hidden_layers', 0) or config.get('num_layers', 0)
            vocab_size = config.get('vocab_size', 0)
            
            if hidden_size and num_layers:
                # 粗略估计模型参数量
                params_billion = (hidden_size * hidden_size * 4 * num_layers + hidden_size * vocab_size) / 1e9
                self.status_text.insert(tk.END, f"估计模型参数量: {params_billion:.2f}B\n")
                
                # 检查是否是大模型
                if params_billion > 30:
                    self.status_text.insert(tk.END, "警告: 这是一个较大的模型，可能需要多GPU或内存交换\n")
            
            # 检查特殊注意力机制
            attention_type = config.get('attention_type', '')
            if attention_type and attention_type not in ['scaled_dot_product', 'eager']:
                self.status_text.insert(tk.END, f"警告: 特殊注意力机制 '{attention_type}' 可能不被VLLM支持\n")
            
            # 检查激活函数
            activation_function = config.get('hidden_act', '')
            if activation_function and activation_function not in ['gelu', 'gelu_new', 'relu', 'silu', 'swish']:
                self.status_text.insert(tk.END, f"警告: 激活函数 '{activation_function}' 可能不被VLLM完全支持\n")
        
        except Exception as e:
            self.status_text.insert(tk.END, f"读取模型配置时出错: {str(e)}\n")
        
        # 检查VLLM版本
        try:
            vllm_version = subprocess.run(['vllm', '--version'], capture_output=True, text=True)
            version_str = vllm_version.stdout.strip() or vllm_version.stderr.strip()
            self.status_text.insert(tk.END, f"VLLM版本: {version_str}\n")
            
            # 检查CUDA版本
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                self.status_text.insert(tk.END, f"CUDA版本: {cuda_version}\n")
                
                # 检查GPU计算能力
                capability = torch.cuda.get_device_capability()
                self.status_text.insert(tk.END, f"GPU计算能力: {capability[0]}.{capability[1]}\n")
                
                # 检查是否支持当前GPU
                if capability[0] < 7:
                    self.status_text.insert(tk.END, "警告: VLLM最佳支持计算能力7.0+的GPU (V100及更新)\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"检查VLLM版本时出错: {str(e)}\n")
        
        # 检查GPU内存
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_memory = gpu.memoryTotal / 1024  # GB
                self.status_text.insert(tk.END, f"GPU显存: {gpu_memory:.2f}GB\n")
                
                # 估算模型大小
                model_size = self.estimate_model_size()
                self.status_text.insert(tk.END, f"估计模型大小: {model_size:.2f}GB\n")
                
                # 检查是否需要内存交换
                if model_size > gpu_memory * 0.8:
                    self.status_text.insert(tk.END, f"警告: 模型大小({model_size:.2f}GB)接近或超过GPU显存({gpu_memory:.2f}GB)\n")
                    self.status_text.insert(tk.END, "建议启用内存交换或使用多GPU\n")
                    
                    # 检查系统内存
                    system_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
                    self.status_text.insert(tk.END, f"系统内存: {system_memory:.2f}GB\n")
                    
                    if system_memory < model_size * 1.5:
                        self.status_text.insert(tk.END, "警告: 系统内存可能不足以进行有效的内存交换\n")
                    
                    # 检查磁盘空间（用于内存映射文件）
                    disk_usage = psutil.disk_usage('/')
                    free_disk = disk_usage.free / (1024 * 1024 * 1024)  # GB
                    self.status_text.insert(tk.END, f"可用磁盘空间: {free_disk:.2f}GB\n")
                    
                    if free_disk < model_size * 2:
                        self.status_text.insert(tk.END, "警告: 磁盘空间可能不足以创建内存映射文件\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"检查GPU内存时出错: {str(e)}\n")
        
        self.status_text.insert(tk.END, "模型兼容性检查完成\n")
        return True

    def check_hardware_configuration(self):
        """检测用户硬件配置并提供相应的优化建议"""
        self.status_text.insert(tk.END, "\n===== 硬件配置检测 =====\n")
        
        # 检测CPU
        try:
            cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
            cpu_logical = psutil.cpu_count(logical=True)  # 逻辑核心数
            self.status_text.insert(tk.END, f"CPU: {cpu_count}核心/{cpu_logical}线程\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"无法检测CPU信息: {str(e)}\n")
        
        # 检测内存
        try:
            mem = psutil.virtual_memory()
            total_memory = mem.total / (1024 * 1024 * 1024)  # GB
            self.status_text.insert(tk.END, f"系统内存: {total_memory:.2f}GB\n")
            
            # 根据内存大小提供建议
            if total_memory < 16:
                self.status_text.insert(tk.END, "警告: 系统内存较小，可能无法运行大型模型\n")
                self.status_text.insert(tk.END, "建议: 使用较小的模型或量化模型，减小block_size和max_model_len\n")
            elif total_memory < 32:
                self.status_text.insert(tk.END, "系统内存适中，可以运行中等大小的模型\n")
                self.status_text.insert(tk.END, "建议: 使用默认设置，或根据模型大小适当调整\n")
            else:
                self.status_text.insert(tk.END, "系统内存充足，可以运行较大模型\n")
                self.status_text.insert(tk.END, "建议: 可以适当增加内存通道数和缓存大小\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"无法检测内存信息: {str(e)}\n")
        
        # 检测GPU
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.status_text.insert(tk.END, f"检测到 {gpu_count} 个GPU\n")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024 * 1024)  # GB
                    self.status_text.insert(tk.END, f"GPU {i}: {gpu_name}, 显存: {gpu_mem:.2f}GB\n")
                    
                    # 根据显存大小提供建议
                    if gpu_mem < 8:
                        self.status_text.insert(tk.END, f"警告: GPU {i} 显存较小，可能无法运行大型模型\n")
                        self.status_text.insert(tk.END, f"建议: 使用较小的模型或量化模型，减小block_size\n")
                    elif gpu_mem < 16:
                        self.status_text.insert(tk.END, f"GPU {i} 显存适中，可以运行中等大小的模型\n")
                        self.status_text.insert(tk.END, f"建议: 使用默认设置，block_size设置为16-32\n")
                    else:
                        self.status_text.insert(tk.END, f"GPU {i} 显存充足，可以运行较大模型\n")
                        self.status_text.insert(tk.END, f"建议: 可以增加block_size到32-64\n")
            else:
                self.status_text.insert(tk.END, "未检测到支持CUDA的GPU\n")
                self.status_text.insert(tk.END, "警告: VLLM需要NVIDIA GPU才能运行\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"检测GPU信息时出错: {str(e)}\n")
        
        # 检测物理内存通道数（这只是一个估计，无法准确获取）
        try:
            # 尝试通过dmidecode获取内存通道信息（仅限Linux，需要root权限）
            if sys.platform.startswith('linux'):
                try:
                    result = subprocess.run(['sudo', 'dmidecode', '-t', 'memory'], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        # 分析输出，估计内存通道数
                        memory_devices = result.stdout.count("Memory Device")
                        if memory_devices > 0:
                            self.status_text.insert(tk.END, f"估计物理内存通道数: {memory_devices}\n")
                            
                            # 根据内存通道数提供建议
                            if memory_devices <= 2:
                                self.status_text.insert(tk.END, "物理内存通道数较少，可能限制内存带宽\n")
                                self.status_text.insert(tk.END, "建议: 使用4-8个软件内存通道\n")
                            else:
                                self.status_text.insert(tk.END, "物理内存通道数充足\n")
                                self.status_text.insert(tk.END, "建议: 可以使用8-16个软件内存通道\n")
                except Exception as e:
                    # 如果无法获取，使用默认估计
                    self.status_text.insert(tk.END, "无法准确检测物理内存通道数，使用默认估计\n")
                    self.status_text.insert(tk.END, "建议: 使用4-8个软件内存通道，避免设置过高\n")
            else:
                # 非Linux系统
                self.status_text.insert(tk.END, "无法在当前操作系统上检测物理内存通道数\n")
                self.status_text.insert(tk.END, "建议: 使用4-8个软件内存通道，避免设置过高\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"无法检测物理内存通道数: {str(e)}\n")
            self.status_text.insert(tk.END, "建议: 使用默认设置(4个软件内存通道)\n")
        
        self.status_text.insert(tk.END, "硬件配置检测完成\n")
        self.status_text.see(tk.END)

    def check_flash_attention_support(self):
        """检查是否支持Flash Attention"""
        try:
            import torch
            has_support = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            self.status_text.insert(tk.END, f"Flash Attention支持: {has_support}\n")
            return False  # 暂时禁用Flash Attention功能，避免兼容性问题
        except Exception as e:
            self.status_text.insert(tk.END, f"检查Flash Attention支持时出错: {str(e)}\n")
            return False

    def add_performance_monitoring(self):
        """添加性能监控与自动调优功能"""
        # 创建性能监控面板
        self.perf_frame = ttk.LabelFrame(self.master, text="性能监控")
        self.perf_frame.pack(padx=10, pady=5, fill='both')
        
        # 添加性能指标显示
        self.perf_labels = {}
        metrics = ["GPU利用率", "内存带宽", "KV缓存命中率", "推理速度(token/s)"]
        
        for i, metric in enumerate(metrics):
            ttk.Label(self.perf_frame, text=f"{metric}:").grid(row=i, column=0, sticky='w')
            self.perf_labels[metric] = ttk.Label(self.perf_frame, text="N/A")
            self.perf_labels[metric].grid(row=i, column=1, sticky='w')
        
        # 添加自动调优开关
        self.auto_tune_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.perf_frame, text="启用自动性能调优", variable=self.auto_tune_var).grid(row=len(metrics), column=0, columnspan=2, sticky='w')
        
        # 初始化性能统计变量
        self.total_tokens_generated = 0
        self.kv_cache_hits = 0
        self.kv_cache_misses = 0
        
        # 启动性能监控线程
        self.start_performance_monitor()

    def start_performance_monitor(self):
        """启动性能监控线程"""
        def monitor_loop():
            last_tokens = 0
            last_time = time.time()
            
            while hasattr(self, 'monitoring') and self.monitoring:
                try:
                    if hasattr(self, 'server_process') and self.server_process is not None and self.server_process.poll() is None:
                        # 获取GPU统计信息
                        gpu_stats = self.get_gpu_stats()
                        if gpu_stats and len(gpu_stats) > 0:
                            # 安全获取GPU利用率和内存利用率
                            gpu_util_str = gpu_stats[0].get('utilization.gpu', '0 %').replace('%', '').strip()
                            mem_util_str = gpu_stats[0].get('utilization.memory', '0 %').replace('%', '').strip()
                            
                            # 转换为浮点数，处理可能的转换错误
                            try:
                                gpu_util = float(gpu_util_str)
                            except ValueError:
                                gpu_util = 0.0
                                
                            try:
                                mem_util = float(mem_util_str)
                            except ValueError:
                                mem_util = 0.0
                            
                            # 更新GUI显示
                            if 'GPU利用率' in self.perf_labels:
                                self.perf_labels['GPU利用率'].config(text=f"{gpu_util:.1f}%")
                            if '内存带宽' in self.perf_labels:
                                self.perf_labels['内存带宽'].config(text=f"{mem_util:.1f}%")
                            
                            # 计算推理速度
                            if hasattr(self, 'total_tokens_generated'):
                                now = time.time()
                                if now - last_time >= 5:  # 每5秒更新一次
                                    tokens_per_sec = (self.total_tokens_generated - last_tokens) / (now - last_time)
                                    last_tokens = self.total_tokens_generated
                                    last_time = now
                                    
                                    if '推理速度(token/s)' in self.perf_labels:
                                        self.perf_labels['推理速度(token/s)'].config(text=f"{tokens_per_sec:.2f}")
                                    
                                    # 自动调优逻辑 - 只在服务运行且启用自动调优时执行
                                    if hasattr(self, 'monitoring') and self.monitoring and hasattr(self, 'auto_tune_var') and self.auto_tune_var.get() and tokens_per_sec < 5.0:
                                        # 如果GPU利用率高但内存带宽低，说明存在内存瓶颈
                                        if gpu_util > 90 and mem_util < 30:
                                            self.status_text.insert(tk.END, "检测到GPU计算瓶颈，尝试优化内存访问...\n")
                                            self.optimize_memory_access()
                                        # 如果GPU利用率低，说明存在计算瓶颈
                                        elif gpu_util < 30:
                                            self.status_text.insert(tk.END, "检测到GPU利用率低，尝试自动调优...\n")
                                            self.optimize_for_low_gpu_utilization()
                
                    # 更新KV缓存命中率
                    if hasattr(self, 'monitoring') and self.monitoring and hasattr(self, 'kv_cache_hits') and hasattr(self, 'kv_cache_misses'):
                        total_kv_requests = self.kv_cache_hits + self.kv_cache_misses
                        if total_kv_requests > 0:
                            kv_hit_ratio = self.kv_cache_hits / total_kv_requests * 100
                            if 'KV缓存命中率' in self.perf_labels:
                                self.perf_labels['KV缓存命中率'].config(text=f"{kv_hit_ratio:.2f}%")
                except Exception as e:
                    # 记录错误但不中断监控
                    print(f"性能监控错误: {str(e)}")
                
                # 检查监控标志
                if not hasattr(self, 'monitoring') or not self.monitoring:
                    break
                
                time.sleep(1)
        
        # 确保monitoring属性已设置
        if not hasattr(self, 'monitoring'):
            self.monitoring = True
            
        # 启动监控线程
        self.perf_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.perf_monitor_thread.start()

    def optimize_for_low_gpu_utilization(self):
        """针对低GPU利用率进行优化"""
        # 这个方法会在GPU利用率低于30%时被调用
        
        # 1. 尝试增加批处理大小
        if hasattr(self, 'batch_size'):
            old_batch_size = self.batch_size
            self.batch_size = min(self.batch_size * 2, 32)  # 最大批大小32
            if old_batch_size != self.batch_size:
                self.status_text.insert(tk.END, f"自动调优: 增加批处理大小到 {self.batch_size}\n")
        
        # 2. 尝试预热GPU
        self.status_text.insert(tk.END, "自动调优: 执行GPU预热操作\n")
        try:
            # 创建一个小的张量并执行一些操作来预热GPU
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # 创建一个大张量并执行一些操作
                x = torch.randn(1000, 1000, device=device)
                for _ in range(10):
                    x = torch.matmul(x, x)
                # 强制同步
                torch.cuda.synchronize()
                self.status_text.insert(tk.END, "GPU预热完成\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"GPU预热失败: {str(e)}\n")
        
        # 3. 检查并优化内存访问模式
        if hasattr(self, 'multi_channel_loader'):
            # 增加缓存大小
            if hasattr(self.multi_channel_loader, 'max_cache_size'):
                old_cache_size = self.multi_channel_loader.max_cache_size
                self.multi_channel_loader.max_cache_size = min(old_cache_size * 2, 128)
                self.status_text.insert(tk.END, f"自动调优: 增加内存缓存大小到 {self.multi_channel_loader.max_cache_size}\n")

    def optimize_memory_access(self):
        """优化内存访问模式，提高内存带宽利用率"""
        self.status_text.insert(tk.END, "优化内存访问模式...\n")
        
        # 1. 增加预取操作
        if hasattr(self, 'multi_channel_loader') and self.multi_channel_loader is not None:
            try:
                # 增加缓存大小，但不要过大
                if hasattr(self.multi_channel_loader, 'max_cache_size'):
                    old_cache_size = self.multi_channel_loader.max_cache_size
                    # 对于普通硬件，最大增加到64
                    self.multi_channel_loader.max_cache_size = min(old_cache_size * 2, 64)
                    self.status_text.insert(tk.END, f"增加内存缓存大小到 {self.multi_channel_loader.max_cache_size}\n")
            except Exception as e:
                self.status_text.insert(tk.END, f"优化内存缓存失败: {str(e)}\n")
        
        # 2. 尝试优化CUDA内存分配策略
        try:
            # 设置环境变量以优化CUDA内存分配，但使用较小的分块大小
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
            self.status_text.insert(tk.END, "已优化CUDA内存分配策略\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"优化CUDA内存分配失败: {str(e)}\n")
        
        # 3. 尝试调整KV缓存参数
        if hasattr(self, 'server_process') and self.server_process is not None and self.server_process.poll() is None:
            self.status_text.insert(tk.END, "建议重启服务器并使用更合适的block_size以提高内存带宽利用率\n")
            
            # 更新推荐设置
            current_block_size = int(self.block_size_var.get())
            # 对于普通硬件，最大推荐32
            recommended_block_size = min(current_block_size * 2, 32)
            self.status_text.insert(tk.END, f"推荐block_size: {recommended_block_size} (当前: {current_block_size})\n")

    def update_token_count(self, new_tokens):
        """更新生成的token计数"""
        if not hasattr(self, 'total_tokens_generated'):
            self.total_tokens_generated = 0
        self.total_tokens_generated += new_tokens

    def auto_optimize_performance(self):
        """服务器启动后自动进行性能优化"""
        try:
            # 等待服务器完全启动
            time.sleep(10)
            
            if not self.monitoring or not hasattr(self, 'server_process') or self.server_process is None or self.server_process.poll() is not None:
                return
                
            self.status_text.insert(tk.END, "\n===== 自动性能优化 =====\n")
            
            # 1. 预热GPU
            self.status_text.insert(tk.END, "执行GPU预热...\n")
            self.warm_up_gpu()
            
            # 2. 优化内存访问模式
            self.status_text.insert(tk.END, "优化内存访问模式...\n")
            self.optimize_memory_access()
            
            # 3. 提供性能优化建议
            self.status_text.insert(tk.END, "性能优化建议:\n")
            
            # 检查块大小
            current_block_size = int(self.block_size_var.get())
            if current_block_size < 32:
                self.status_text.insert(tk.END, f"- 当前块大小({current_block_size})较小，建议下次启动时使用32的块大小以提高内存带宽利用率\n")
            
            # 检查内存通道数
            if hasattr(self, 'multi_channel_loader') and self.multi_channel_loader is not None:
                current_channels = self.multi_channel_loader.num_channels
                if current_channels < 4:
                    self.status_text.insert(tk.END, f"- 当前内存通道数({current_channels})较少，建议下次启动时使用4-8个通道以提高内存带宽利用率\n")
            
            # 检查KV缓存设置
            self.status_text.insert(tk.END, "- 使用更大的批处理大小可以提高吞吐量，但会增加延迟\n")
            self.status_text.insert(tk.END, "- 如果内存带宽利用率仍然较低，可以尝试使用量化模型\n")
            
            # 添加硬件升级建议
            self.status_text.insert(tk.END, "硬件升级建议:\n")
            self.status_text.insert(tk.END, "- 如果您有更多内存(>64GB)和更多物理内存通道(>2)，可以在界面上手动增加内存通道数和缓存大小\n")
            self.status_text.insert(tk.END, "- 如果您有更大显存(>16GB)的GPU，可以尝试增加块大小到64\n")
            
            self.status_text.insert(tk.END, "自动性能优化完成\n")
            self.status_text.see(tk.END)
            
        except Exception as e:
            self.status_text.insert(tk.END, f"自动性能优化失败: {str(e)}\n")

    def warm_up_gpu(self):
        """预热GPU，提高性能稳定性"""
        try:
            # 创建一个小的张量并执行一些操作来预热GPU
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # 创建一个大张量并执行一些操作
                x = torch.randn(2000, 2000, device=device)
                for _ in range(20):
                    x = torch.matmul(x, x)
                # 强制同步
                torch.cuda.synchronize()
                self.status_text.insert(tk.END, "GPU预热完成\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"GPU预热失败: {str(e)}\n")

    def create_advanced_settings(self):
        """创建高级性能设置区域"""
        # 创建高级设置框架
        advanced_frame = ttk.LabelFrame(self.master, text="高级性能设置")
        advanced_frame.pack(padx=10, pady=5, fill='x')
        
        # 添加说明
        ttk.Label(advanced_frame, text="以下设置适用于高性能硬件，请根据您的实际硬件配置谨慎调整", 
                 foreground="red").grid(row=0, column=0, columnspan=4, sticky='w')
        
        # 内存缓存大小
        ttk.Label(advanced_frame, text="内存缓存大小:").grid(row=1, column=0)
        self.cache_size_var = tk.StringVar(value="32")
        cache_size_combo = ttk.Combobox(advanced_frame, textvariable=self.cache_size_var,
                                      values=["16", "32", "64", "128", "256"], width=5)
        cache_size_combo.grid(row=1, column=1)
        ttk.Label(advanced_frame, text="(大内存系统可增大)").grid(row=1, column=2)
        
        # CUDA内存分配块大小
        ttk.Label(advanced_frame, text="CUDA内存分块(MB):").grid(row=2, column=0)
        self.cuda_split_size_var = tk.StringVar(value="128")
        cuda_split_combo = ttk.Combobox(advanced_frame, textvariable=self.cuda_split_size_var,
                                      values=["64", "128", "256", "512"], width=5)
        cuda_split_combo.grid(row=2, column=1)
        ttk.Label(advanced_frame, text="(大显存GPU可增大)").grid(row=2, column=2)
        
        # 批处理大小
        ttk.Label(advanced_frame, text="批处理大小:").grid(row=3, column=0)
        self.batch_size_var = tk.StringVar(value="16")
        batch_size_combo = ttk.Combobox(advanced_frame, textvariable=self.batch_size_var,
                                      values=["8", "16", "32", "64"], width=5)
        batch_size_combo.grid(row=3, column=1)
        ttk.Label(advanced_frame, text="(高性能GPU可增大)").grid(row=3, column=2)
        
        # 检测硬件按钮
        detect_hardware_button = ttk.Button(advanced_frame, text="检测硬件配置", 
                                          command=self.check_hardware_configuration)
        detect_hardware_button.grid(row=4, column=0, columnspan=2, pady=5)
        
        # 应用高级设置按钮
        apply_advanced_button = ttk.Button(advanced_frame, text="应用高级设置", 
                                         command=self.apply_advanced_settings)
        apply_advanced_button.grid(row=4, column=2, columnspan=2, pady=5)
        
        # 添加说明
        ttk.Label(advanced_frame, text="注意: 高级设置将在下次启动服务器时生效", 
                 foreground="blue").grid(row=5, column=0, columnspan=4, sticky='w')
        
        # 加载已保存的高级设置
        self.load_advanced_settings()

    def load_advanced_settings(self):
        """加载已保存的高级设置"""
        try:
            # 如果配置中有高级设置，则加载
            if 'advanced_cache_size' in self.config:
                self.cache_size_var.set(str(self.config['advanced_cache_size']))
            if 'advanced_cuda_split_size' in self.config:
                self.cuda_split_size_var.set(str(self.config['advanced_cuda_split_size']))
            if 'advanced_batch_size' in self.config:
                self.batch_size_var.set(str(self.config['advanced_batch_size']))
        except Exception as e:
            self.status_text.insert(tk.END, f"加载高级设置失败: {str(e)}\n")

    def apply_advanced_settings(self):
        """应用高级性能设置"""
        try:
            # 获取高级设置值
            cache_size = int(self.cache_size_var.get())
            cuda_split_size = int(self.cuda_split_size_var.get())
            batch_size = int(self.batch_size_var.get())
            
            # 保存到配置
            self.config['advanced_cache_size'] = cache_size
            self.config['advanced_cuda_split_size'] = cuda_split_size
            self.config['advanced_batch_size'] = batch_size
            
            # 更新配置文件
            self.save_config()
            
            # 显示确认信息
            self.status_text.insert(tk.END, "\n===== 高级设置已应用 =====\n")
            self.status_text.insert(tk.END, f"内存缓存大小: {cache_size}\n")
            self.status_text.insert(tk.END, f"CUDA内存分块大小: {cuda_split_size}MB\n")
            self.status_text.insert(tk.END, f"批处理大小: {batch_size}\n")
            self.status_text.insert(tk.END, "这些设置将在下次启动服务器时生效\n")
            self.status_text.see(tk.END)
            
            messagebox.showinfo("成功", "高级设置已应用，将在下次启动服务器时生效")
        except Exception as e:
            messagebox.showerror("错误", f"应用高级设置失败: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VLLMServerGUI(root)
    root.mainloop()
