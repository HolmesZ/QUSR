#!/usr/bin/env python3
"""
启动vLLM服务脚本
用于部署Qwen2.5-VL-7B-Instruct模型
"""

import subprocess
import argparse
import os
import sys

def start_vllm_service(model_path, port=8000, host="0.0.0.0", dtype="bfloat16", max_model_len=8192):
    """
    启动vLLM服务
    
    Args:
        model_path: 模型路径
        port: 服务端口
        host: 服务主机地址
        dtype: 数据类型
        max_model_len: 最大模型长度
    """
    
    # 构建vLLM启动命令 - 使用vllm serve命令
    cmd = [
        "vllm", "serve", model_path,
        "--port", str(port),
        "--host", host,
        "--dtype", dtype,
        "--max-model-len", str(max_model_len)
    ]
    
    print(f"启动vLLM服务...")
    print(f"模型路径: {model_path}")
    print(f"服务地址: {host}:{port}")
    print(f"数据类型: {dtype}")
    print(f"最大模型长度: {max_model_len}")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        # 启动服务
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动vLLM服务失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n服务已停止")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动vLLM服务")
    parser.add_argument("--model_path", type=str, 
                       default="/data2/Solar_Data/PiSA-SR/Qwen2.5-VL-7B-Instruct",
                       help="模型路径")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                       choices=["bfloat16", "float16", "float32"],
                       help="数据类型")
    parser.add_argument("--max_model_len", type=int, default=8192, 
                       help="最大模型长度（减少此值以节省GPU内存）")
    
    args = parser.parse_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        sys.exit(1)
    
    # 启动服务
    start_vllm_service(
        model_path=args.model_path,
        port=args.port,
        host=args.host,
        dtype=args.dtype,
        max_model_len=args.max_model_len
    ) 