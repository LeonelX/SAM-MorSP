#!/bin/bash

# 定义查找空闲端口的函数
find_free_network_port() {
    python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

# 初始化变量
cfg_file=""
task_fold=""
gpu_ids=""

# 使用 getopts 解析命令行参数
while getopts "c:t:g:" opt; do
    case $opt in
        c)
            cfg_file="$OPTARG"
            ;;
        t)
            task_fold="$OPTARG"
            ;;
        g)
            gpu_ids="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Usage: $0 -c <cfg_file> -t <task_fold> -g <gpu_ids>"
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            echo "Usage: $0 -c <cfg_file> -t <task_fold> -g <gpu_ids>"
            exit 1
            ;;
    esac
done

# 检查参数是否都已提供
if [ -z "$cfg_file" ] || [ -z "$task_fold" ] || [ -z "$gpu_ids" ]; then
    echo "Missing required arguments."
    echo "Usage: $0 -c <cfg_file> -t <task_fold> -g <gpu_ids>"
    exit 1
fi

# 配置GPU环境
port=$(find_free_network_port)
export CUDA_VISIBLE_DEVICES=$gpu_ids

# 生成命令
torchun_path=$(which torchrun)
if [ -z "$torchun_path" ]; then
    echo "torchrun not found. Please check your environment."
    exit 1
fi

gpu_count=$(echo "$gpu_ids" | tr ',' '\n' | wc -l)
cmd_line=("$torchun_path" "--nproc_per_node=$gpu_count")
cmd_line+=("--master_port=$port" "--nnodes=1")
cmd_line+=("trainer.py" "--cfg_file=$cfg_file" "--task_fold=$task_fold")

# 打印并执行命令
echo "Running command: ${cmd_line[*]}"
"${cmd_line[@]}"
