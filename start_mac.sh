#!/bin/bash
echo "================================================"
echo "   矩形装箱求解器 - Mac/Linux 启动脚本"
echo "================================================"
echo

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3！"
    echo "Mac: brew install python3"
    echo "Ubuntu: sudo apt install python3"
    exit 1
fi

echo "[1/3] 检测到 Python："
python3 --version
echo

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "[2/3] 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境并安装依赖
source venv/bin/activate
pip install numpy -q
echo "[2/3] 依赖安装完成！"
echo

echo "[3/3] 启动 Web 服务..."
echo
echo "================================================"
echo "   浏览器会自动打开"
echo "   如果没有，请访问: http://localhost:8080"
echo "================================================"
echo
echo "按 Ctrl+C 停止服务"
echo

python3 web_server.py
