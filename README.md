# 矩形装箱求解器 📦

一个可视化的矩形装箱问题求解器，支持任意角度旋转。

## 在线演示

[点击访问](你的部署地址)

## 功能

- 🎯 判断多个矩形能否放入容器
- 🔄 支持任意角度旋转
- 📊 实时可视化放置结果
- ⚡ 并行计算加速

## 本地运行

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python web_server.py
```

访问 http://localhost:8080

## 截图

在网页界面中：
1. 设置容器尺寸
2. 添加矩形（宽×高）
3. 点击"开始求解"
4. 查看可视化结果

## 技术栈

- Python 3
- NumPy（几何计算）
- 原生 HTTP 服务器（无需 Flask）
