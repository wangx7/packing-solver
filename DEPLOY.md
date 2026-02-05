# 矩形装箱求解器 - 部署指南

## 🚀 免费部署平台推荐

### 1. Render（推荐 ⭐）
最简单的方式，支持免费部署 Python 应用。

**步骤：**
1. 访问 https://render.com 注册账号
2. 点击 "New" → "Web Service"
3. 连接你的 GitHub 仓库（需先上传代码到 GitHub）
4. 配置：
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python web_server.py`
5. 点击 "Create Web Service"

**注意**: 免费版会在 15 分钟无访问后休眠，首次访问需等待约 30 秒启动。

---

### 2. Hugging Face Spaces（推荐 ⭐）
适合 AI/ML 相关项目，完全免费。

**步骤：**
1. 访问 https://huggingface.co 注册账号
2. 点击头像 → "New Space"
3. 选择 "Docker" 作为 SDK
4. 上传代码文件
5. 创建 `Dockerfile`（见下方）

---

### 3. Railway
提供每月 $5 免费额度。

**步骤：**
1. 访问 https://railway.app 注册
2. "New Project" → "Deploy from GitHub repo"
3. 选择仓库，自动检测 Python 项目
4. 自动部署

---

### 4. PythonAnywhere
专门为 Python 设计的免费托管。

**步骤：**
1. 访问 https://www.pythonanywhere.com 注册
2. 上传代码文件
3. 在 Web 选项卡配置 WSGI 应用

---

## 📦 上传到 GitHub

```bash
# 初始化 Git 仓库
cd /Users/wx/packing_solver
git init
git add .
git commit -m "Initial commit: 矩形装箱求解器"

# 创建 GitHub 仓库后
git remote add origin https://github.com/你的用户名/packing-solver.git
git push -u origin main
```

---

## 📁 部署所需文件

确保仓库包含以下文件：
- `web_server.py` - Web 服务器
- `packing_solver.py` - 求解器核心
- `requirements.txt` - Python 依赖
- `Procfile` - Render/Heroku 启动配置
