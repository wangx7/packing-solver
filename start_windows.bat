@echo off
chcp 65001 >nul
echo ================================================
echo    矩形装箱求解器 - Windows 启动脚本
echo ================================================
echo.

:: 检查 Python 是否安装
python --version
if errorlevel 1 (
    echo [错误] 未检测到 Python！
    echo.
    echo 请先安装 Python：
    echo 1. 访问 https://www.python.org/downloads/
    echo 2. 下载并安装（勾选 "Add Python to PATH"）
    echo 3. 重新运行此脚本
    echo.
    pause
    exit /b 1
)

echo [1/3] 检测到 Python，版本：
python --version
echo.

:: 安装依赖
echo [2/3] 安装依赖...
pip install numpy -q
if errorlevel 1 (
    echo [警告] pip 安装失败，尝试使用 py 命令...
    py -m pip install numpy -q
)
echo 依赖安装完成！
echo.

:: 启动服务
echo [3/3] 启动 Web 服务...
echo.
echo ================================================
echo    服务即将启动，浏览器会自动打开
echo    如果没有自动打开，请手动访问：
echo    http://localhost:8080
echo ================================================
echo.
echo 按 Ctrl+C 可停止服务
echo.

python web_server.py

pause
