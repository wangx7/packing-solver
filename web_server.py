"""
矩形装箱求解器 - Web 可视化界面
启动方式: python web_server.py
访问地址: http://localhost:8080
"""

import json
import http.server
import socketserver
from urllib.parse import parse_qs, urlparse
import threading
import webbrowser
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from packing_solver import solve, SolveMode

# 支持环境变量端口（云部署需要）
PORT = int(os.environ.get('PORT', 8080))

HTML_CONTENT = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>矩形装箱求解器</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 20px;
        }
        
        @media (max-width: 900px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        
        .card h2 {
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        
        .form-group {
            margin-bottom: 16px;
        }
        
        label {
            display: block;
            margin-bottom: 6px;
            color: #555;
            font-weight: 500;
        }
        
        input[type="number"], select {
            width: 100%;
            padding: 10px 14px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input[type="number"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .size-inputs {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        
        .rectangles-list {
            margin-top: 16px;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .rect-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 14px;
            background: #f5f5f5;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        
        .rect-item span {
            font-weight: 500;
            color: #333;
        }
        
        .rect-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
        }
        
        .rect-info {
            display: flex;
            align-items: center;
            flex: 1;
        }
        
        .btn-remove {
            background: #ff4757;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 12px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        
        .btn-remove:hover {
            background: #ff3344;
        }
        
        .btn-add {
            width: 100%;
            padding: 12px;
            background: #2ed573;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s;
            margin-top: 10px;
        }
        
        .btn-add:hover {
            background: #26c066;
        }
        
        .btn-solve {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
            margin-top: 20px;
        }
        
        .btn-solve:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-solve:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .canvas-container {
            position: relative;
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 400px;
        }
        
        #canvas {
            border: 3px solid #333;
            border-radius: 4px;
            background: #fafafa;
        }
        
        .result-info {
            margin-top: 20px;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        
        .result-success {
            background: #d4edda;
            color: #155724;
        }
        
        .result-fail {
            background: #f8d7da;
            color: #721c24;
        }
        
        .result-info h3 {
            margin-bottom: 8px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .placement-details {
            margin-top: 16px;
            font-size: 14px;
            text-align: left;
        }
        
        .placement-details table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .placement-details th, .placement-details td {
            padding: 8px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        
        .placement-details th {
            background: #f5f5f5;
        }
        
        .presets {
            margin-bottom: 20px;
        }
        
        .preset-btn {
            padding: 8px 16px;
            margin-right: 8px;
            margin-bottom: 8px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }
        
        .preset-btn:hover {
            background: #667eea;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📦 矩形装箱求解器</h1>
        
        <div class="main-content">
            <div class="card">
                <h2>⚙️ 参数设置</h2>
                
                <div class="presets">
                    <label>快速示例：</label>
                    <button class="preset-btn" onclick="loadPreset(1)">简单3矩形</button>
                    <button class="preset-btn" onclick="loadPreset(2)">紧密排列</button>
                    <button class="preset-btn" onclick="loadPreset(3)">不可放入</button>
                </div>
                
                <div class="form-group">
                    <label>容器尺寸</label>
                    <div class="size-inputs">
                        <input type="number" id="containerW" value="20" min="1" placeholder="宽度">
                        <input type="number" id="containerH" value="15" min="1" placeholder="高度">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>添加矩形</label>
                    <div class="size-inputs">
                        <input type="number" id="rectW" value="6" min="1" placeholder="宽度">
                        <input type="number" id="rectH" value="4" min="1" placeholder="高度">
                    </div>
                    <button class="btn-add" onclick="addRectangle()">+ 添加矩形</button>
                </div>
                
                <div class="form-group">
                    <label>矩形列表 (<span id="rectCount">0</span>/10)</label>
                    <div class="rectangles-list" id="rectList"></div>
                </div>
                
                <div class="form-group">
                    <label>求解模式</label>
                    <select id="solveMode">
                        <option value="fast">快速模式 (贪心算法)</option>
                        <option value="precise" selected>精确模式 (模拟退火)</option>
                    </select>
                </div>
                
                <button class="btn-solve" id="solveBtn" onclick="solve()">🚀 开始求解</button>
            </div>
            
            <div class="card">
                <h2>📊 可视化结果</h2>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>正在求解中...</p>
                </div>
                
                <div class="canvas-container" id="canvasContainer">
                    <canvas id="canvas" width="500" height="400"></canvas>
                </div>
                
                <div class="result-info" id="resultInfo" style="display: none;"></div>
            </div>
        </div>
    </div>
    
    <script>
        const COLORS = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
            '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
            '#BB8FCE', '#85C1E9'
        ];
        
        let rectangles = [];
        
        function addRectangle() {
            if (rectangles.length >= 10) {
                alert('最多只能添加10个矩形！');
                return;
            }
            
            const w = parseFloat(document.getElementById('rectW').value);
            const h = parseFloat(document.getElementById('rectH').value);
            
            if (w <= 0 || h <= 0 || isNaN(w) || isNaN(h)) {
                alert('请输入有效的矩形尺寸！');
                return;
            }
            
            rectangles.push({ w, h, color: COLORS[rectangles.length % COLORS.length] });
            updateRectList();
            drawEmpty();
        }
        
        function removeRectangle(index) {
            rectangles.splice(index, 1);
            // 重新分配颜色
            rectangles.forEach((r, i) => r.color = COLORS[i % COLORS.length]);
            updateRectList();
            drawEmpty();
        }
        
        function updateRectList() {
            const list = document.getElementById('rectList');
            document.getElementById('rectCount').textContent = rectangles.length;
            
            if (rectangles.length === 0) {
                list.innerHTML = '<p style="color: #999; text-align: center; padding: 20px;">暂无矩形，请添加</p>';
                return;
            }
            
            list.innerHTML = rectangles.map((r, i) => `
                <div class="rect-item">
                    <div class="rect-info">
                        <div class="rect-color" style="background: ${r.color}"></div>
                        <span>矩形${i + 1}: ${r.w} × ${r.h}</span>
                    </div>
                    <button class="btn-remove" onclick="removeRectangle(${i})">删除</button>
                </div>
            `).join('');
        }
        
        function drawEmpty() {
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const W = parseFloat(document.getElementById('containerW').value) || 20;
            const H = parseFloat(document.getElementById('containerH').value) || 15;
            
            // 计算缩放
            const maxWidth = 500;
            const maxHeight = 400;
            const scale = Math.min(maxWidth / W, maxHeight / H) * 0.9;
            
            canvas.width = W * scale + 40;
            canvas.height = H * scale + 40;
            
            ctx.fillStyle = '#fafafa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // 绘制容器
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2;
            ctx.strokeRect(20, 20, W * scale, H * scale);
            
            // 标注尺寸
            ctx.fillStyle = '#666';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(W + '', 20 + W * scale / 2, 15);
            ctx.save();
            ctx.translate(12, 20 + H * scale / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(H + '', 0, 0);
            ctx.restore();
        }
        
        function drawResult(data) {
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const W = data.container_w;
            const H = data.container_h;
            
            // 计算缩放
            const maxWidth = 500;
            const maxHeight = 400;
            const scale = Math.min(maxWidth / W, maxHeight / H) * 0.9;
            
            canvas.width = W * scale + 40;
            canvas.height = H * scale + 40;
            
            ctx.fillStyle = '#fafafa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const offsetX = 20;
            const offsetY = 20;
            
            // 绘制容器
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2;
            ctx.strokeRect(offsetX, offsetY, W * scale, H * scale);
            
            // 绘制矩形
            if (data.success && data.placements) {
                data.placements.forEach((p, i) => {
                    ctx.save();
                    ctx.translate(offsetX + p.cx * scale, offsetY + (H - p.cy) * scale);
                    ctx.rotate(-p.angle);
                    
                    ctx.fillStyle = COLORS[i % COLORS.length] + 'CC';
                    ctx.fillRect(-p.w * scale / 2, -p.h * scale / 2, p.w * scale, p.h * scale);
                    
                    ctx.strokeStyle = COLORS[i % COLORS.length];
                    ctx.lineWidth = 2;
                    ctx.strokeRect(-p.w * scale / 2, -p.h * scale / 2, p.w * scale, p.h * scale);
                    
                    // 标注序号
                    ctx.fillStyle = '#000';
                    ctx.font = 'bold 16px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText((i + 1) + '', 0, 0);
                    
                    ctx.restore();
                });
            }
            
            // 标注尺寸
            ctx.fillStyle = '#666';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(W + '', offsetX + W * scale / 2, 15);
            ctx.save();
            ctx.translate(12, offsetY + H * scale / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(H + '', 0, 0);
            ctx.restore();
        }
        
        async function solve() {
            if (rectangles.length === 0) {
                alert('请先添加矩形！');
                return;
            }
            
            const W = parseFloat(document.getElementById('containerW').value);
            const H = parseFloat(document.getElementById('containerH').value);
            const mode = document.getElementById('solveMode').value;
            
            if (W <= 0 || H <= 0 || isNaN(W) || isNaN(H)) {
                alert('请输入有效的容器尺寸！');
                return;
            }
            
            // 显示加载
            document.getElementById('loading').classList.add('active');
            document.getElementById('canvasContainer').style.display = 'none';
            document.getElementById('resultInfo').style.display = 'none';
            document.getElementById('solveBtn').disabled = true;
            
            try {
                const response = await fetch('/solve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        container_w: W,
                        container_h: H,
                        rectangles: rectangles.map(r => [r.w, r.h]),
                        mode: mode
                    })
                });
                
                const data = await response.json();
                
                // 隐藏加载
                document.getElementById('loading').classList.remove('active');
                document.getElementById('canvasContainer').style.display = 'flex';
                document.getElementById('solveBtn').disabled = false;
                
                // 显示结果
                drawResult(data);
                
                const resultInfo = document.getElementById('resultInfo');
                resultInfo.style.display = 'block';
                
                if (data.success) {
                    resultInfo.className = 'result-info result-success';
                    let html = `<h3>✅ 可以放入！</h3>
                        <p>求解模式: ${data.mode_used} | 耗时: ${data.time_ms.toFixed(1)}ms</p>`;
                    
                    if (data.placements && data.placements.length > 0) {
                        html += `<div class="placement-details">
                            <table>
                                <tr><th>矩形</th><th>尺寸</th><th>中心点</th><th>旋转角度</th></tr>
                                ${data.placements.map((p, i) => `
                                    <tr>
                                        <td>${i + 1}</td>
                                        <td>${p.w} × ${p.h}</td>
                                        <td>(${p.cx.toFixed(1)}, ${p.cy.toFixed(1)})</td>
                                        <td>${(p.angle * 180 / Math.PI).toFixed(1)}°</td>
                                    </tr>
                                `).join('')}
                            </table>
                        </div>`;
                    }
                    resultInfo.innerHTML = html;
                } else {
                    resultInfo.className = 'result-info result-fail';
                    resultInfo.innerHTML = `<h3>❌ 无法放入</h3>
                        <p>这些矩形无法全部放入容器中</p>
                        <p>求解模式: ${data.mode_used} | 耗时: ${data.time_ms.toFixed(1)}ms</p>`;
                }
                
            } catch (error) {
                document.getElementById('loading').classList.remove('active');
                document.getElementById('canvasContainer').style.display = 'flex';
                document.getElementById('solveBtn').disabled = false;
                alert('求解出错: ' + error.message);
            }
        }
        
        function loadPreset(id) {
            rectangles = [];
            
            if (id === 1) {
                // 简单3矩形
                document.getElementById('containerW').value = 20;
                document.getElementById('containerH').value = 15;
                rectangles = [
                    { w: 8, h: 6, color: COLORS[0] },
                    { w: 7, h: 5, color: COLORS[1] },
                    { w: 6, h: 4, color: COLORS[2] }
                ];
            } else if (id === 2) {
                // 紧密排列
                document.getElementById('containerW').value = 25;
                document.getElementById('containerH').value = 20;
                rectangles = [
                    { w: 10, h: 8, color: COLORS[0] },
                    { w: 9, h: 6, color: COLORS[1] },
                    { w: 7, h: 5, color: COLORS[2] },
                    { w: 8, h: 4, color: COLORS[3] },
                    { w: 5, h: 5, color: COLORS[4] }
                ];
            } else if (id === 3) {
                // 不可放入
                document.getElementById('containerW').value = 10;
                document.getElementById('containerH').value = 10;
                rectangles = [
                    { w: 7, h: 7, color: COLORS[0] },
                    { w: 7, h: 7, color: COLORS[1] }
                ];
            }
            
            updateRectList();
            drawEmpty();
        }
        
        // 初始化
        updateRectList();
        drawEmpty();
    </script>
</body>
</html>
'''


class PackingHandler(http.server.SimpleHTTPRequestHandler):
    """处理 HTTP 请求"""
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/solve':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                W = float(data['container_w'])
                H = float(data['container_h'])
                rects = [(float(r[0]), float(r[1])) for r in data['rectangles']]
                mode_str = data.get('mode', 'precise')
                
                # 选择求解模式
                if mode_str == 'fast':
                    mode = SolveMode.FAST
                else:
                    mode = SolveMode.PRECISE
                
                # 求解
                result = solve(W, H, rects, mode=mode, parallel=True)
                
                # 构建响应
                response = {
                    'success': result.success,
                    'container_w': W,
                    'container_h': H,
                    'mode_used': result.mode_used,
                    'time_ms': result.time_ms,
                    'placements': []
                }
                
                if result.success:
                    for p in result.placements:
                        response['placements'].append({
                            'w': p.w,
                            'h': p.h,
                            'cx': p.cx,
                            'cy': p.cy,
                            'angle': p.angle
                        })
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    # 绑定 0.0.0.0 以便云服务访问
    with socketserver.TCPServer(("0.0.0.0", PORT), PackingHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print("=" * 50)
        print("📦 矩形装箱求解器 - Web 可视化界面")
        print("=" * 50)
        print(f"✅ 服务已启动: {url}")
        print("按 Ctrl+C 停止服务")
        print("=" * 50)
        
        # 本地运行时自动打开浏览器
        if os.environ.get('PORT') is None:
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务已停止")


if __name__ == '__main__':
    main()
