"""Flask应用服务器启动脚本

启动方式：
    python -m app.server
    
或者：
    python app/server.py
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径，确保可以导入其他模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from flask import Flask
from flasgger import Swagger
from api.compression import compression_api_bp
from api.upload import upload_api_bp
from api.convert import convert_api_bp
from api.compile import compile_api_bp
from config.settings import Config
from config.swagger import swagger_template, swagger_config

# 确保必要的目录存在
Config.ensure_dirs()

# 创建Flask应用实例
app = Flask(__name__)

# 初始化Swagger
Swagger(app, template=swagger_template, config=swagger_config)

# 注册API Blueprint
app.register_blueprint(compression_api_bp)
app.register_blueprint(upload_api_bp)
app.register_blueprint(convert_api_bp)
app.register_blueprint(compile_api_bp)

if __name__ == '__main__':
    print("=" * 60)
    print("模型转换和部署功能 - Flask服务器")
    print("=" * 60)
    print(f"服务器地址: http://{Config.HOST}:{Config.PORT}")
    print(f"调试模式: {Config.DEBUG}")
    print(f"日志级别: {Config.LOG_LEVEL}")
    print("=" * 60)
    print("\n可用API端点:")
    print("  POST /detect-capabilities  - 检测模型能力")
    print("  POST /execute              - 执行压缩操作")
    print("  POST /upload-extra-files   - 上传额外文件")
    print("  POST /convert-format       - 格式转换")
    print("  POST /list-supported-formats - 列出支持的格式")
    print("  POST /compile              - 硬件编译")
    print("  GET  /list-hardware        - 列出支持的硬件平台")
    print("\nSwagger API文档:")
    print(f"  http://{Config.HOST}:{Config.PORT}/apidocs/")
    print("\n按 Ctrl+C 停止服务器\n")
    print("=" * 60)
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )

