"""Swagger配置"""

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "模型转换与部署 API",
        "version": "0.1.0",
        "description": "独立项目：上传/登记模型，转换与压缩，编译部署；集中字段标识与 Schema 暴露。",
    },
    "basePath": "/",
}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
}

