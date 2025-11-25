@echo off
chcp 65001 >nul
echo ========================================
echo VGG16 模型测试命令集合
echo ========================================
echo.

set BASE_DIR=D:/项目空间/image_classification_inceptionv4/模型转换和部署功能/artifacts/m_test_vgg/v_00001
set API_URL=http://localhost:5000

echo [测试1] 检测模型能力（基础）
echo ----------------------------------------
curl -X POST %API_URL%/detect-capabilities -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\"}"
echo.
echo.

echo [测试2] 检测模型能力（带 extra_dir）
echo ----------------------------------------
curl -X POST %API_URL%/detect-capabilities -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"extra_dir\": \"%BASE_DIR%/extra\"}"
echo.
echo.

echo [测试3] FP16量化
echo ----------------------------------------
curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"fp16\"}"
echo.
echo.

echo [测试4] INT8动态量化
echo ----------------------------------------
curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"int8_dynamic\"}"
echo.
echo.

echo [测试5] INT8静态量化（需要校准数据）
echo ----------------------------------------
curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"extra_dir\": \"%BASE_DIR%/extra\", \"method\": {\"quantize\": {\"enable\": true, \"precision\": \"int8_static\", \"calib_num\": 100}}}"
echo.
echo.

echo [测试6] 自动量化
echo ----------------------------------------
curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"quantize\": {\"enable\": true, \"auto\": true}}}"
echo.
echo.

echo [测试7] 结构化剪枝（30%%稀疏度，默认BN-based）
echo ----------------------------------------
curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"type\": \"structured\", \"target_sparsity\": 0.3}}}"
echo.
echo.

echo [测试8] 自动剪枝（VGG默认使用结构化剪枝）
echo ----------------------------------------
curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"type\": \"auto\", \"target_sparsity\": 0.3}}}"
echo.
echo.

echo [测试9] 非结构化剪枝（40%%稀疏度）
echo ----------------------------------------
curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"type\": \"unstructured\", \"target_sparsity\": 0.4}}}"
echo.
echo.

echo [测试10] 组合压缩：结构化剪枝30%% + FP16量化
echo ----------------------------------------
curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"type\": \"structured\", \"target_sparsity\": 0.3}, \"quantize\": {\"enable\": true, \"precision\": \"fp16\"}}}"
echo.
echo.

echo [测试11] 组合压缩：自动剪枝 + 自动量化
echo ----------------------------------------
curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"type\": \"auto\", \"target_sparsity\": 0.3}, \"quantize\": {\"enable\": true, \"auto\": true}}}"
echo.
echo.

echo [测试12] 转换为ONNX格式
echo ----------------------------------------
curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"onnx\"]}"
echo.
echo.

echo [测试13] 转换压缩后的模型为ONNX（FP16量化后）
echo ----------------------------------------
curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/optimized\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"onnx\"], \"model_file\": \"model_quantized_fp16.pt\"}"
echo.
echo.

echo [测试14] 转换压缩后的模型为ONNX（剪枝后）
echo ----------------------------------------
curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/optimized\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"onnx\"], \"model_file\": \"model_pruned_30pct.pt\"}"
echo.
echo.

echo [测试15] 转换压缩后的模型为ONNX（组合压缩后）
echo ----------------------------------------
curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/optimized\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"onnx\"], \"model_file\": \"model_pruned_30pct_quantized_fp16.pt\"}"
echo.
echo.

echo [测试16] 转换为TorchScript格式
echo ----------------------------------------
curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"torchscript\"]}"
echo.
echo.

echo [测试17] 同时转换为ONNX和TorchScript
echo ----------------------------------------
curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"onnx\", \"torchscript\"]}"
echo.
echo.

echo [测试18] 列出支持的格式
echo ----------------------------------------
curl -X POST %API_URL%/list-supported-formats -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\"}"
echo.
echo.

echo [测试19] 列出支持的硬件平台
echo ----------------------------------------
curl -X POST %API_URL%/list-supported-platforms -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\"}"
echo.
echo.

echo ========================================
echo 测试完成！
echo ========================================
pause

