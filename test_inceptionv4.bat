@echo off
chcp 65001 >nul
echo ========================================
echo InceptionV4 单个测试命令
echo ========================================
echo.
echo 请选择要执行的测试：
echo.
echo 1. 检测模型能力（基础）
echo 2. 检测模型能力（带 extra_dir）
echo 3. FP16量化
echo 4. INT8量化（自动选择：有校准数据用静态，否则用动态）
echo 5. INT8动态量化（旧API，向后兼容）
echo 6. INT8静态量化（需要校准数据，旧API，向后兼容）
echo 7. 自动量化（auto）
echo 8. 结构化剪枝（30%%稀疏度）
echo 9. 自动剪枝（auto_pruning）
echo 10. 非结构化剪枝（40%%稀疏度）
echo 11. 组合压缩：剪枝30%% + FP16量化
echo 12. 组合压缩：FP16量化 + 剪枝40%%
echo 13. 知识蒸馏（需要教师模型和训练数据）
echo 14. 自动蒸馏（auto_distillation）
echo 15. 列出支持的格式
echo 16. 转换为 ONNX
echo 17. 转换为 TorchScript
echo 18. 同时转换为 ONNX 和 TorchScript
echo 19. 转换压缩后的模型为ONNX（指定文件）
echo 20. 列出支持的硬件平台
echo 21. TensorRT编译（从原始 .pt 文件）
echo 22. TensorRT编译（从 ONNX 文件）
echo 23. Ascend编译（从原始 .pt 文件）
echo 24. Ascend编译（从 ONNX 文件）
echo.
set /p choice=请输入测试编号（1-24）: 

set BASE_DIR=D:/项目空间/image_classification_inceptionv4/模型转换和部署功能/artifacts/my_test_inceptionv4/v_001
set API_URL=http://localhost:5000

if "%choice%"=="1" (
    echo [测试1] 检测模型能力（基础）
curl -X POST %API_URL%/detect-capabilities -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\"}"    
)
if "%choice%"=="2" (
    echo [测试2] 检测模型能力（带 extra_dir）
    curl -X POST %API_URL%/detect-capabilities -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"extra_dir\": \"%BASE_DIR%/extra\"}"
)
if "%choice%"=="3" (
    echo [测试3] FP16量化
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"fp16\"}"
)
if "%choice%"=="4" (
    echo [测试4] INT8量化（自动选择：有校准数据用静态，否则用动态）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"extra_dir\": \"%BASE_DIR%/extra\", \"method\": \"int8\"}"
)
if "%choice%"=="5" (
    echo [测试5] INT8动态量化（旧API，向后兼容）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"int8_dynamic\"}"
)
if "%choice%"=="6" (
    echo [测试6] INT8静态量化（需要校准数据，旧API，向后兼容）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"extra_dir\": \"%BASE_DIR%/extra\", \"method\": \"int8_static\"}"
)
if "%choice%"=="7" (
    echo [测试7] 自动量化（auto）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"extra_dir\": \"%BASE_DIR%/extra\", \"method\": \"auto\"}"
)
if "%choice%"=="8" (
    echo [测试8] 结构化剪枝（30%%稀疏度）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"structured_pruning\"}"
)
if "%choice%"=="9" (
    echo [测试9] 自动剪枝（auto_pruning）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"auto_pruning\"}"
)
if "%choice%"=="10" (
    echo [测试10] 非结构化剪枝（40%%稀疏度）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"method\": \"unstructured\", \"target_sparsity\": 0.4}}}"
)
if "%choice%"=="11" (
    echo [测试11] 组合压缩：剪枝30%% + FP16量化
 curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"method\": \"structured\", \"target_sparsity\": 0.3}, \"quantize\": {\"enable\": true, \"precision\": \"fp16\"}}}"   
)
if "%choice%"=="12" (
    echo [测试12] 组合压缩：FP16量化 + 剪枝40%%
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"quantize\": {\"enable\": true, \"precision\": \"fp16\"}, \"prune\": {\"enable\": true, \"method\": \"structured\", \"target_sparsity\": 0.4}}}"
)
if "%choice%"=="13" (
    echo [测试13] 知识蒸馏（需要教师模型和训练数据）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"extra_dir\": \"%BASE_DIR%/extra\", \"method\": {\"distill\": {\"enable\": true, \"temperature\": 4.0, \"alpha\": 0.7, \"epochs\": 20}}}"
)
if "%choice%"=="14" (
    echo [测试14] 自动蒸馏（auto_distillation）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"extra_dir\": \"%BASE_DIR%/extra\", \"method\": \"auto_distillation\"}"
)
if "%choice%"=="15" (
    echo [测试15] 列出支持的格式
    curl -X POST %API_URL%/list-supported-formats -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\"}"
)
if "%choice%"=="16" (
    echo [测试16] 转换为 ONNX
    curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"onnx\"]}"
)
if "%choice%"=="17" (
    echo [测试17] 转换为 TorchScript
    curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"torchscript\"]}"
)
if "%choice%"=="18" (
    echo [测试18] 同时转换为 ONNX 和 TorchScript
    curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"onnx\", \"torchscript\"]}"
)
if "%choice%"=="19" (
    echo [测试19] 转换压缩后的模型为ONNX（指定文件）
curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/optimized\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"onnx\"], \"model_file\": \"model_quantized_fp16.pt\"}"    
)
if "%choice%"=="20" (
    echo [测试20] 列出支持的硬件平台
    curl -X GET %API_URL%/list-hardware
)
if "%choice%"=="21" (
    echo [测试21] TensorRT编译（从原始 .pt 文件）
    curl -X POST %API_URL%/compile -H "Content-Type: application/json" -d "{\"model_path\": \"%BASE_DIR%/raw/inceptionv4.pt\", \"result_dir\": \"%BASE_DIR%/compiled\", \"target\": \"tensorrt\", \"options\": {\"optimization\": {\"fp16\": true}}}"
)
if "%choice%"=="22" (
    echo [测试22] TensorRT编译（从 ONNX 文件）
    curl -X POST %API_URL%/compile -H "Content-Type: application/json" -d "{\"model_path\": \"%BASE_DIR%/optimized/inceptionv4.onnx\", \"result_dir\": \"%BASE_DIR%/compiled\", \"target\": \"tensorrt\", \"options\": {\"optimization\": {\"fp16\": true, \"workspace_size\": 4096}}}"
)
if "%choice%"=="23" (
    echo [测试23] Ascend编译（从原始 .pt 文件）
    curl -X POST %API_URL%/compile -H "Content-Type: application/json" -d "{\"model_path\": \"%BASE_DIR%/raw/inceptionv4.pt\", \"result_dir\": \"%BASE_DIR%/compiled\", \"target\": \"ascend\", \"options\": {\"device\": \"Ascend 310\", \"input_shape\": \"1,3,299,299\"}}"
)
if "%choice%"=="24" (
    echo [测试24] Ascend编译（从 ONNX 文件）
    curl -X POST %API_URL%/compile -H "Content-Type: application/json" -d "{\"model_path\": \"%BASE_DIR%/optimized/inceptionv4.onnx\", \"result_dir\": \"%BASE_DIR%/compiled\", \"target\": \"ascend\", \"options\": {\"device\": \"Ascend 310\", \"input_shape\": \"1,3,299,299\", \"input_format\": \"NCHW\"}}"
)

echo.
echo.
pause

