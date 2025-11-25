@echo off
chcp 65001 >nul
echo ========================================
echo LSTM 模型完整测试脚本
echo ========================================
echo.

REM 设置基础路径
set BASE_DIR=D:/项目空间/image_classification_inceptionv4/模型转换和部署功能/artifacts/m_lstm_test/v_001
set API_URL=http://localhost:5000
set WEIGHTS_DIR=D:/项目空间/image_classification_inceptionv4/模型转换和部署功能/weights/pytorch/lstm

REM 创建测试目录结构
echo [准备] 创建测试目录结构...
if not exist "%BASE_DIR%\raw" mkdir "%BASE_DIR%\raw"
if not exist "%BASE_DIR%\optimized" mkdir "%BASE_DIR%\optimized"
if not exist "%BASE_DIR%\extra" mkdir "%BASE_DIR%\extra"

REM 复制LSTM模型文件
echo [准备] 复制LSTM模型文件...
if exist "%WEIGHTS_DIR%\lstm.pth" (
    copy "%WEIGHTS_DIR%\lstm.pth" "%BASE_DIR%\raw\" >nul
    echo ✅ 模型文件已复制到: %BASE_DIR%\raw\lstm.pth
) else (
    echo ❌ 未找到模型文件: %WEIGHTS_DIR%\lstm.pth
    echo 请确保LSTM模型文件存在
    pause
    exit /b 1
)
echo.

echo ========================================
echo 请选择要执行的测试：
echo ========================================
echo.
echo 1. 检测模型能力（基础）
echo 2. 检测模型能力（带extra_dir）
echo 3. 自动量化（LSTM专用：INT8动态，仅Linear层）
echo 4. INT8动态量化（显式指定）
echo 5. FP16量化（通用方法）
echo 6. 自动剪枝（LSTM专用：非结构化，仅Linear层）
echo 7. 非结构化剪枝（20%%稀疏度）
echo 8. 非结构化剪枝（30%%稀疏度）
echo 9. 组合压缩：自动剪枝 + 自动量化
echo 10. 组合压缩：剪枝20%% + INT8动态量化
echo 11. 组合压缩：剪枝30%% + FP16量化
echo 12. 转换为ONNX格式
echo 13. 转换压缩后的模型为ONNX
echo 14. 列出支持的格式
echo 15. 列出支持的硬件平台
echo.
set /p choice=请输入测试编号（1-15）: 

if "%choice%"=="1" (
    echo [测试1] 检测模型能力（基础）
    curl -X POST %API_URL%/detect-capabilities -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\"}"
)
if "%choice%"=="2" (
    echo [测试2] 检测模型能力（带extra_dir）
    curl -X POST %API_URL%/detect-capabilities -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"extra_dir\": \"%BASE_DIR%/extra\"}"
)
if "%choice%"=="3" (
    echo [测试3] 自动量化（LSTM专用：INT8动态，仅Linear层）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"quantize\": {\"enable\": true, \"auto\": true}}}"
)
if "%choice%"=="4" (
    echo [测试4] INT8动态量化（显式指定）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"int8_dynamic\"}"
)
if "%choice%"=="5" (
    echo [测试5] FP16量化（通用方法）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": \"fp16\"}"
)
if "%choice%"=="6" (
    echo [测试6] 自动剪枝（LSTM专用：非结构化，仅Linear层）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"type\": \"auto\"}}}"
)
if "%choice%"=="7" (
    echo [测试7] 非结构化剪枝（20%%稀疏度）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"type\": \"unstructured\", \"target_sparsity\": 0.2}}}"
)
if "%choice%"=="8" (
    echo [测试8] 非结构化剪枝（30%%稀疏度）
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"type\": \"unstructured\", \"target_sparsity\": 0.3}}}"
)
if "%choice%"=="9" (
    echo [测试9] 组合压缩：自动剪枝 + 自动量化
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"type\": \"auto\", \"target_sparsity\": 0.2}, \"quantize\": {\"enable\": true, \"auto\": true}}}"
)
if "%choice%"=="10" (
    echo [测试10] 组合压缩：剪枝20%% + INT8动态量化
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"type\": \"unstructured\", \"target_sparsity\": 0.2}, \"quantize\": {\"enable\": true, \"precision\": \"int8_dynamic\"}}}"
)
if "%choice%"=="11" (
    echo [测试11] 组合压缩：剪枝30%% + FP16量化
    curl -X POST %API_URL%/execute -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"method\": {\"prune\": {\"enable\": true, \"type\": \"unstructured\", \"target_sparsity\": 0.3}, \"quantize\": {\"enable\": true, \"precision\": \"fp16\"}}}"
)
if "%choice%"=="12" (
    echo [测试12] 转换为ONNX格式
    curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"onnx\"]}"
)
if "%choice%"=="13" (
    echo [测试13] 转换压缩后的模型为ONNX
    curl -X POST %API_URL%/convert-format -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/optimized\", \"result_dir\": \"%BASE_DIR%/optimized\", \"target_formats\": [\"onnx\"], \"model_file\": \"model_pruned_20pct.pt\"}"
)
if "%choice%"=="14" (
    echo [测试14] 列出支持的格式
    curl -X POST %API_URL%/list-supported-formats -H "Content-Type: application/json" -d "{\"model_dir\": \"%BASE_DIR%/raw\"}"
)
if "%choice%"=="15" (
    echo [测试15] 列出支持的硬件平台
    curl -X GET %API_URL%/list-hardware
)

echo.
echo.
pause

