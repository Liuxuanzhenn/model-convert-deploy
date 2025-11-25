# 模型转换和部署功能

一个完整的深度学习模型压缩、转换和部署系统，支持模型上传、模型压缩、格式转换和硬件编译。

---

## 📋 目录

- [一、项目结构](#一项目结构)
- [二、模型上传模块](#二模型上传模块)
- [三、模型压缩模块](#三模型压缩模块)
- [四、格式转换模块](#四格式转换模块)
- [五、API接口](#五api接口)
- [六、运行和测试](#六运行和测试)

---

## 一、项目结构

### 1.1 完整目录结构

```
模型转换和部署功能/
├── adapters/                          # 模型适配器（核心模块）
│   ├── base.py                        # 适配器基类，定义统一接口
│   ├── registry.py                    # 适配器注册表，动态注册和查找
│   ├── pytorch_yolo.py                # YOLO模型适配器
│   ├── pytorch_resnet.py              # ResNet模型适配器
│   ├── pytorch_vgg.py                 # VGG模型适配器
│   ├── pytorch_vit.py                 # Vision Transformer适配器
│   ├── pytorch_inceptionv4.py        # InceptionV4适配器
│   ├── pytorch_cnn.py                 # 通用CNN适配器
│   ├── pytorch_transformer.py        # Transformer适配器
│   ├── pytorch_lstm.py                # LSTM适配器
│   ├── pytorch_rnn.py                 # RNN适配器
│   ├── pytorch_vae.py                 # VAE适配器
│   ├── pytorch_van.py                 # VAN适配器
│   ├── pytorch_gcn.py                 # GCN适配器
│   ├── pytorch_generic.py             # PyTorch通用适配器
│   ├── tensorflow_generic.py          # TensorFlow适配器
│   ├── paddle_generic.py              # PaddlePaddle适配器
│   ├── onnx_generic.py                # ONNX适配器
│   └── classic.py                     # 传统ML模型适配器（K-Means/DBSCAN/谱聚类）
│
├── api/                               # API接口模块
│   ├── compression.py                 # 压缩API接口（/detect-capabilities, /execute）
│   ├── upload.py                      # 文件上传API接口（/upload-extra-files）
│   ├── method_mapper.py               # 方法映射器（API method → strategy）
│   └── schemas.py                     # Pydantic数据验证Schema
│
├── app/                               # Flask应用
│   └── server.py                      # Flask服务器启动脚本
│
├── core/                              # 核心业务逻辑模块
│   ├── engine.py                      # 核心优化引擎（execute_optimize, execute_compile）
│   ├── enums.py                       # 枚举定义
│   └── exceptions.py                  # 异常定义
│
├── services/                          # 服务模块
│   ├── model.py                       # 模型检测和验证服务（ModelDetector, TeacherValidator）
│   ├── files.py                       # 额外文件管理服务（ExtraFilesManager）
│   ├── compression.py                 # 压缩服务统一导入接口
│   ├── estimator.py                   # 压缩效果预估器
│   ├── recommender.py                 # 策略推荐引擎
│   └── validator.py                   # 配置验证器
│
├── strategies/                        # 压缩策略模块
│   ├── quant/                         # 量化策略
│   │   ├── ptq.py                     # 后训练量化（FP16/INT8动态/INT8静态）
│   │   ├── qat.py                     # 量化感知训练
│   │   └── auto.py                    # 自动量化策略选择器
│   ├── prune/                         # 剪枝策略
│   │   ├── structured.py              # 结构化剪枝
│   │   ├── unstructured.py            # 非结构化剪枝
│   │   ├── auto.py                    # 自动剪枝策略选择器
│   │   └── finetune.py                # 剪枝后微调
│   ├── distill/                       # 知识蒸馏策略
│   │   ├── kd_cls.py                  # 分类任务蒸馏
│   │   ├── kd_det_stub.py             # 检测任务蒸馏（占位）
│   │   └── strategy.py                # 蒸馏策略选择器
│   └── common.py                      # 公共工具函数
│
├── compilers/                         # 硬件编译器
│   ├── base.py                        # 编译器基类
│   ├── registry.py                    # 编译器注册表
│   ├── tensorrt.py                    # NVIDIA TensorRT编译器
│   ├── ascend.py                      # 华为昇腾NPU编译器
│   ├── cambricon.py                   # 寒武纪MLU编译器
│   └── m9.py                          # 天数智芯M9编译器（占位）
│
├── compression/                       # 压缩能力配置模块
│   └── capabilities_v2.py             # 模型能力注册表（加载model_capabilities.json）
│
├── config/                            # 配置模块
│   ├── settings.py                    # 应用配置管理
│   ├── logging.py                     # 日志配置
│   └── swagger.py                     # Swagger API文档配置
│
├── configs/                           # 配置文件目录
│   └── model_capabilities.json         # 模型能力配置（定义支持的压缩方法和导出格式）
│
├── utils/                             # 工具模块
│   ├── path.py                        # 路径管理工具（PathManager）
│   ├── file.py                        # 文件操作工具
│   ├── error.py                       # 错误处理和错误码定义
│   ├── security.py                    # 安全工具（路径清理、输入验证）
│   └── data.py                        # 数据预处理工具
│
├── evaluators/                        # 模型评估器
│   ├── size.py                        # 模型大小评估
│   ├── latency.py                     # 延迟评估
│   └── accuracy_stub.py               # 精度评估（占位实现）
│
├── storage/                           # 数据存储（运行时生成）
│   ├── models_db.json                 # 模型数据库
│   ├── jobs_db.json                   # 任务数据库
│   └── logs/                          # 日志目录
│
├── artifacts/                         # 测试产物目录
│   └── new_test_upload/               # 测试上传的模型和结果
│
├── requirements.txt                   # Python依赖
├── README.md                          # 项目文档（本文件）
└── CODE_QUALITY_REPORT.md             # 代码质量报告
```

### 1.2 核心模块说明

| 模块 | 作用 | 关键文件 |
|------|------|---------|
| **adapters** | 为不同框架和模型架构提供统一接口 | `base.py`, `registry.py`, `pytorch_*.py` |
| **api** | 提供RESTful API接口 | `compression.py`, `upload.py` |
| **core** | 核心业务逻辑，执行优化和编译 | `engine.py` |
| **services** | 业务服务（模型检测、文件管理、预估推荐） | `model.py`, `files.py`, `estimator.py` |
| **strategies** | 压缩策略实现（量化/剪枝/蒸馏） | `quant/`, `prune/`, `distill/` |
| **compilers** | 硬件编译器（TensorRT/Ascend/Cambricon） | `tensorrt.py`, `ascend.py`, `cambricon.py` |
| **compression** | 模型能力配置管理 | `capabilities_v2.py` |
| **configs** | 配置文件 | `model_capabilities.json` |

---

## 二、模型上传模块

### 2.1 支持的模型格式

系统支持以下模型格式的上传和识别：

#### 2.1.1 PyTorch格式
- **`.pt`** - PyTorch模型文件（state_dict或完整模型）
- **`.pth`** - PyTorch模型文件（同.pt）
- **`.safetensors`** - SafeTensors格式（安全序列化）

#### 2.1.2 TensorFlow格式
- **`.pb`** - Protocol Buffer格式（冻结图）
- **`.h5`** - Keras HDF5格式
- **`.ckpt`** - TensorFlow检查点格式
- **`savedmodel/`** - SavedModel目录（包含saved_model.pb）

#### 2.1.3 ONNX格式
- **`.onnx`** - ONNX模型文件

#### 2.1.4 PaddlePaddle格式
- **`.pdmodel`** - PaddlePaddle模型定义文件
- **`.pdparams`** - PaddlePaddle模型参数文件

#### 2.1.5 传统机器学习格式
- **`.pkl`** - Pickle格式（sklearn模型）
- **`.joblib`** - Joblib格式（sklearn模型）

### 2.2 目录结构说明

系统使用三个核心目录进行数据交换：

#### 2.2.1 model_dir（模型目录）

**用途**：存储用户上传的原始模型文件

**位置**：由上传模块或系统其他模块提供

**结构示例**：
```
model_dir/
├── yolov8n.pt              # PyTorch模型文件
└── config.json             # 可选：模型配置文件
```

**说明**：
- 系统会自动识别`model_dir`中的模型格式和类型
- 支持单个模型文件或多个相关文件（如`.pdmodel`和`.pdparams`）
- 识别逻辑：`services/model.py` → `ModelDetector.detect_from_dir()`

#### 2.2.2 extra_dir（额外文件目录）

**用途**：存储用户提供的额外文件（校准数据、训练数据、验证数据、教师模型等）

**位置**：用户通过`/upload-extra-files` API上传zip文件后自动解压到此目录

**结构约定**：
```
extra_dir/
├── calibration_data/            # 校准数据（INT8静态量化）
│   └── images/                  # ImageFolder格式
│       ├── class1/
│       └── class2/
├── train_data/                  # 训练数据（QAT、蒸馏）
│   └── images/                  # ImageFolder格式
├── val_data/                    # 验证数据（剪枝评估）
│   └── images/                  # ImageFolder格式
├── teacher_model/               # 教师模型（知识蒸馏）
│   └── teacher.pt               # 教师模型文件
└── metadata/                    # 元数据文件（可选）
    └── config.json
```

**管理方式**：
- 使用`services/files.py`中的`ExtraFilesManager`类统一管理
- 自动检查必需文件是否存在
- 提供文件列表查询功能
- 支持zip文件自动解压和识别

#### 2.2.3 res_dir（结果目录）

**用途**：存储压缩和转换后的模型文件、导出格式文件、评估指标等

**位置**：由系统自动创建或由调用方指定

**结构示例**：
```
res_dir/
├── model_quantized_fp16.pt      # 量化后的模型
├── model_pruned_30pct.pt        # 剪枝后的模型
├── model_quantized_pruned.pt    # 量化+剪枝后的模型
├── model.onnx                   # ONNX格式导出
├── model.torchscript.pt         # TorchScript格式导出
├── metrics.json                 # 评估指标
└── compiled/                    # 硬件编译结果（可选）
    ├── tensorrt/
    │   └── model.engine
    └── ascend/
        └── model.om
```

**说明**：
- 目录结构由系统自动组织
- 所有产物文件都会记录在`artifacts`列表中
- 评估指标保存在`metrics.json`中

### 2.3 历史版本查询

**当前状态**：系统支持模型和任务的历史记录

**实现方式**：
- `storage/models_db.json`：存储模型元信息
- `storage/jobs_db.json`：存储任务执行记录

**查询方式**：
- 通过`model_id`和`version_id`标识模型版本
- 每个压缩任务都会生成唯一的`job_id`
- 可通过`job_id`查询任务执行历史和结果

**未来扩展**：
- 计划支持通过API查询历史版本
- 支持版本对比和回滚功能

---

## 三、模型压缩模块

### 3.1 适配器和支持的模型种类

#### 3.1.1 已实现的适配器

系统通过适配器模式为不同框架和模型架构提供统一接口：

| 适配器文件 | 框架 | 模型家族 | 说明 |
|-----------|------|---------|------|
| `pytorch_yolo.py` | PyTorch | yolo | YOLO检测模型 |
| `pytorch_resnet.py` | PyTorch | resnet | ResNet分类模型 |
| `pytorch_vgg.py` | PyTorch | vgg | VGG分类模型 |
| `pytorch_vit.py` | PyTorch | vit | Vision Transformer |
| `pytorch_inceptionv4.py` | PyTorch | inceptionv4 | InceptionV4模型 |
| `pytorch_cnn.py` | PyTorch | cnn | 通用CNN模型 |
| `pytorch_transformer.py` | PyTorch | transformer | Transformer模型 |
| `pytorch_lstm.py` | PyTorch | lstm | LSTM时序模型 |
| `pytorch_rnn.py` | PyTorch | rnn | RNN时序模型 |
| `pytorch_vae.py` | PyTorch | vae | 变分自编码器 |
| `pytorch_van.py` | PyTorch | van | Vision Attention Network |
| `pytorch_gcn.py` | PyTorch | gcn | 图卷积网络 |
| `pytorch_generic.py` | PyTorch | generic | PyTorch通用适配器 |
| `tensorflow_generic.py` | TensorFlow | generic | TensorFlow通用适配器 |
| `paddle_generic.py` | PaddlePaddle | generic | PaddlePaddle通用适配器 |
| `onnx_generic.py` | ONNX | generic | ONNX通用适配器 |
| `classic.py` | sklearn | kmeans/dbscan/spectral_clustering | 传统ML模型 |

#### 3.1.2 模型识别流程

1. **Framework识别**：根据文件扩展名自动识别
   - `.pt`, `.pth`, `.safetensors` → `pytorch`
   - `.pb`, `.h5`, `.ckpt`, `savedmodel/` → `tensorflow`
   - `.onnx` → `onnx`
   - `.pdmodel`, `.pdparams` → `paddlepaddle`
   - `.pkl`, `.joblib` → `sklearn`

2. **Family识别**：加载模型对象，分析模型结构
   - 使用generic适配器加载模型
   - 分析模型类名和结构字符串
   - 匹配已知的模型家族特征
   - 返回识别的family（如yolo/resnet/vgg等）

3. **原始格式识别**：结合文件扩展名和framework
   - 优先根据文件扩展名判断
   - 如果无法确定，根据framework推断默认格式

**代码位置**：`services/model.py` → `ModelDetector.detect_from_dir()`

### 3.2 压缩后输出格式

**核心原则**：**压缩后保持原格式输出**

系统会自动识别原始格式，压缩后的模型**始终使用相同格式保存**：

| 原始格式（从model_dir识别） | 压缩后输出格式 | 说明 |
|---------------------------|--------------|------|
| `.pt`, `.pth` | `model_quantized.pt` | PyTorch格式 |
| `.safetensors` | `model_quantized.safetensors` | SafeTensors格式 |
| `.onnx` | `model_quantized.onnx` | ONNX格式 |
| `.pb`, `savedmodel/` | `model_quantized.pb` | TensorFlow格式 |
| `.pdmodel`, `.pdparams` | `model_quantized.pdmodel/pdparams` | PaddlePaddle格式 |
| `.pkl`, `.joblib` | `model_quantized.pkl` | Pickle格式 |

**重要说明**：
- ✅ **压缩模块不提供格式转换功能**
- ✅ 压缩后的模型格式与输入格式一致
- ✅ 如需格式转换（如`.pt`转`.onnx`），需要在**格式转换模块**中单独处理

**实现逻辑**：
- 系统从`model_dir`识别`original_format`
- 压缩后自动使用`original_format`保存
- 文件名格式：`model_{操作}.{原格式扩展名}`

**代码位置**：
- 格式识别：`services/model.py` → `detect_original_format()`
- 格式保存：`core/engine.py` → `execute_optimize()`

### 3.3 压缩方法可复选

**支持情况**：✅ **完全支持**

用户可以同时选择多个压缩方法，系统会按顺序执行：

#### 3.3.1 支持的组合

| 组合 | 执行顺序 | 说明 |
|------|---------|------|
| 量化 + 剪枝 | 剪枝 → 量化 | 先剪枝再量化，效果叠加 |
| 量化 + 蒸馏 | 量化 → 蒸馏 | 先量化再蒸馏 |
| 剪枝 + 蒸馏 | 剪枝 → 蒸馏 | 先剪枝再蒸馏 |
| 量化 + 剪枝 + 蒸馏 | 量化 → 剪枝 → 蒸馏 | 三种方法组合 |

#### 3.3.2 执行流程

```
用户选择：量化（FP16）+ 剪枝（结构化，30%稀疏度）
    ↓
1. 执行量化：yolov8n.pt → model_quantized_fp16.pt（保持.pt格式）
    ↓
2. 执行剪枝：model_quantized_fp16.pt → model_quantized_pruned.pt（保持.pt格式）
    ↓
3. 评估指标：生成metrics.json
```

**注意**：
- 压缩后输出格式与输入格式一致（`.pt` → `.pt`）
- 如需转换为其他格式（如`.onnx`），需要在**格式转换模块**中单独处理

**代码位置**：`core/engine.py` → `execute_optimize()`

### 3.4 自动量化/剪枝/蒸馏的实现依据

#### 3.4.1 自动量化（Auto Quantization）

**实现依据**：基于**模型类型（family）**和**可用资源**自动选择最优量化方法

**选择逻辑**（`strategies/quant/auto.py`）：

| 模型类型 | 自动选择方法 | 依据 |
|---------|------------|------|
| LSTM/RNN | INT8动态量化（仅Linear层） | 保留LSTM/RNN层为FP32，避免精度损失 |
| GCN | INT8动态量化（仅Linear层） | 保留GraphConv层为FP32 |
| VAE | 混合策略（encoder INT8 + decoder FP16） | 编码器可量化，解码器需要更高精度 |
| Transformer | INT8动态量化（注意力感知） | Transformer结构适合动态量化 |
| 视觉模型（YOLO/ResNet/VGG等） | 有校准数据→INT8静态<br>无校准数据→INT8动态 | 校准数据可提升精度 |
| 指定bits=16 | FP16量化 | 用户明确指定16位精度 |

**代码位置**：`strategies/quant/auto.py` → `decide_and_apply_quant()`

#### 3.4.2 自动剪枝（Auto Pruning）

**实现依据**：基于**模型类型（family）**自动选择剪枝方法

**选择逻辑**（`strategies/prune/auto.py`）：

| 模型类型 | 自动选择方法 | 依据 |
|---------|------------|------|
| Transformer/ViT/BERT | 非结构化剪枝 | Transformer结构更适合非结构化剪枝 |
| CNN模型（ResNet/VGG/YOLO等） | 结构化剪枝 | CNN结构更适合结构化剪枝，硬件友好 |
| 其他 | 结构化剪枝（默认） | 通用选择 |

**稀疏度选择**：
- 如果指定了`flops_reduction`或`search_space`，使用`select_sparsity()`智能选择
- 否则使用默认值0.3（30%稀疏度）

**代码位置**：`strategies/prune/auto.py` → `decide_and_apply_prune()`

#### 3.4.3 自动蒸馏（Auto Distillation）

**当前状态**：⚠️ **部分支持**

**实现依据**：
- 基于任务类型（分类/检测）选择蒸馏方法
- 分类任务：使用`kd_cls.py`
- 检测任务：使用`kd_det_stub.py`（占位实现）

**代码位置**：`strategies/distill/strategy.py`

### 3.5 额外文件上传（Zip格式）

#### 3.5.1 上传格式要求

**格式**：**必须是zip文件**

**API接口**：`POST /upload-extra-files`

**请求参数**：
- `file`：zip文件（multipart/form-data）
- `extra_dir`：目标目录路径

#### 3.5.2 Zip文件结构示例

**标准结构**：
```
extra_files.zip
├── calibration_data/          # 校准数据目录
│   └── images/
│       ├── class1/
│       │   ├── img1.jpg
│       │   └── img2.jpg
│       └── class2/
│           ├── img3.jpg
│           └── img4.jpg
├── train_data/                # 训练数据目录
│   └── images/
│       ├── class1/
│       └── class2/
├── val_data/                  # 验证数据目录
│   └── images/
│       ├── class1/
│       └── class2/
└── teacher_model/             # 教师模型目录
    └── teacher.pt
```

**识别规则**：
- 系统根据**顶层目录名称**自动识别文件类型
- 支持的目录名称（不区分大小写）：
  - `calibration_data`, `calib`, `calibration` → `calibration_data/`
  - `train_data`, `train`, `training` → `train_data/`
  - `val_data`, `val`, `validation`, `valid` → `val_data/`
  - `teacher_model`, `teacher` → `teacher_model/`

**代码位置**：`services/files.py` → `ExtraFilesManager.extract_and_distribute()`

#### 3.5.3 Zip识别后的功能变化

**识别成功后**：

1. **文件自动解压**：zip文件内容自动解压到`extra_dir`的对应子目录

2. **可选的新压缩技术**：
   - ✅ **INT8静态量化**：如果识别到`calibration_data/`，可以选择INT8静态量化
   - ✅ **QAT量化感知训练**：如果识别到`train_data/`，可以选择QAT
   - ✅ **知识蒸馏**：如果识别到`teacher_model/`和`train_data/`，可以选择知识蒸馏
   - ✅ **剪枝评估**：如果识别到`val_data/`，剪枝时可以评估精度损失

3. **方法可用性更新**：
   - `/detect-capabilities` API会返回`method_availability`字段
   - 显示哪些方法现在可用（`available: true/false`）
   - 显示哪些方法有回退选项（`fallback`）

**代码位置**：`api/compression.py` → `detect_capabilities()`

#### 3.5.4 自动方法在文件上传前后的区别

| 场景 | 上传文件前 | 上传文件后 |
|------|-----------|-----------|
| **自动量化** | 视觉模型：INT8动态量化（无校准数据） | 视觉模型：INT8静态量化（有校准数据，精度更高） |
| **自动剪枝** | 使用默认稀疏度0.3，无精度评估 | 可以使用`val_data/`评估精度损失，调整稀疏度 |
| **知识蒸馏** | 不可用（缺少必需文件） | 可用（有`teacher_model/`和`train_data/`） |
| **QAT量化** | 不可用（缺少训练数据） | 可用（有`train_data/`） |

**示例**：

**上传文件前**：
```json
{
  "method_availability": {
    "quantize.int8_static": {
      "available": true,
      "fallback": "int8_dynamic",
      "optional_files_status": {
        "calibration_data": false
      }
    }
  }
}
```

**上传文件后**（识别到calibration_data）：
```json
{
  "method_availability": {
    "quantize.int8_static": {
      "available": true,
      "fallback": null,
      "optional_files_status": {
        "calibration_data": true
      }
    }
  }
}
```

---

## 四、格式转换模块

### 4.1 功能定位

格式转换是**独立于模型压缩**的模块，用于将模型从一种格式转换为另一种格式，主要用于硬件编译准备。

**转换流程**：
```
压缩后的模型（原格式） → 格式转换 → 目标格式 → 硬件编译（可选）
```

### 4.2 支持的格式转换

| 源格式 | 目标格式 | 实现状态 | 说明 |
|--------|---------|---------|------|
| `.pt`, `.pth` | `.onnx` | ✅ 已实现 | PyTorch转ONNX |
| `.pt`, `.pth` | `.torchscript` | ✅ 已实现 | PyTorch转TorchScript |
| `.pb`, `savedmodel/` | `.onnx` | ✅ 已实现 | TensorFlow转ONNX（需tf2onnx） |
| `.pdmodel`, `.pdparams` | `.onnx` | ✅ 已实现 | PaddlePaddle转ONNX（需paddle2onnx） |
| `.safetensors` | `.onnx` | ⚠️ 部分支持 | 需完整模型，不支持state_dict |
| `.onnx` | `.pt` | ⚠️ 部分支持 | ONNX转PyTorch |

**代码位置**：
- PyTorch转换：`adapters/base.py` → `_export_onnx()`, `_export_torchscript()`
- TensorFlow转换：`adapters/tensorflow_generic.py` → `export()`
- PaddlePaddle转换：`adapters/paddle_generic.py` → `export()`

### 4.3 硬件编译支持

#### 4.3.1 支持的硬件编译器

| 硬件编译器 | 输出格式 | 工具依赖 |
|-----------|---------|---------|
| **TensorRT** | `.engine` | NVIDIA TensorRT SDK |
| **昇腾NPU** | `.om` | 华为ATC工具 |
| **寒武纪MLU** | `.cambricon` | 寒武纪CNCC工具 |
| **天数智芯M9** | `.m9` | M9 SDK（占位） |

#### 4.3.2 输入格式要求

**所有硬件编译器都需要ONNX格式作为输入**，系统会自动处理格式转换：

| model_dir中的格式 | 硬件编译支持 | 转换方式 |
|-----------------|------------|---------|
| `.onnx` | ✅ 直接支持 | 直接传递给硬件编译器 |
| `.pt`, `.pth` | ✅ 自动转换 | 自动调用`torch.onnx.export()`转ONNX |
| `.pb`, `savedmodel/` | ✅ 需先转ONNX | 通过格式转换模块转ONNX |
| `.pdmodel`, `.pdparams` | ✅ 需先转ONNX | 通过格式转换模块转ONNX |
| `.safetensors` | ⚠️ 需完整模型 | 需先转PyTorch再转ONNX |
| `.pkl`, `.joblib` | ❌ 不支持 | 传统ML模型不支持硬件编译 |

**代码位置**：`compilers/base.py` → `_convert_pytorch_to_onnx()`, `_detect_input_format()`

### 4.4 完整工作流程

**示例：压缩 + 格式转换 + 硬件编译**

```
步骤1：模型压缩（保持原格式）
yolov8n.pt → [量化+剪枝] → model_quantized_pruned.pt

步骤2：格式转换（独立模块）
model_quantized_pruned.pt → [pt→onnx] → model_quantized_pruned.onnx

步骤3：硬件编译（可选）
model_quantized_pruned.onnx → [TensorRT] → model.engine
```

**说明**：
- 压缩后格式与输入格式一致（`.pt` → `.pt`）
- 格式转换是独立步骤，需要单独调用
- 硬件编译需要ONNX格式，系统会自动转换支持的格式

---

## 五、API接口

### 5.1 检测模型能力

**接口**：`POST /detect-capabilities`

**功能**：检测模型支持的压缩操作、导出格式和额外文件可用性

**请求参数**：
```json
{
  "model_dir": "/path/to/model",
  "extra_dir": "/path/to/extra"  // 可选
}
```

**返回结果**：
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "framework": "pytorch",
    "family": "yolo",
    "original_format": "pt",
    "supported_operations": {
      "quantize": {
        "enabled": true,
        "methods": ["fp16", "int8_dynamic", "int8_static", "qat"],
        "recommended": "fp16"
      },
      "prune": {
        "enabled": true,
        "methods": ["structured_pruning", "unstructured_pruning"],
        "recommended": "structured_pruning"
      }
    },
    "operation_requirements": {
      "quantize": {
        "int8_static": {
          "required_extra_files": [],
          "optional_extra_files": ["calibration_data"]
        }
      }
    },
    "available_files": {
      "calibration_data": ["image1.jpg", "image2.jpg"],
      "train_data": ["train1.jpg"]
    },
    "method_availability": {
      "quantize.int8_static": {
        "available": true,
        "fallback": "int8_dynamic",
        "optional_files_status": {
          "calibration_data": true
        }
      }
    }
  }
}
```

**代码位置**：`api/compression.py` → `detect_capabilities()`

### 5.2 上传额外文件

**接口**：`POST /upload-extra-files`

**功能**：上传zip文件，自动解压并识别文件类型

**请求参数**：
- `file`：zip文件（multipart/form-data）
- `extra_dir`：目标目录路径

**返回结果**：
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "extra_dir": "/path/to/extra",
    "recognized_files": {
      "calibration_data": ["image1.jpg", "image2.jpg"],
      "train_data": ["train1.jpg"]
    },
    "file_count": 3
  }
}
```

**代码位置**：`api/upload.py` → `upload_extra_files()`

### 5.3 执行压缩操作

**接口**：`POST /execute`

**功能**：执行模型压缩和格式转换

**请求参数**：
```json
{
  "model_dir": "/path/to/model",
  "result_dir": "/path/to/result",
  "extra_dir": "/path/to/extra",
  "method": {
    "quantize": {
      "enable": true,
      "precision": "fp16"
    },
    "prune": {
      "enable": true,
      "type": "structured",
      "target_sparsity": 0.3
    }
  }
}
```

**注意**：压缩模块不包含`export_formats`参数，压缩后自动保持原格式。

**返回结果**：
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "job_id": "j_xxx",
    "result_dir": "/path/to/result",
    "artifacts": [
      "/path/to/result/model_quantized_fp16.pt",
      "/path/to/result/model.onnx"
    ],
    "metrics": {
      "size_before_mb": 12.2,
      "size_after_mb": 6.1,
      "latency_ms_cpu": 25.5
    }
  }
}
```

**代码位置**：`api/compression.py` → `execute_compression()`

---

## 六、运行和测试

### 6.1 安装依赖

```bash
pip install -r requirements.txt
```

### 6.2 启动服务

```bash
python -m app.server
```

服务默认运行在：`http://localhost:5000`

### 6.3 测试示例

#### 测试1：检测模型能力

```bash
curl -X POST http://localhost:5000/detect-capabilities ^
  -H "Content-Type: application/json" ^
  -d "{\"model_dir\": \"D:/path/to/model\"}"
```

#### 测试2：上传额外文件

```bash
curl -X POST http://localhost:5000/upload-extra-files ^
  -F "file=@extra_files.zip" ^
  -F "extra_dir=D:/path/to/extra"
```

#### 测试3：执行压缩操作

```bash
curl -X POST http://localhost:5000/execute ^
  -H "Content-Type: application/json" ^
  -d "{\"model_dir\": \"D:/path/to/model\", \"result_dir\": \"D:/path/to/result\", \"method\": {\"quantize\": {\"enable\": true, \"precision\": \"fp16\"}}, \"export_formats\": [\"pt\", \"onnx\"]}"
```

### 6.4 注意事项

1. **目录路径**：确保`model_dir`、`res_dir`、`extra_dir`路径正确
2. **文件格式**：`extra_dir`中的文件需要按照约定目录结构组织
3. **模型格式**：支持PyTorch、TensorFlow、ONNX等格式
4. **额外文件**：某些压缩方法需要额外文件，请提前准备
5. **硬件编译**：需要安装对应的硬件SDK（TensorRT/Ascend/Cambricon）

---

**最后更新**：2024年12月
