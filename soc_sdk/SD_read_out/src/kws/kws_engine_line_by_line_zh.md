# `kws_engine.c` 按行解析

> 行号基于当前仓库版本，便于与源码交叉参考。

## 头文件与宏常量（L1-L52）
- L1 引入自身头文件 `kws_engine.h`，导出公共 API 与结构体声明。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L1】
- L3-L22 使用 `xil_printf.h`、`ff.h`、`math.h`、`float.h` 等标准头，为 FatFs、串口日志与 FFT/Mel 变换所需的三角/指数函数提供依赖。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L3-L22】
- L24-L52 设定 DMA 录音与 KWS 模型共享的常量（采样率、FFT/Mel 尺寸、池化输出大小等），保持与 Python 训练脚本一致。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L24-L52】

## 权重 / Scratch 结构体（L47-L115）
- L47-L52 `KwsWeightHeader` 描述 SD 卡权重文件的魔数、版本与类别数。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L47-L52】
- L54-L75 `KwsModel` 保存卷积/BN/全连接层的权重、偏置指针，初始化时逐段分配内存并填充数据。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L54-L75】
- L77-L92 `KwsScratch` 记录推理所需的输入张量、FFT、各层输出缓冲，避免重复 `malloc`。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L77-L92】
- L100-L106 `KwsFeatureTables` 缓存 Hann、sin/cos、Mel 滤波器矩阵，只在首次初始化时计算一次。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L100-L106】
- **权重生成提示**：`tools/export_kws_weights.py` 会先把 PyTorch checkpoint 展平成 `kws_weights.txt`，可读性强，随后你可通过 `tools/txt_to_bin.py` 重新打包成符合小端布局的 `kws_weights.bin`，完全契合这里定义的结构体字段顺序。【F:sdk_appsrc/Zedboard_DMA/tools/export_kws_weights.py†L1-L165】【F:sdk_appsrc/Zedboard_DMA/tools/txt_to_bin.py†L1-L98】

## 全局状态与前向声明（L108-L157）
- L108-L115 维护 FatFs 对象、模型/工作区句柄以及 `gEngineReady`、`gHasResult` 状态位。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L108-L115】
- L116-L157 声明内部静态函数：从 SD 卡加载权重、分配/释放缓存、提取特征、执行卷积/池化/Dense 层等，为后续实现做准备。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L116-L157】

## 初始化 / 关闭接口（L159-L203）
- `KwsEngine_Initialize`（L159-L190）依次挂载 SD、读取权重、分配 scratch、初始化特征表；任一步失败都会清理已分配资源并返回 `XST_FAILURE`。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L159-L190】
- `KwsEngine_Shutdown`（L192-L198）释放模型与缓存，同时复位状态位，便于热更新模型文件。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L192-L198】
- `KwsEngine_IsReady`（L200-L203）返回引擎就绪标志供主循环查询。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L200-L203】

## 处理录音与取回结果（L205-L266）
- `KwsEngine_ProcessRecording`（L205-L255）是对外主入口：
  - L210-L223 校验引擎是否 ready、录音缓冲是否齐全以及帧数是否满足 decimation 后的一秒长度。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L210-L223】
  - L224-L227 调用 `extract_logmel` 把立体声采样转成 40×98 对数 Mel 频谱；失败时返回错误码。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L224-L227】
  - L229-L237 执行 `run_network` 获取 logits 并找出最大值与类别索引。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L229-L237】
  - L240-L244 以 `expf` 计算 Softmax 分母，输出 1/∑exp 作为置信度并更新 `gHasResult`。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L240-L244】
- `KwsEngine_GetLogits`（L257-L266）在结果有效时返回 logits 指针和类别数量，供调试或上层二次处理。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L257-L266】

## 资源管理与文件系统（L268-L561）
- `free_model`（L268-L286）与 `free_scratch`（L288-L304）逐项释放权重和工作缓冲，防止悬挂指针。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L268-L304】
- `mount_sd_if_needed`（L306-L320）封装 FatFs 挂载逻辑，只在未挂载时调用 `f_mount`。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L306-L320】
- `read_exact`（L322-L333）确保从 SD 卡读满指定字节数，简化权重加载流程中的错误处理。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L322-L333】
- `load_weights`（L335-L460）解析权重文件：校验头部、分配每层参数数组并逐段读取；任一读/分配失败都会清理并返回错误。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L335-L460】
- `allocate_scratch`（L463-L500）根据网络尺寸分配输入张量、FFT、卷积/池化、中间展平、全连接输出等缓冲。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L463-L500】
- `init_feature_tables`（L502-L562）首次调用时使用 `cosf`/`sinf` 计算 Hann 窗与 sin/cos 表，再通过 `log10f`、`powf` 生成 Mel 滤波矩阵，并记录初始化标志。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L502-L562】

## 特征提取（L564-L615）
- `extract_logmel` 把 DMA 缓冲转换成网络输入：
  - L573-L584 按 `KWS_DECIMATION_FACTOR` 抽取并归一化为 16 kHz 单声道波形。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L573-L584】
  - L586-L599 逐帧乘 Hann 窗并使用预计算的 sin/cos 累加得到功率谱。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L586-L599】
  - L602-L612 与 Mel 滤波矩阵点乘，并以 `logf` 写入对数能量（内部会对极小值加上保护），生成 40×98 输入张量。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L602-L612】

## 推理算子实现（L618-L767）
- `conv2d_forward`（L618-L667）实现 3×3 SAME 卷积+BN+可选激活，输出空间尺寸与输入一致。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L618-L667】
- `maxpool2d`（L669-L704）执行 2×2、步长 2 的最大池化，空间尺寸按宏 `POOL_OUT_DIM` 推导。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L669-L704】
- `adaptive_avg_pool`（L706-L739）使用纯整数的区间划分公式 `(oy * in_rows)/out_rows` 与 `((oy+1)*in_rows + out_rows - 1)/out_rows` 计算窗口，再求均值，效果与 PyTorch `AdaptiveAvgPool2d(5,5)` 对齐。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L706-L739】
- `dense_forward`（L741-L767）完成全连接矩阵乘、BN 与激活输出。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L741-L767】

## 整体前向流程（L769-L855）
- `run_network` 串联所有算子：
  - L771-L788 运行 Conv1→ReLU→MaxPool，空间尺寸从 40×98 缩到 20×49，同时将通道扩展至 32。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L771-L788】
  - L789-L817 通过二值 Conv2/Conv3 与池化，把通道扩展到 128，并逐步把空间尺寸压缩至 5×12。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L789-L817】
  - L819-L835 使用自适应平均池化得到 5×5 的特征图并展平成 3200 维向量。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L819-L835】
  - L837-L854 依次执行 `fc1`、`fc_out`，输出 logits 供 Softmax 使用。【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L837-L854】

---
通过以上逐段解析，可将嵌入式 KWS 推理流程与 Python 训练端对齐，并明确各个算子如何在 `math.h` 提供的基础函数配合下复现训练时的数值行为，便于在裸机环境中调试与扩展。
