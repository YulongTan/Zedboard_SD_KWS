# KWS 权重转换工具

本目录提供两个 Python 脚本，帮助你把 PyTorch 训练得到的 `BNN_KWS` 模型权重依次导出为可读的文本文件，再转换成适用于 Zynq PS 端推理的 `kws_weights.bin`。

## 1. 从 `.pt` 到 `.txt`/`.bin`

```bash
python3 export_kws_weights.py \
    path/to/kws_checkpoint.pt \
    --txt-out kws_weights.txt \
    --bin-out kws_weights.bin \
    [--bin-first] [--bin-last]
```

脚本会：

1. 读取 PyTorch `state_dict`（支持 `checkpoint["model"]` 格式）；
2. 按 `kws_engine.c` 期望的顺序提取各卷积/全连接权重及折叠后的 BatchNorm 斜率与偏置；
3. 将所有浮点值写入 `kws_weights.txt`，便于人工检查或在导出到 `.bin` 前做微调；
4. 同步生成小端序的 `kws_weights.bin`，文件头包含魔数 `0x4B575331`、版本 `0x00010000` 和类别数。

> **提示**：如果训练时启用了 `--bin-first`，记得在导出时带上同名开关，以便脚本读取正确的权重张量。固件侧的全连接输出保持浮点实现，因此暂不支持 `--bin-last` 导出的完全二值分类头。

## 2. 从 `.txt` 再生成 `.bin`

若你只想基于文本结果进行修改，然后重新生成二进制文件，可单独运行：

```bash
python3 txt_to_bin.py kws_weights.txt --bin-out kws_weights.bin
```

脚本会解析 `magic/version/num_classes/reserved` 头部以及每个 `section <name> <count>` 段落，并用 **小端浮点** (`<f`) 写入 `kws_weights.bin`，完全符合 Zynq A9 的内存布局要求——低地址保存最低有效字节。

## 文本格式说明

- 以 `#` 开头的行会被忽略，可用于备注；
- 头部字段依次为 `magic`、`version`、`num_classes`、`reserved`；
- 每个段落以 `section 名称 元素个数` 开始，随后的 `元素个数` 行给出对应的 `float32` 数值。

该格式直接对应 `kws_engine.c` 中的读取逻辑【F:sdk_appsrc/Zedboard_DMA/src/kws/kws_engine.c†L333-L413】；默认卷积核、BatchNorm 和全连接层的排布与固件保持一致，确保生成的 `kws_weights.bin` 可以被 PS 端应用直接加载。
