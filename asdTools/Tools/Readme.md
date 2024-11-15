[README (中文)](README.md)

[README (English)](README_EN.md)

# 工具支持

[toc]

#### 介绍

- 包括以下内容：

  1. 即开即用的工具。
  2. 为快捷方式设计的工具。
  3. 外部调用示例。
- 日志记录：

  1. 支持日志记录，所有输出将默认保存至调用路径下的 ./Logs 中。
  2. 在初始化时加入参数 `log_dir="./OtherLogDir"` 以修改保存文件夹。
  3. 在初始化时加入参数 `log_file="otherLogFile.txt"` 以修改保存日志名称。
  4. 如 `GPUMonitor(log_dir="/root/logs", log_file="GPUMonitor.log")`

#### AI 工具支持

1. GPUMonitor - 调用示例
   - 获取可用GPU的ID。
2. SDStylesBeautify - 快捷方式
   - 将 Stable Diffusion Web UI 的 styles.csv 美化（包括排序，加分类头，去除空行）。

#### API 工具支持

1. Chat_LLMs - 即开即用
   - 用于和LLMs聊天，示例为与ChatGLM本地部署的API通信。

#### Datasets 工具支持

1. RandomDatasetSplitter_SemanticSegmentation - 即开即用
   * 对内部文件名相同的两个文件夹进行随机划分，如语义分割中的RGB图与GT图。

#### Image 工具支持

1. ImageResizer_256x144 - 快捷方式
   - 调整图片的分辨率至256x144，提供了Windows右键菜单快捷方式。
2. ConvertGT2MMSeg - 即开即用
   * 将GT图片转为MMSegmentation所需的图片格式。支持RGB图与灰度图的输入。
3. RGB2Gray3Channel - 即开即用
   * 转换RGB图像为三通道灰度图。
4. ColorImgWithGT - 即开即用
   * 使用彩色分割GT为原图上色。
5. ColorGT - 即开即用
   * 为GT图片上色，支持RGB图与灰度图的输入及输出。
6. VisualizeHeatmapOfReID - 调用示例
   * 可视化热力图，测试于ReID。
7. VisualizeRankOfReID - 调用示例
   * 可视化ReID Rank，使用markdown表格，极快。
8. VisualizeSegGTDistribution - 调用示例
   * 可视化语义分割GT的分布情况。

#### 爬虫工具支持

1. WordPressSpiderInMarkdown - 即开即用
   * 爬取我的博客文章，并以markdown形式返回内容。

#### 存储支持

1. SaveExpData - 即开即用
   * 排除无用文件，保存实验数据。
