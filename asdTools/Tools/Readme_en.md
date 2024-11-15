[README (中文)](README.md)

[README (English)](README_EN.md)

# Tool Support

[toc]

#### Introduction

* Includes the following:
  1. Out-of-the-box tools.
  2. Tools designed for shortcuts.
  3. External call examples.
* Logging:
  1. Supports logging, and all outputs will be saved by default to ./Logs under the calling path.
  2. Add the parameter `log_dir="./OtherLogDir"` during initialization to modify the save folder.
  3. Add the parameter `log_file="otherLogFile.txt"` during initialization to modify the saved log name.
  4. Example: `GPUMonitor(log_dir="/root/logs", log_file="GPUMonitor.log")`

#### AI Tool Support

1. GPUMonitor - Call example
   * Get the ID of available GPUs.
2. SDStylesBeautify - Shortcut
   - Beautify the styles.csv of the Stable Diffusion Web UI (including sorting, adding category headers, removing empty lines).

#### API Tool Support

1. Chat_LLMs - Out-of-the-box
   * Used for chatting with LLMs, with an example of communicating with the locally deployed ChatGLM API.

#### Datasets Tool Support

1. RandomDatasetSplitter_SemanticSegmentation - Out-of-the-box
   * Randomly split two folders with the same internal file names, such as RGB images and GT (Ground Truth) images in semantic segmentation.

#### Image Tool Support

1. ImageResizer_256x144 - Shortcut
   * Resize the resolution of the image to 256x144 and provides a Windows right-click menu shortcut.
2. ConvertGT2MMSeg - Out-of-the-box
   * Convert GT images to the image format required by MMSegmentation. Support input RGB images and grayscale images.
3. RGB2Gray3Channel - Out-of-the-box
   * Convert RGB images into gray images with 3 channels.
4. ColorImgWithGT - Out-of-the-box
   * Colorize the image with the colorful segmentation GT.
5. ColorGT - Out-of-the-box
   1. Colorize GT images. Support input and output RGB images and grayscale images.
6. VisualizeHeatmapOfReID - Call example
   * Visualize heatmap, only test in ReID.
7. VisualizeRankOfReID - Call example
   * Visualize ReID rank, very fast by markdown table
8. VisualizeSegGTDistribution - Call example
   1. Visualize GT distribution of semantic segmentation.

#### Spider Tool Support

1. WordPressSpiderInMarkdown - Out-of-the-box
   1. Crawl post of my blog, and output in markdown.

#### Storage Tool Support

1. SaveExpData - Out-of-the-box
   1. Exclude useless files and save experiments data.
