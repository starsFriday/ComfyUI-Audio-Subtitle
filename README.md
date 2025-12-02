# ComfyUI-Audio-Subtitle
一个用于 **ComfyUI** 的自定义节点，利用 **OpenAI Whisper** 自动识别音频生成字幕，并使用 **FFmpeg** 将字幕**硬压制（Hardcode）**到视频帧中。

支持自定义字体、大小、颜色、描边、阴影及位置对齐，适合制作带字幕的短视频或序列帧动画。

<p align="center">
  <img src="https://ai.static.ad2.cc/subtitle.png" width="600" />
</p>

---

## ✨ 特性

- 🎤 **自动语音转文字**：基于 OpenAI Whisper 模型（支持多语言）。
- 🔥 **硬字幕压制**：使用 FFmpeg `subtitles` 滤镜，支持复杂样式。
- 🚀 **流式处理**：输入图片序列 + 音频 → 输出带字幕帧，可直接送入 Video Combine。
- 🛠 **高度可定制**：支持字体、颜色、描边、阴影、位置等参数调节。

---

## ⚙️ 安装说明

### 1. 克隆仓库

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/starsFriday/ComfyUI-Audio-Subtitle.git
```

---

### 2. 安装依赖

本节点依赖：

- openai-whisper  
- imageio  
- torchaudio  
- numpy  

请在 **ComfyUI 所使用的 Python 环境**中安装：

```bash
pip install openai-whisper imageio[ffmpeg] torchaudio numpy
sudo apt-get install fonts-wqy-zenhei
```

---

### 3. ⚠️ 安装 FFmpeg（必须）

本节点通过调用系统的 `ffmpeg` 进行字幕渲染。

#### Windows

1. 下载 FFmpeg 并解压  
2. 将 `/bin` 目录加入系统环境变量 Path  
3. 验证：

```bash
ffmpeg -version
```

#### Linux / macOS

```bash
sudo apt install ffmpeg
# 或
brew install ffmpeg
```

---

## 🎮 使用方法（Workflow）

1. 使用 `Load Video (Upload)` 或 `VHS Load Video` 加载视频  
2. 将视频的 **IMAGE** 输出连接到本节点的 `images`  
3. 将视频的 **AUDIO** 输出连接到本节点的 `audio`  
4. 设置 fps（建议从 VHS Info 获取）  
5. 将本节点输出的 `frames` 与 `audio` 送入 `Video Combine` 节点合成最终视频  

---

## 📝 参数详解（Styling Guide）

本节点通过 FFmpeg 的 `force_style` 参数控制字幕样式。

---

### 1. 模型设置（Whisper）

| model_size | 特点 |
|-----------|-------|
| tiny/base | 速度快，精度一般 |
| small/medium | 推荐，速度与精度平衡 |
| large | 精度最高，但占用显存大、速度慢 |

---

### 2. 字体与文字（Font）

- **Fontname**：字体名称  
  - Windows 常用：Arial, SimHei, Microsoft YaHei（支持中文）
- **Fontsize**：字号（像素）

---

### 3. 颜色设置（ASS Colors 🎨）

ASS 颜色格式：

```
&H[透明度][蓝][绿][红]
```

即：`&H AABBGGRR`（注意：BGR 顺序）

#### 常用颜色示例

| 颜色 | ASS 代码 |
|------|----------|
| 纯白 | `&H00FFFFFF` |
| 纯黄 | `&H0000FFFF` |
| 纯红 | `&H000000FF` |
| 纯黑 | `&H00000000` |

---

### 4. 边框与阴影（Border & Shadow）

- **BorderStyle**
  - `1`：普通文字 + 描边（常用）
  - `3`：带不透明背景框
- **Outline**：描边宽度（像素）
- **Shadow**：阴影深度（像素）

---

### 5. 位置与对齐（Alignment & Margin）

Alignment 与小键盘布局类似：

| 对齐值 | 位置 |
|-------|------|
| 1 | 左下角 |
| 2 | 下方居中（推荐） |
| 3 | 右下角 |
| 5 | 中央 |
| 7 | 左上角 |

- **MarginV**：垂直边距（Alignment=2 时控制与底部距离）

---

## ❓ 常见问题 FAQ

### Q: 中文显示为方块？
A: 字体不支持中文，请设置 Fontname 为：

- Microsoft YaHei  
- SimHei  
- 任何支持中文的字体

---

### Q: 报错 `FileNotFoundError: [WinError 2]`？
A: 未安装 FFmpeg 或未加入系统 Path。

---

### Q: 字幕颜色显示不正确？
A: 因为 ASS 使用 **BGR** 排序，不是 RGB。

---

## 📜 License

MIT License

