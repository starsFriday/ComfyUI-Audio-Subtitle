一个用于 ComfyUI 的自定义节点，利用 OpenAI Whisper 自动识别音频生成字幕，并使用 FFmpeg 将字幕**硬压制（Hardcode）**到视频帧中。
支持自定义字体、大小、颜色、描边、阴影及位置对齐，适合制作带字幕的短视频或序列帧动画。


<img width="884" height="821" alt="db712424c4b427f076ee1f1287489312" src="https://ai.static.ad2.cc/subtitle.png" />

✨ 特性

自动语音转文字：基于 OpenAI Whisper 模型（支持 multilingual）。

硬字幕压制：使用 FFmpeg 的 subtitles 滤镜，支持复杂的样式控制。

流式处理：接收图片序列和音频，输出处理后的图片序列，可直接连接至 Video Combine 节点。

高度可定制：支持调整字体、颜色、边框、阴影、位置等。

⚙️ 安装说明
1. 克隆仓库

将本仓库克隆到你的 ComfyUI custom_nodes 目录下：

cd ComfyUI/custom_nodes/
git clone https://github.com/starsFriday/ComfyUI-Audio-Subtitle.git

2. 安装依赖

本节点依赖：

openai-whisper

imageio

torchaudio

numpy

请在 ComfyUI 的 Python 环境中运行：

pip install openai-whisper imageio[ffmpeg] torchaudio numpy

3. ⚠️ 安装 FFmpeg（必须）

本节点使用系统 ffmpeg 命令进行字幕渲染，必须确保 FFmpeg 可用。

Windows

下载FFmpeg并解压。

将 bin 文件夹加入系统环境变量 Path。

验证：

ffmpeg -version

Linux / macOS
sudo apt install ffmpeg
# 或
brew install ffmpeg

🎮 使用方法

基础工作流连接

使用 Load Video (Upload) 或 VHS Load Video 加载视频。

将视频的 IMAGE 输出连接到本节点的 images。

将视频的 AUDIO 输出连接到本节点的 audio。

设置 fps（通常手动指定，如 25.0 / 30.0，可从 VHS Info 获取）。

将本节点的 frames 与 audio 输出连接至 Video Combine 节点进行合成。

📝 参数详解（Styling Guide）

本节点通过 FFmpeg 的 force_style 参数控制字幕样式。

1. 模型设置（Whisper）
model_size	特点
tiny/base	速度快，精度一般
small/medium	推荐，速度与精度平衡
large	精度最高，但占用显存大，速度慢
2. 字体与文字（Font）

Fontname：字体名称

Windows 示例：Arial, SimHei, Microsoft YaHei

⚠️ 若要显示中文，请使用中文字体（否则显示方块）

Fontsize：字号（像素）

3. 颜色设置（ASS Colors 🎨）

ASS 颜色格式：

&H[透明度][蓝][绿][红]


即：&H AABBGGRR（注意是 BGR！）

常用颜色
颜色	代码
纯白	&H00FFFFFF
纯黄	&H0000FFFF
纯红	&H000000FF
纯黑	&H00000000
4. 边框与阴影（Border & Shadow）

BorderStyle

1：普通文字 + 描边（常用）

3：不透明矩形背景框

Outline：描边宽度（像素）

Shadow：阴影深度（像素）

5. 位置与对齐（Alignment & Margin）

Alignment：类似小键盘布局

对齐值	位置
1	左下角
2	下方居中（推荐）
3	右下角
5	屏幕中心
7	左上角

MarginV：垂直边距

Alignment=2 时控制与底部距离

❓ 常见问题 FAQ
Q: 为什么中文显示为方块？

A: 字体不支持中文。请将 Fontname 修改为：

Microsoft YaHei

SimHei

或其他支持中文的字体

Q: 报错 FileNotFoundError: [WinError 2]？

A: 未安装 FFmpeg 或未配置环境变量。
请查看安装说明。

Q: 字幕颜色不正确？

A: 因为 ASS 字幕使用 BGR 不是 RGB。
例如：红色为 0000FF 而不是 FF0000。

📜 License

MIT