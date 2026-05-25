import torch
import numpy as np
import whisper
import os
import subprocess
import tempfile
import shutil
import re
import unicodedata
import torchaudio
import imageio
from datetime import timedelta
from fractions import Fraction

LOG_PREFIX = "[AudioSubtitle]"


def _normalise_image_tensor(images):
    if images is None:
        raise ValueError("images 不能为空")

    if isinstance(images, torch.Tensor):
        frames = images.detach()
    else:
        frames = torch.as_tensor(images).detach()

    if frames.dim() == 3:
        frames = frames.unsqueeze(0)
    if frames.dim() != 4:
        raise ValueError(f"images 必须是 4 维张量，当前维度: {frames.dim()}")

    if frames.shape[1] in (3, 4) and frames.shape[-1] not in (3, 4):
        frames = frames.permute(0, 2, 3, 1)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.shape[-1] != 3:
        raise ValueError("images 必须是 [frames,height,width,3/4] 或 [frames,3/4,height,width]")
    if frames.shape[0] == 0:
        raise ValueError("images 至少需要 1 帧")

    return frames.contiguous()


def _frame_size(images):
    frames = _normalise_image_tensor(images)
    return int(frames.shape[1]), int(frames.shape[2])


def _image_tensor_to_uint8_tensor(images):
    frames = _normalise_image_tensor(images).cpu()
    if torch.is_floating_point(frames):
        frames = frames.to(torch.float32)
        if frames.numel() and float(frames.max()) > 1.0 + 1e-6:
            frames = frames / 255.0
        frames = (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    else:
        frames = frames.clamp(0, 255).to(torch.uint8)
    return frames.contiguous()


def _pad_to_even_rgb(frames):
    if frames.shape[2] % 2:
        frames = torch.cat((frames, frames[:, :, -1:, :]), dim=2)
    if frames.shape[1] % 2:
        frames = torch.cat((frames, frames[:, -1:, :, :]), dim=1)
    return frames.contiguous()


def _image_tensor_to_uint8_numpy(images):
    return _pad_to_even_rgb(_image_tensor_to_uint8_tensor(images)).numpy()


def _save_visual_video_pyav(images, fps, video_path, crf=19):
    import av

    fps_value = max(float(fps or 0), 1.0)
    output_dir = os.path.dirname(os.path.abspath(video_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frames = _pad_to_even_rgb(_image_tensor_to_uint8_tensor(images))
    frame_rate_fraction = Fraction(round(fps_value * 1000), 1000)

    with av.open(video_path, mode="w", format="mp4") as output:
        video_stream = output.add_stream("h264", rate=frame_rate_fraction)
        video_stream.width = int(frames.shape[2])
        video_stream.height = int(frames.shape[1])
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": str(int(crf))}

        for image in frames:
            frame = av.VideoFrame.from_ndarray(image.numpy(), format="rgb24")
            frame = frame.reformat(format="yuv420p")
            for packet in video_stream.encode(frame):
                output.mux(packet)

        for packet in video_stream.encode(None):
            output.mux(packet)


def _save_visual_video_fast(images, fps, video_path):
    try:
        print(f"{LOG_PREFIX} 使用独立 PyAV 写入临时视频...")
        _save_visual_video_pyav(images, fps, video_path, crf=19)
        return
    except Exception as exc:
        print(f"{LOG_PREFIX} PyAV 写入失败，回退 imageio: {exc}")

    video_np = _image_tensor_to_uint8_numpy(images)
    imageio.mimwrite(video_path, video_np, fps=fps, codec="libx264", quality=8)


def _load_video_frames_imageio(video_path):
    reader = imageio.get_reader(video_path)
    output_frames = []
    try:
        for frame in reader:
            output_frames.append(frame)
    finally:
        reader.close()

    if not output_frames:
        raise RuntimeError("输出视频未读取到任何帧")

    frames_np = np.asarray(output_frames)
    if frames_np.shape[-1] > 3:
        frames_np = frames_np[..., :3]
    if frames_np.dtype != np.uint8:
        frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)
    frames_np = np.ascontiguousarray(frames_np)
    return torch.from_numpy(frames_np).to(torch.float32).div_(255.0)


def _load_video_frames_fast(video_path):
    try:
        import decord

        with decord.bridge.use_torch():
            reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
            frame_count = len(reader)
            if frame_count <= 0:
                raise RuntimeError("decord 未读取到视频帧")
            frames = reader.get_batch(range(frame_count))

        if frames.dtype == torch.uint8:
            frames = frames.to(torch.float32).div_(255.0)
        else:
            frames = frames.to(torch.float32)
            if frames.numel() and float(frames.max()) > 1.0:
                frames.div_(255.0)
            frames.clamp_(0.0, 1.0)
        return frames.contiguous()
    except Exception as exc:
        print(f"{LOG_PREFIX} decord 快速读取失败，回退 imageio: {exc}")
        return _load_video_frames_imageio(video_path)


# ==========================================
# 颜色定义库 (RGB HEX 格式)
# ==========================================
COLOR_PALETTE = {
    # --- 灰度 / Grayscale ---
    "White": "FFFFFF",
    "LightGray": "D3D3D3",
    "Silver": "C0C0C0",
    "Gray": "808080",
    "DarkGray": "A9A9A9",
    "DimGray": "696969",
    "Black": "000000",

    # --- 红色系 / Reds ---
    "LightPink": "FFB6C1",
    "Pink": "FFC0CB",
    "HotPink": "FF69B4",
    "DeepPink": "FF1493",
    "Salmon": "FA8072",
    "Red": "FF0000",
    "Crimson": "DC143C",
    "FireBrick": "B22222",
    "DarkRed": "8B0000",
    "Maroon": "800000",

    # --- 橙色与棕色系 / Oranges & Browns ---
    "PeachPuff": "FFDAB9",
    "Bisque": "FFE4C4",
    "SandyBrown": "F4A460",
    "Orange": "FFA500",
    "DarkOrange": "FF8C00",
    "Coral": "FF7F50",
    "Tomato": "FF6347",
    "Peru": "CD853F",
    "Chocolate": "D2691E",
    "SaddleBrown": "8B4513",
    "Brown": "A52A2A",
    "DarkBrown": "5C4033",

    # --- 黄色系 / Yellows ---
    "Cream": "FFFDD0",
    "LightYellow": "FFFFE0",
    "LemonChiffon": "FFFACD",
    "PaleGoldenrod": "EEE8AA",
    "Khaki": "F0E68C",
    "Yellow": "FFFF00",
    "Gold": "FFD700",
    "Goldenrod": "DAA520",
    "DarkGoldenrod": "B8860B",

    # --- 绿色系 / Greens ---
    "PaleGreen": "98FB98",
    "LightGreen": "90EE90",
    "Lime": "00FF00",
    "LimeGreen": "32CD32",
    "YellowGreen": "9ACD32",
    "LawnGreen": "7CFC00",
    "Green": "008000",
    "DarkGreen": "006400",
    "ForestGreen": "228B22",
    "Olive": "808000",
    "OliveDrab": "6B8E23",
    "SeaGreen": "2E8B57",
    "MediumSeaGreen": "3CB371",
    "DarkSeaGreen": "8FBC8F",

    # --- 青色与天蓝系 / Cyans ---
    "LightCyan": "E0FFFF",
    "PaleTurquoise": "AFEEEE",
    "Aquamarine": "7FFFD4",
    "Turquoise": "40E0D0",
    "Cyan": "00FFFF",
    "Aqua": "00FFFF",
    "DarkTurquoise": "00CED1",
    "LightSeaGreen": "20B2AA",
    "Teal": "008080",

    # --- 蓝色系 / Blues ---
    "PowderBlue": "B0E0E6",
    "LightBlue": "ADD8E6",
    "SkyBlue": "87CEEB",
    "DeepSkyBlue": "00BFFF",
    "DodgerBlue": "1E90FF",
    "CornflowerBlue": "6495ED",
    "RoyalBlue": "4169E1",
    "Blue": "0000FF",
    "MediumBlue": "0000CD",
    "DarkBlue": "00008B",
    "Navy": "000080",
    "MidnightBlue": "191970",

    # --- 紫色系 / Purples ---
    "Lavender": "E6E6FA",
    "Thistle": "D8BFD8",
    "Plum": "DDA0DD",
    "Violet": "EE82EE",
    "Orchid": "DA70D6",
    "Magenta": "FF00FF",
    "MediumOrchid": "BA55D3",
    "BlueViolet": "8A2BE2",
    "DarkViolet": "9400D3",
    "Purple": "800080",
    "Indigo": "4B0082",
    "SlateBlue": "6A5ACD",
    "DarkSlateBlue": "483D8B",
}

# 将 RGB HEX 转为 ASS BGR HEX
def rgb_to_ass_hex(hex_rgb):
    hex_rgb = hex_rgb.lstrip('#')
    if len(hex_rgb) != 6:
        return "FFFFFF" # 错误回退到白色
    r = hex_rgb[0:2]
    g = hex_rgb[2:4]
    b = hex_rgb[4:6]
    return f"{b}{g}{r}" # 翻转为 BGR

def format_timestamp(seconds):
    """将秒数 (float) 转换为 SRT 时间戳格式 (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def _char_display_units(char):
    if char.isspace():
        return 0.35
    if unicodedata.east_asian_width(char) in {"W", "F"}:
        return 1.0
    if char in set(",.;:!?\"'，。！？；：、】【》）」』、】【…"):
        return 0.5
    return 0.6


def _tokenize_for_wrap(text):
    normalized = re.sub(r"\s+", " ", text.strip())
    tokens = []
    i = 0
    while i < len(normalized):
        char = normalized[i]
        if char == " ":
            tokens.append(char)
            i += 1
            continue
        if unicodedata.east_asian_width(char) in {"W", "F"}:
            tokens.append(char)
            i += 1
            continue
        j = i
        while j < len(normalized):
            current = normalized[j]
            if current == " " or unicodedata.east_asian_width(current) in {"W", "F"}:
                break
            j += 1
        tokens.append(normalized[i:j])
        i = j
    return tokens


def _token_display_units(token):
    return sum(_char_display_units(ch) for ch in token)


def _split_token_to_fit(token, max_units):
    if not token:
        return []

    chunks = []
    current = ""
    current_units = 0.0
    for ch in token:
        ch_units = _char_display_units(ch)
        if current and current_units + ch_units > max_units:
            chunks.append(current)
            current = ch
            current_units = ch_units
        else:
            current += ch
            current_units += ch_units
    if current:
        chunks.append(current)
    return chunks


def _normalize_tokens_for_wrap(tokens, max_units):
    normalized = []
    for token in tokens:
        token_units = _token_display_units(token)
        if token.strip() and token_units > max_units:
            normalized.extend(_split_token_to_fit(token, max_units))
        else:
            normalized.append(token)
    return normalized


def _line_units(tokens, start, end):
    return sum(_token_display_units(token) for token in tokens[start:end])


def _build_lines_from_breaks(tokens, breaks):
    lines = []
    start = 0
    for end in breaks:
        line = "".join(tokens[start:end]).strip()
        if line:
            lines.append(line)
        start = end
    return lines


def _best_wrap_for_line_count(tokens, max_units, line_count):
    n = len(tokens)
    memo = {}

    def solve(start, remaining_lines):
        key = (start, remaining_lines)
        if key in memo:
            return memo[key]

        if start >= n:
            result = (0.0, [])
            memo[key] = result
            return result

        if remaining_lines == 1:
            units = _line_units(tokens, start, n)
            if units <= max_units:
                result = ((max_units - units) ** 2, [n])
            else:
                result = None
            memo[key] = result
            return result

        best = None
        for end in range(start + 1, n + 1):
            units = _line_units(tokens, start, end)
            if units > max_units:
                break
            rest = solve(end, remaining_lines - 1)
            if rest is None:
                continue
            score = (max_units - units) ** 2 + rest[0]
            candidate = (score, [end] + rest[1])
            if best is None or candidate[0] < best[0]:
                best = candidate

        memo[key] = best
        return best

    result = solve(0, line_count)
    if result is None:
        return None
    return _build_lines_from_breaks(tokens, result[1])


def _greedy_wrap(tokens, max_units):
    lines = []
    current_line = []
    current_units = 0.0

    def flush_line():
        nonlocal current_line, current_units
        line = "".join(current_line).strip()
        if line:
            lines.append(line)
        current_line = []
        current_units = 0.0

    for token in tokens:
        token_units = _token_display_units(token)
        token_text = token.lstrip() if token.isspace() else token
        token_text_units = _token_display_units(token_text)

        if not current_line:
            if token_text:
                current_line.append(token_text)
                current_units = token_text_units
            continue

        if current_units + token_units <= max_units:
            current_line.append(token)
            current_units += token_units
            continue

        flush_line()
        if token_text:
            current_line.append(token_text)
            current_units = token_text_units

    flush_line()
    return lines


def wrap_subtitle_text(text, max_units):
    if not text:
        return text

    tokens = _normalize_tokens_for_wrap(_tokenize_for_wrap(text), max_units)
    total_units = _line_units(tokens, 0, len(tokens))
    fullwidth_ratio = _fullwidth_ratio(text)

    if fullwidth_ratio >= 0.6:
        # 对日语/CJK 更激进：长句不能因为“勉强放得下”就硬撑一行。
        single_line_limit = min(max_units * 0.48, 18)
        two_line_limit = min(max_units * 1.05, 36)
    else:
        single_line_limit = max_units * 0.88
        two_line_limit = max_units * 1.75

    if total_units <= single_line_limit:
        line_candidates = (1, 2, 3)
    elif total_units <= two_line_limit:
        line_candidates = (2, 3)
    else:
        line_candidates = (3,)

    for line_count in line_candidates:
        lines = _best_wrap_for_line_count(tokens, max_units, line_count)
        if lines is not None:
            return "\n".join(lines)

    lines = _greedy_wrap(tokens, max_units)
    return "\n".join(lines) if lines else text


def _fullwidth_ratio(text):
    visible_chars = [ch for ch in text if not ch.isspace()]
    if not visible_chars:
        return 0.0
    fullwidth_count = sum(1 for ch in visible_chars if unicodedata.east_asian_width(ch) in {"W", "F"})
    return fullwidth_count / len(visible_chars)


def generate_srt(transcription, max_units=None):
    """生成 SRT 内容，并按画面宽度预换行。"""
    srt_content = ""
    for i, segment in enumerate(transcription['segments']):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip()
        if max_units is not None:
            effective_max_units = max_units
            if _fullwidth_ratio(text) >= 0.6:
                # 日语/CJK 进一步压窄单行容量，避免视觉上仍然过长。
                effective_max_units = max(min(max_units * 0.62, 18), 6)
            text = wrap_subtitle_text(text, effective_max_units)

        srt_content += f"{i + 1}\n"
        srt_content += f"{start} --> {end}\n"
        srt_content += f"{text}\n\n"
    return srt_content


class AudioSubtitle:
    def __init__(self):
        self.model = None
        self.current_model_size = None

    @classmethod
    def INPUT_TYPES(s):
        color_list = sorted(list(COLOR_PALETTE.keys()))
        
        return {
            "required": {
                "images": ("IMAGE",), 
                "audio": ("AUDIO",), 
                "fps": ("FLOAT", {"default": 25.0, "min": 0.1, "max": 120.0, "step": 0.01}),
                "model_size": (["tiny", "base", "small", "medium", "large"], {"default": "small"}),
                
                "Fontname": (["Arial", "WenQuanYi Zen Hei"], {"default": "Arial"}),    # apt-get install fonts-wqy-zenhei
                "Fontsize": ("INT", {"default": 10, "min": 5, "max": 100}),
                
                # --- 颜色选择 ---
                "PrimaryColour": (color_list, {"default": "Yellow"}),     # 主体文字颜色
                "OutlineColour": (color_list, {"default": "Black"}),      # 描边颜色
                "BackColour": (color_list, {"default": "Black"}),         # 背景块颜色
                
                # --- 透明度控制 (0=不透明, 255=全透明) ---
                "OutlineAlpha": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "BackAlpha": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}), # 默认半透明背景
                
                "BorderStyle": ([1, 3], {"default": 3}), # 3=不透明背景框, 1=普通描边
                "Outline": ("INT", {"default": 1, "min": 0, "max": 10}),
                "Shadow": ("INT", {"default": 0, "min": 0, "max": 10}),
                "Alignment": ("INT", {"default": 2, "min": 1, "max": 9}), # 2 = 底部居中
                "MarginV": ("INT", {"default": 25, "min": 0, "max": 500}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("frames", "audio", "fps")
    FUNCTION = "process_video_subtitles"
    CATEGORY = "Custom/Audio Subtitles"

    def get_full_ass_color(self, color_name, alpha_int):
        """组合 Alpha 和 BGR 颜色代码"""
        # 1. 获取 RGB Hex
        rgb_hex = COLOR_PALETTE.get(color_name, "FFFFFF")
        # 2. 转为 BGR Hex
        bgr_hex = rgb_to_ass_hex(rgb_hex)
        # 3. 处理 Alpha (转为2位16进制)
        alpha_hex = f"{alpha_int:02X}"
        # 4. 组合: &H + Alpha + BGR
        return f"&H{alpha_hex}{bgr_hex}"

    def process_video_subtitles(self, images, audio, fps, model_size, 
                              Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, 
                              OutlineAlpha, BackAlpha,
                              BorderStyle, Outline, Shadow, Alignment, MarginV):
        
        temp_dir = tempfile.mkdtemp()
        current_dir = os.getcwd()
        
        try:
            # 主文字通常完全不透明 (Alpha 0)
            primary_code = self.get_full_ass_color(PrimaryColour, 0)
            # 描边颜色
            outline_code = self.get_full_ass_color(OutlineColour, OutlineAlpha)
            # 背景颜色
            back_code = self.get_full_ass_color(BackColour, BackAlpha)

            waveform = audio['waveform']
            sample_rate = audio['sample_rate']
            if waveform.dim() == 3: waveform = waveform.squeeze(0)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            audio_path = os.path.join(temp_dir, "temp_audio.wav")
            torchaudio.save(audio_path, waveform, sample_rate)

            video_input_path = os.path.join(temp_dir, "input_visual.mp4")
            _, video_width = _frame_size(images)
            margin_h = max(int(video_width * 0.02), Fontsize * 2)
            subtitle_width = max(video_width - margin_h * 2, int(video_width * 0.5))
            max_text_units = max(subtitle_width / max(Fontsize, 1), 6)
            _save_visual_video_fast(images, fps, video_input_path)

            if self.model is None or self.current_model_size != model_size:
                print(f"Loading Whisper model: {model_size}")
                self.model = whisper.load_model(model_size)
                self.current_model_size = model_size

            result = self.model.transcribe(audio_path, verbose=False)
            srt_content = generate_srt(result, max_units=max_text_units)
            
            srt_file_name = "subtitles.srt"
            srt_path = os.path.join(temp_dir, srt_file_name)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            style = (
                f"Fontname={Fontname},"
                f"Fontsize={Fontsize},"
                f"PrimaryColour={primary_code},"
                f"OutlineColour={outline_code},"
                f"BackColour={back_code},"
                f"BorderStyle={BorderStyle},"
                f"Outline={Outline},"
                f"Shadow={Shadow},"
                f"Alignment={Alignment},"
                f"MarginL={margin_h},"
                f"MarginR={margin_h},"
                f"MarginV={MarginV},"
                f"WrapStyle=0"
            )
            
            print(f"Style Config: {style}")

            output_video_path = os.path.join(temp_dir, "output_burned.mp4")
            
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-i", "input_visual.mp4",
                "-i", "temp_audio.wav",
                "-vf", f"subtitles='{srt_file_name}':force_style='{style}'",
                "-c:v", "libx264",
                "-preset", "veryfast", 
                "-crf", "18",
                "-threads", "0",
                "-c:a", "aac",
                "-map", "0:v",
                "-map", "1:a",
                "output_burned.mp4"
            ]

            subprocess.run(ffmpeg_cmd, cwd=temp_dir, check=True)

            if not os.path.exists(output_video_path):
                raise Exception("FFmpeg 输出文件未生成")

            output_tensor = _load_video_frames_fast(output_video_path)
            
            print("处理完成，清理临时文件。")
            
        except Exception as e:
            print(f"处理出错: {e}")
            raise Exception(f"处理出错: {e}")
            # return (images, audio, fps)
            
        finally:
            os.chdir(current_dir)
            shutil.rmtree(temp_dir, ignore_errors=True)

        return (output_tensor, audio, fps)

NODE_CLASS_MAPPINGS = {
    "AudioSubtitle": AudioSubtitle
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSubtitle": "📺 Audio Subtitles"
}
