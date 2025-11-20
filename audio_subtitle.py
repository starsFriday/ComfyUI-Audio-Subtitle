import torch
import numpy as np
import whisper
import os
import sys
import subprocess
import tempfile
import shutil
import torchaudio
import imageio
from datetime import timedelta

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

def generate_srt(transcription):
    """生成 SRT 内容"""
    srt_content = ""
    for i, segment in enumerate(transcription['segments']):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip()
        
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
                
                "Fontname": ("STRING", {"default": "Arial", "multiline": False}),
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
            video_np = (images.cpu().numpy() * 255).astype(np.uint8)
            imageio.mimwrite(video_input_path, video_np, fps=fps, codec='libx264', quality=8)

            if self.model is None or self.current_model_size != model_size:
                print(f"Loading Whisper model: {model_size}")
                self.model = whisper.load_model(model_size)
                self.current_model_size = model_size

            result = self.model.transcribe(audio_path, verbose=False)
            srt_content = generate_srt(result)
            
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
                f"MarginV={MarginV}"
            )
            
            print(f"Style Config: {style}")

            output_video_path = os.path.join(temp_dir, "output_burned.mp4")
            
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i", "input_visual.mp4",
                "-i", "temp_audio.wav",
                "-vf", f"subtitles='{srt_file_name}':force_style='{style}'",
                "-c:v", "libx264",
                "-preset", "fast", 
                "-crf", "18",
                "-c:a", "aac",
                "-map", "0:v",
                "-map", "1:a",
                "output_burned.mp4"
            ]

            subprocess.run(ffmpeg_cmd, cwd=temp_dir, check=True)

            if not os.path.exists(output_video_path):
                raise Exception("FFmpeg 输出文件未生成")

            reader = imageio.get_reader(output_video_path)
            output_frames = []
            for frame in reader:
                output_frames.append(frame)
            reader.close()

            output_tensor = torch.from_numpy(np.array(output_frames)).float() / 255.0
            
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
