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
        return {
            "required": {
                "images": ("IMAGE",), 
                "audio": ("AUDIO",), 
                "fps": ("FLOAT", {"default": 25.0, "min": 0.1, "max": 120.0, "step": 0.01}),
                "model_size": (["tiny", "base", "small", "medium", "large"], {"default": "small"}),
                
                "Fontname": ("STRING", {"default": "Arial", "multiline": False}),
                "Fontsize": ("INT", {"default": 16, "min": 5, "max": 100}),
                "PrimaryColour": ("STRING", {"default": "&H000000FF"}),  # 默认黄色 (ASS颜色代码)
                "OutlineColour": ("STRING", {"default": "&H0000FFFF"}),  # 默认半透明黑
                "BackColour": ("STRING", {"default": "&H80000000"}),
                "BorderStyle": ("INT", {"default": 3, "min": 1, "max": 4}),
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

    def process_video_subtitles(self, images, audio, fps, model_size, 
                              Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, 
                              BorderStyle, Outline, Shadow, Alignment, MarginV):
        
        temp_dir = tempfile.mkdtemp()
        current_dir = os.getcwd()
        
        try:
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

            print("正在识别音频...")
            result = self.model.transcribe(audio_path, verbose=False)
            srt_content = generate_srt(result)
            
            srt_file_name = "subtitles.srt"
            srt_path = os.path.join(temp_dir, srt_file_name)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            style = (
                f"Fontname={Fontname},"
                f"Fontsize={Fontsize},"
                f"PrimaryColour={PrimaryColour},"
                f"OutlineColour={OutlineColour},"
                f"BackColour={BackColour},"
                f"BorderStyle={BorderStyle},"
                f"Outline={Outline},"
                f"Shadow={Shadow},"
                f"Alignment={Alignment},"
                f"MarginV={MarginV}"
            )
            
            print(f"Style: {style}")

            output_video_path = os.path.join(temp_dir, "output_burned.mp4")
            
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i", "input_visual.mp4",
                "-i", "temp_audio.wav",
                "-vf", f"subtitles='{srt_file_name}':force_style='{style}'",
                "-c:v", "libx264",
                "-preset", "fast", 
                "-crf", "18",      # 保证画质
                "-c:a", "aac",
                "-map", "0:v",     # 使用第一个输入的视频流
                "-map", "1:a",     # 使用第二个输入的音频流
                "output_burned.mp4"
            ]

            print("运行 FFmpeg 压制字幕...")
            subprocess.run(ffmpeg_cmd, cwd=temp_dir, check=True)

            if not os.path.exists(output_video_path):
                raise Exception("FFmpeg 输出文件未生成")

            print("正在将视频转回序列帧...")
            reader = imageio.get_reader(output_video_path)
            output_frames = []
            for frame in reader:
                output_frames.append(frame)
            reader.close()

            # 转换回 Tensor [Batch, H, W, C] / 255.0
            output_tensor = torch.from_numpy(np.array(output_frames)).float() / 255.0
            
            print("处理完成，清理临时文件。")
            
        except Exception as e:
            print(f"处理出错: {e}")
            return (images, audio, fps)
            
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