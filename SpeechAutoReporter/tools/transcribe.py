"""
这个代码是用于实现语音识别的，对transformer实现的语音识别库进行封装
"""
import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor
from pydub import AudioSegment
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from opencc import OpenCC
#print("通过导入语言列表来查看 Whisper 支持的所有可能语言:",TO_LANGUAGE_CODE)

models_path = 'models/openai/whisper-base'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 预处理音频函数
def preprocess_audio(audio_file_path):
    audio_data, sampling_rate = sf.read(audio_file_path) #读取音频文件
    print(audio_data.shape)
    
    # 强制单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # 转换为 int16
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)
    
    # 重采样到16kHz
    if sampling_rate != 16000:
        # 将音频数据转换为AudioSegment对象以便重采样
        # audio_data.tobytes(): 将numpy数组转换为字节序列
        # frame_rate: 原始采样率
        # sample_width: 样本位宽（每个采样点占用的字节数）
        # channels: 声道数（这里是单声道）
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=sampling_rate,
            sample_width=audio_data.dtype.itemsize,
            channels=1
        )
        audio_segment = audio_segment.set_frame_rate(16000)
        audio_data = np.array(audio_segment.get_array_of_samples())
        sampling_rate = 16000
    
    return audio_data, sampling_rate

# 使用transformer自带的模型载入方法
processor = AutoProcessor.from_pretrained(
    models_path,#模型路径
)
"""
models_path : str
    预训练模型的本地路径
torch_dtype : torch.dtype
    模型使用的数据类型,这里使用float16以降低显存占用
device_map : str
    设备映射策略,"auto"表示自动选择合适的设备
attn_implementation : str
    注意力机制的实现方式,"sdpa"表示使用Flash Attention优化的注意力计算
"""
model = WhisperForConditionalGeneration.from_pretrained(
    models_path,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
).to(device)

# 因为whisper训练时不能处理长时间音频（30s以上），因此需要进行切分然后识别,block_duration_seconds切分时长，不要超过30s最好，overlap_seconds重叠度
def stream_transcribe(audio_file_path, block_duration_seconds=30, overlap_seconds=1):
    audio_data, sampling_rate = preprocess_audio(audio_file_path)
    block_length_samples = int(block_duration_seconds * sampling_rate)
    overlap_samples = int(overlap_seconds * sampling_rate)
    step_samples = block_length_samples - overlap_samples
    total_samples = len(audio_data)
    full_transcription = ""
    
    i = 0
    block_idx = 1
    while i < total_samples:
        start = i
        end = min(i + block_length_samples, total_samples)
        block_audio = audio_data[start:end]
        sf.write(f"debug_block_{block_idx}.wav", block_audio, sampling_rate)#保存切分后的音频
        # 特征提取
        inputs = processor(
            block_audio, #输入的音频数据块
            sampling_rate=sampling_rate, # 采样率
            return_tensors="pt", #指定返回 PyTorch 张量格式"pt" 代表 PyTorch
            padding="max_length", #将输入填充到指定的最大长度
            max_length=block_length_samples, #设置输入的最大长度
            truncation=True, #允许截断超出最大长度的输入
            do_normalize=True #对输入进行归一化处理
        ).to(device, dtype=torch.float16)
        
        # 生成文本
        predicted_ids = model.generate(
            inputs.input_features, #音频特征
            max_new_tokens=400 #最大生成token个数
        )
        #解码成普通话
        current_transcript = processor.batch_decode(predicted_ids, language='mandarin', skip_special_tokens=True)[0]
        #拼接
        full_transcription += current_transcript
        
        print(f"Block {block_idx}: {current_transcript}")
        i += step_samples
        block_idx += 1
    # 使用opencc库进行繁简转换
    cc = OpenCC('t2s')  # 繁体转简体
    full_transcription = cc.convert(full_transcription)
    return full_transcription

if __name__ == '__main__':
    audio_file_path = "/root/autodl-tmp/whisper/datasets/ozj_7.mp3"
    stream_transcribe(audio_file_path, block_duration_seconds=30, overlap_seconds=1)