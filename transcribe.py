import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor
from pydub import AudioSegment
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from opencc import OpenCC
#print("通过导入语言列表来查看 Whisper 支持的所有可能语言:",TO_LANGUAGE_CODE)

models_path = '/root/autodl-tmp/models/openai/whisper-large-v3-turbo'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 预处理音频函数
def preprocess_audio(audio_file_path):
    audio_data, sampling_rate = sf.read(audio_file_path)
    print(audio_data.shape)
    
    # 强制单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # 转换为 int16
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)
    
    # 重采样到16kHz
    if sampling_rate != 16000:
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

processor = AutoProcessor.from_pretrained(
    models_path,
)
model = WhisperForConditionalGeneration.from_pretrained(
    models_path,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
).to(device)

# 流式转录函数
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
        sf.write(f"debug_block_{block_idx}.wav", block_audio, sampling_rate)
        # 特征提取
        inputs = processor(
            block_audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=block_length_samples,
            truncation=True,
            do_normalize=True
        ).to(device, dtype=torch.float16)
        
        # 生成文本
        predicted_ids = model.generate(
            inputs.input_features,
            max_new_tokens=400
        )
        current_transcript = processor.batch_decode(predicted_ids, language='mandarin', skip_special_tokens=True)[0]
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