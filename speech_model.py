# coding=utf-8

import dashscope
from dashscope.audio.tts_v2 import *

# 若没有将API Key配置到环境变量中，需将your-api-key替换为自己的API Key
dashscope.api_key = "sk-1ced20ec42ab409db5dd4a6a9baa5522"

model = "cosyvoice-v1"
voice = "longxiaochun"

synthesizer = SpeechSynthesizer(model=model, voice=voice)
audio = synthesizer.call("高丛最爱她的老公了！爱得不要不要的！")
print('[Metric] requestId: {}, first package delay ms: {}'.format(
    synthesizer.get_last_request_id(),
    synthesizer.get_first_package_delay()))

with open('output.mp3', 'wb') as f:
    f.write(audio)