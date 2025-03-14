import os
from openai import OpenAI


user_input = "俄乌战火三年未熄，大国博弈暗涌！从特朗普的“和平蓝图”到利雅得密谈，美俄在暗中交易？乌欧是否已从盟友沦为美国弃子？2月24日19:30，主持人雷小雪，邀请军事专家王强、华东师范大学俄罗斯研究中心副主任张昕，与您一同关注俄乌和平之路在何方。"

prompt = f"帮我总结下这段文字：{user_input}"


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-1ced20ec42ab409db5dd4a6a9baa5522", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-plus", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}],
    )
    
response = completion.choices[0].message.content

