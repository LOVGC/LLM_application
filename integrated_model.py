import os
from typing import Optional
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

class Config:
    """配置管理类"""
    def __init__(
        self,
        dashscope_api_key: str = "sk-1ced20ec42ab409db5dd4a6a9baa5522",
        openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        text_model: str = "qwen-plus",
        tts_model: str = "cosyvoice-v1",
        tts_voice: str = "longxiaochun"
    ):
        self.dashscope_api_key = dashscope_api_key
        self.openai_base_url = openai_base_url
        self.text_model = text_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice

class TEXT_SUMMARIZER:
    """基于LangChain的摘要生成器"""
    def __init__(self, config: Config):
        dashscope.api_key = config.dashscope_api_key
        
        self.llm = ChatOpenAI(
            model=config.text_model,
            base_url=config.openai_base_url,
            api_key=config.dashscope_api_key
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的新闻分析助手。你需要用清晰简洁的书面中文总结用户提供的内容。"),
            ("human", "请总结以下文字，突出主要矛盾和关键细节：\n{text}")
        ])

        # 每一个 chain 就是一个 LLM + 它的 prompt_template
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )

    def generate_summary(self, text: str) -> str:
        """生成文本摘要"""
        return self.chain.invoke({"text": text})["text"]

class TTS:
    """语音合成客户端"""
    def __init__(self, config: Config):
        self.config = config
        self.synthesizer = SpeechSynthesizer(
            model=self.config.tts_model,
            voice=self.config.tts_voice,
        )

    def synthesize_speech(self, text: str, output_path: str = "output.mp3"):
        """合成语音并保存"""
        audio = self.synthesizer.call(text)
        
        print(f'[Metric] requestId: {self.synthesizer.get_last_request_id()}, '
              f'first package delay ms: {self.synthesizer.get_first_package_delay()}')
        
        with open(output_path, 'wb') as f:
            f.write(audio)

class SummaryPipeline:
    """完整的摘要处理流水线"""
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.summarizer = TEXT_SUMMARIZER(self.config)
        self.tts_client = TTS(self.config)

    def process(self, text: str, output_path: str = "output.mp3"):
        """完整处理流程：摘要生成 -> 语音合成"""
        summary = self.summarizer.generate_summary(text)
        print(f"【生成摘要】\n{summary}\n")
        self.tts_client.synthesize_speech(summary, output_path)
        print(f"语音已保存至：{output_path}")

if __name__ == "__main__":
    # 示例使用
    user_input = """
    俄乌战火三年未熄，大国博弈暗涌！从特朗普的“和平蓝图”到利雅得密谈，美俄在暗中交易？
    乌欧是否已从盟友沦为美国弃子？2月24日19:30，主持人雷小雪，邀请军事专家王强、
    华东师范大学俄罗斯研究中心副主任张昕，与您一同关注俄乌和平之路在何方。
    """
    
    pipeline = SummaryPipeline()
    pipeline.process(user_input)
