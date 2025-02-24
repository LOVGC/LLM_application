# 步骤1：安装必要库（在终端执行）
# pip install langchain dashscope

# 步骤2：实现代码
import dashscope
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from typing import Optional, List, Dict, Any

# 自定义阿里云大模型封装
class AliyunLLM(LLM):
    api_key: str  # DashScope API密钥
    model_name: str = "qwen-max"  # 默认使用qwen-max模型

    @property
    def _llm_type(self) -> str:
        return "aliyun"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # 设置API密钥
        dashscope.api_key = self.api_key
        
        # 调用通义千问API
        response = dashscope.Generation.call(
            model=self.model_name,
            messages=[{
                'role': 'user', 
                'content': f"请用简洁的中文总结以下内容：{prompt}"
            }],
        )
        
        # 解析响应结果
        if response.status_code == 200:
            return response.output.choices[0].message['content']
        else:
            raise Exception(f"API请求失败，状态码：{response.status_code}，错误信息：{response.message}")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

# 主程序
if __name__ == "__main__":
    # 阿里云API配置（从环境变量或安全存储获取）
    api_key = "sk-1ced20ec42ab409db5dd4a6a9baa5522"
    
    # 初始化模型
    llm = AliyunLLM(api_key=api_key)
    
    # 构建提示模板
    template = """{text}"""
    prompt = PromptTemplate(input_variables=["text"], template=template)
    
    # 创建处理链
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 用户输入示例
    user_input = """
    人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。
    人工智能领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，
    理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。
    """
    
    # 执行总结
    summary = chain.run(text=user_input)
    
    # 输出结果
    print("==== 原始文本 ====")
    print(user_input)
    print("\n==== 总结结果 ====")
    print(summary)
