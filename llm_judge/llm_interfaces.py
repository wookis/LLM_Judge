import os
from typing import Optional
from .core import LLMInterface
from utils.logger import logger

# Note: To use these classes, you need to install the respective libraries:
# pip install openai anthropic

class OpenAILLM(LLMInterface):
    """OpenAI GPT 모델과의 상호작용을 위한 클래스"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI 라이브러리가 설치되지 않았습니다. 'pip install openai'를 실행해주세요.")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, prompt: str) -> str:
        logger.debug(f"{self.model_name} 연동")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content or ""  
            #return response 
        except Exception as e:
            return f"Error calling OpenAI API: {e}"

    def get_model_name(self) -> str:
        return self.model_name

class AnthropicLLM(LLMInterface):
    """Anthropic Claude 모델과의 상호작용을 위한 클래스"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Anthropic 라이브러리가 설치되지 않았습니다. 'pip install anthropic'를 실행해주세요.")
            
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
            
        self.model_name = model_name
        self.client = Anthropic(api_key=self.api_key)

    def generate_response(self, prompt: str) -> str:
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error calling Anthropic API: {e}"

    def get_model_name(self) -> str:
        return self.model_name