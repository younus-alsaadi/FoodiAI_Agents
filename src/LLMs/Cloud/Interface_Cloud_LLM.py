from abc import ABC, abstractmethod

class Interface_Cloud_LLM(ABC):
    @abstractmethod
    def set_generation_model(self, model_id: str):
        pass

    @abstractmethod
    def set_model_task(self,model_task:str):
        pass

    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                      temperature: float = None):
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str):
        pass