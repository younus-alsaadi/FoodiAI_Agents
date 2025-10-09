from abc import ABC, abstractmethod

class Interface_FT_LLM(ABC):
    @abstractmethod
    def download_model_and_tokenizer(self, model_id: str):
        pass

    @abstractmethod
    def load_model_and_tokenizer(self, model_path: str, tokenizer_path: str):
        pass


    @abstractmethod
    def generate_answer(self, messages: list, max_new_tokens: int = 512, temperature: float = 0.7):
        pass





