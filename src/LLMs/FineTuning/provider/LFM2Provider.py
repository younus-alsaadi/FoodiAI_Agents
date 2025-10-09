from __future__ import annotations
from src.LLMs.FineTuning.Interface_FT_LLM import Interface_FT_LLM
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from huggingface_hub import snapshot_download



class LFM2Provider(Interface_FT_LLM):
    def __init__(
        self,
        model_id: str = "LiquidAI/LFM2-2.6B",
        torch_dtype: Union[str, torch.dtype] = "float32",
        device_map: Optional[str] = None,
        use_flash_attention_2: bool = False,
        trust_remote_code: bool = False,
    ):
        self.model_id = model_id
        self.device_map = device_map
        self.use_flash_attention_2 = use_flash_attention_2
        self.trust_remote_code = trust_remote_code


        if isinstance(torch_dtype, str):
            _map = {
                "float32": torch.float32, "fp32": torch.float32,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
            }
            torch_dtype = _map.get(torch_dtype.lower(), torch.float32)
        self.dtype = torch_dtype

        self.model = None
        self.tokenizer = None

    def _ensure_pad_token(self):
        # Some causal LMs may not have a pad token. Safe default to eos.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def download_model_and_tokenizer(self, model_id: Optional[str] = None):
        """
        Download weights + tokenizer to local cache dirs and return those paths.
        """
        model_id = model_id or self.model_id

        model_cache_dir = snapshot_download(repo_id=model_id,
                                            allow_patterns=["*.bin", "*.safetensors", "*.json", "*.py", "config.*",
                                                            "model.*"])
        tok_cache_dir = snapshot_download(repo_id=model_id,
                                          allow_patterns=["tokenizer.*", "*.model", "vocab.*", "merges.*",
                                                          "special_tokens_map.json"])

        return model_cache_dir, tok_cache_dir


    def load_model_and_tokenizer(self, model_path: str, tokenizer_path: str):
        """
        Load model + tokenizer from local directories (or directly from model_id).
        """

        model_kwargs = {
            # use new arg name
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
          }

        # Only pass device_map if explicitly set (and you installed `accelerate`).
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        # Optional flash-attention-2, only on compatible GPUs + builds
        if self.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=self.trust_remote_code)
        self._ensure_pad_token()
        return self

        # --- convenience: load straight from hub without a separate download step

    def load_from_hub(self, model_id: Optional[str] = None):
        model_id = model_id or self.model_id
        return self.load_model_and_tokenizer(model_id, model_id)



    def generate_answer(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        repetition_penalty: float = 1.05,
        typical_p: Optional[float] = None,
        min_p: Optional[float] = None,  # Some transformers versions support min_p; harmless if ignored
        do_sample: bool = True,
        stop_on_eos: bool = True,
        return_full_text: bool = False,
    ) -> str:
        """
        messages: list like [{"role":"user","content":"hi"}, {"role":"assistant","content":"..."}]
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded. Call load_from_hub(...) or load_model_and_tokenizer(...).")

        # Build input ids from chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)

        # Build generation config
        gen_kwargs = {
            "do_sample": do_sample,
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
            "repetition_penalty": float(repetition_penalty),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if stop_on_eos and self.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if typical_p is not None:
            gen_kwargs["typical_p"] = float(typical_p)
        if min_p is not None:
            # Supported in recent transformers; ignored if not supported
            gen_kwargs["min_p"] = float(min_p)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)

        # Decode
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=not return_full_text)
        if not return_full_text:
            # Try to trim to only the assistant's final turn if needed
            # Many chat templates include the prompt too; a simple heuristic:
            prompt_len = input_ids.shape[-1]
            new_tokens = output_ids[0][prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return text.strip()



