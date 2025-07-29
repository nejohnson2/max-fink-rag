from typing import Optional, List, Any, Dict
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field
import requests

class RemoteOllamaLLM(LLM):
    """Custom LLM that connects to a remote Ollama API with headers."""

    model: str = Field(...)
    base_url: str = Field(..., description="Base URL for the Ollama API")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="Headers to include in the request")

    @property
    def _llm_type(self) -> str:
        return "remote_ollama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()["response"]

    def invoke(self, prompt: str, **kwargs: Any) -> Generation:
        output = self._call(prompt)
        return Generation(text=output)
