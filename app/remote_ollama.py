from typing import Optional, List, Any, Dict
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field
import requests
import json
from requests.exceptions import ChunkedEncodingError, ReadTimeout, ConnectionError as RequestsConnectionError
from urllib3.exceptions import ProtocolError as Urllib3ProtocolError

class RemoteOllamaLLM(LLM):
    """Custom LLM that connects to a remote Ollama API with headers."""

    model: str = Field(...)
    base_url: str = Field(..., description="Base URL for the Ollama API")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="Headers to include in the request")

    @property
    def _llm_type(self) -> str:
        return "remote_ollama"
    
    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        # Avoid double /api
        if base.endswith("/api"):
            return f"{base}/generate"
        return f"{base}/api/generate"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        url = self._endpoint()
        base_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }
        headers = {**base_headers, **(self.headers or {})}

        payload = {"model": self.model, "prompt": prompt, "stream": True}
        if stop:
            payload["stop"] = stop

        # Try streaming NDJSON; tolerate truncated streams
        try:
            with requests.post(url, json=payload, headers=headers, timeout=180, stream=True) as resp:
                resp.raise_for_status()
                ct = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" in ct:
                    snippet = (resp.text or "")[:200].replace("\n", " ")
                    raise ValueError(f"Ollama base_url is not the API. Got HTML from {url}. Body[:200]={snippet}")

                parts: List[str] = []
                try:
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if "response" in obj:
                                parts.append(obj["response"])
                            elif "message" in obj and isinstance(obj["message"], dict):
                                parts.append(obj["message"].get("content", ""))
                            if obj.get("done") is True:
                                break
                        except Exception:
                            continue
                except (ChunkedEncodingError, Urllib3ProtocolError, ReadTimeout, RequestsConnectionError):
                    # Accept partial content
                    pass

                if parts:
                    return "".join(parts)
        except (ChunkedEncodingError, Urllib3ProtocolError, ReadTimeout, RequestsConnectionError):
            # Fall through to fallback
            pass

        # Fallback: request non-stream mode, but read with stream=True to avoid buffering errors
        payload["stream"] = False
        with requests.post(url, json=payload, headers=headers, timeout=180, stream=True) as resp:
            resp.raise_for_status()
            ct = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" in ct:
                snippet = (resp.text or "")[:200].replace("\n", " ")
                raise ValueError(f"Ollama base_url is not the API. Got HTML from {url}. Body[:200]={snippet}")

            chunks: List[str] = []
            try:
                for chunk in resp.iter_content(chunk_size=4096, decode_unicode=True):
                    if chunk:
                        chunks.append(chunk)
            except (ChunkedEncodingError, Urllib3ProtocolError, ReadTimeout, RequestsConnectionError):
                # Accept partial body
                pass

            text = "".join(chunks)

        # Try JSON, then NDJSON, then error
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                if "response" in data:
                    return data["response"]
                if "message" in data and isinstance(data["message"], dict):
                    return data["message"].get("content", "")
        except Exception:
            pass

        parts: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "response" in obj:
                    parts.append(obj["response"])
                elif "message" in obj and isinstance(obj["message"], dict):
                    parts.append(obj["message"].get("content", ""))
            except Exception:
                continue
        if parts:
            return "".join(parts)

        snippet = text[:200].replace("\n", " ")
        raise ValueError(f"Ollama returned unexpected/truncated body. Content-Type={ct} Body[:200]={snippet}")
    # def _call(
    #     self,
    #     prompt: str,
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    # ) -> str:
    #     url = f"{self.base_url}/api/generate"
    #     payload = {
    #         "model": self.model,
    #         "prompt": prompt,
    #         "stream": False,
    #     }

    #     response = requests.post(url, json=payload, headers=self.headers)
    #     response.raise_for_status()
    #     return response.json()["response"]

    # # def invoke(self, prompt: str, **kwargs: Any) -> Generation:
    # #     output = self._call(prompt)
    # #     return Generation(text=output)
