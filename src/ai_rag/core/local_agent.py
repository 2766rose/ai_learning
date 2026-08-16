# src/ai_rag/core/local_agent.py
import time
import json
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class LocalAgentError(Exception):
    """LocalAgent 专用结构化异常"""
    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{error_code}] {message}")

class LocalAgent:
    # Ollama 默认端点
    BASE_URL = "http://localhost:11434"
    # 强制保持模型常驻内存，避免冷启动
    KEEP_ALIVE = "-1" 
    # 企业级超时阈值 (秒)
    TIMEOUT_SEC = 30 

    def __init__(self, model_name: str = "qwen3:8b"):
        self.model_name = model_name
        self._verify_service()

    def _verify_service(self):
        """启动时验证 Ollama 服务可用性及 Keep-Alive 配置"""
        try:
            resp = requests.get(f"{self.BASE_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model_name not in models:
                raise LocalAgentError("MODEL_NOT_FOUND", f"模型 {self.model_name} 未在 Ollama 中加载")
            logger.info(f"✅ LocalAgent 初始化成功，模型: {self.model_name}, Keep-Alive: {self.KEEP_ALIVE}")
        except requests.exceptions.ConnectionError:
            raise LocalAgentError("SERVICE_UNREACHABLE", "无法连接到 Ollama 服务，请检查 11434 端口")
        except Exception as e:
            raise LocalAgentError("INIT_FAILED", str(e))

    def chat(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> Dict[str, Any]:
        """
        统一执行入口，返回标准化字典
        """
        start_time = time.time()
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,  # 企业级应用必须使用流式以降低 TTFT
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "keep_alive": self.KEEP_ALIVE
            }
        }

        full_response = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}

        try:
            with requests.post(
                f"{self.BASE_URL}/api/generate", 
                json=payload, 
                stream=True, 
                timeout=self.TIMEOUT_SEC
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line: continue
                    chunk = json.loads(line.decode('utf-8'))
                    
                    # 拼接流式内容
                    if "response" in chunk:
                        full_response.append(chunk["response"])
                    
                    # Ollama 在最后一个 chunk 返回 eval_count 等统计信息
                    if chunk.get("done", False):
                        usage["prompt_tokens"] = chunk.get("prompt_eval_count", 0)
                        usage["completion_tokens"] = chunk.get("eval_count", 0)

            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "response": "".join(full_response),
                "usage": usage,
                "latency_ms": latency_ms,
                "status": "success"
            }

        except requests.exceptions.Timeout:
            logger.error(f"⏱️ LocalAgent 超时 ({self.TIMEOUT_SEC}s)，触发降级条件")
            raise LocalAgentError("TIMEOUT", f"本地推理超过 {self.TIMEOUT_SEC}s 限制")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ LocalAgent 网络/服务异常: {e}")
            raise LocalAgentError("NETWORK_ERROR", str(e))
        except Exception as e:
            logger.error(f"💥 LocalAgent 未知异常: {e}")
            raise LocalAgentError("UNKNOWN_ERROR", str(e))