"""Shared LLM client helpers used by pipeline modules."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("请先安装 openai：pip install openai") from exc

        if not api_key.strip():
            raise ValueError("missing llm api_key")

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/") if base_url else None)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        **kwargs: Any,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        content = response.choices[0].message.content if response.choices else ""
        if not isinstance(content, str):
            raise ValueError("LLM returned empty content")
        if content.strip().startswith("<!") or "<html" in content.lower():
            raise ValueError("LLM API returned HTML instead of model output; please check base_url")
        return content

    def parse_response(self, response: str) -> Dict[str, Any]:
        json_text = response
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if fenced:
            json_text = fenced.group(1)
        else:
            brace = re.search(r"\{.*\}", response, re.DOTALL)
            if brace:
                json_text = brace.group(0)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return {"nodes": [], "edges": []}

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(edges, list):
            edges = []

        for extra_key in ("exercises", "nodes_additional"):
            extra_nodes = data.get(extra_key)
            if isinstance(extra_nodes, list):
                nodes.extend(item for item in extra_nodes if isinstance(item, dict))

        normalized_edges = []
        recovered_nodes = []
        for item in edges:
            if not isinstance(item, dict):
                continue
            has_edge_shape = all(key in item for key in ("source", "target", "type"))
            if item.get("label") == "Exercise" and not has_edge_shape:
                recovered_nodes.append(item)
            else:
                normalized_edges.append(item)

        nodes.extend(recovered_nodes)
        return {"nodes": nodes, "edges": normalized_edges}


def create_llm_client(
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
) -> LLMClient:
    if provider != "openai":
        raise ValueError(f"unsupported llm provider: {provider}")
    return OpenAIClient(model=model, api_key=api_key, base_url=base_url)
