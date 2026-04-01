#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM客户端模块
src/build_graph/run_kg_by_subject_grade.sh OpenAI API 调用
"""

import json
import os
import re
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """LLM客户端抽象基类"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成响应

        Args:
            prompt: 输入prompt
            **kwargs: 其他参数

        Returns:
            LLM的响应文本
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应，提取节点和边

        Args:
            response: LLM的响应文本

        Returns:
            包含nodes和edges的字典
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI API客户端（不再使用 response_format.json_schema，只要求模型输出JSON）"""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        初始化OpenAI客户端

        Args:
            model: 模型名称，如 "gpt-4", "gpt-3.5-turbo"
            api_key: API密钥，如果为None则从环境变量OPENAI_API_KEY读取
            base_url: API基础URL，如果为None则从环境变量OPENAI_BASE_URL读取，再为None则使用默认值
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")

        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("需要提供OPENAI_API_KEY环境变量或api_key参数")

        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")

        if base_url:
            base_url = base_url.rstrip('/')
            print(f"Using base_url: {base_url}")
            print("Note: OpenAI SDK will append /v1/chat/completions to this URL")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, temperature: float = 0.3,
                 max_tokens: int = 4000, **kwargs) -> str:
        """生成响应"""
        request_kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        request_kwargs.update(kwargs)

        try:
            response = self.client.chat.completions.create(**request_kwargs)

            if isinstance(response, str):
                if response.strip().startswith('<!') or '<html' in response.lower():
                    raise ValueError(
                        "API returned HTML instead of JSON. This usually means:\n"
                        "1. The API endpoint is incorrect\n"
                        "2. The API service is down or returning an error page\n"
                        "3. The base_url is pointing to a web page instead of API endpoint\n"
                        f"Response preview: {response[:200]}"
                    )
                return response

            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content and (content.strip().startswith('<!') or '<html' in content.lower()):
                    raise ValueError(
                        "API returned HTML content instead of JSON. Check your API endpoint.\n"
                        f"Response preview: {content[:200]}"
                    )
                return content

            result = str(response)
            if result.strip().startswith('<!') or '<html' in result.lower():
                raise ValueError(f"API returned HTML instead of JSON. Response: {result[:200]}")
            return result
        except Exception as e:
            error_msg = f"Error calling LLM API: {e}"
            if hasattr(e, 'response'):
                error_msg += f"\nResponse: {e.response}"
            if hasattr(e, 'status_code'):
                error_msg += f"\nHTTP Status: {e.status_code}"
            raise RuntimeError(error_msg) from e

    def parse_response(self, response: str) -> Dict[str, Any]:
        """解析响应"""
        json_text = response

        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            brace_match = re.search(r'\{.*\}', response, re.DOTALL)
            if brace_match:
                json_text = brace_match.group(0)

        try:
            data = json.loads(json_text)
            if "nodes" not in data:
                data["nodes"] = []
            if "edges" not in data:
                data["edges"] = []

            # 兼容不同提示词版本：练习题节点可能在 exercises 或 nodes_additional
            for extra_key in ["exercises", "nodes_additional"]:
                extra_nodes = data.get(extra_key)
                if isinstance(extra_nodes, list) and extra_nodes:
                    for ex in extra_nodes:
                        if isinstance(ex, dict):
                            data["nodes"].append(ex)

            edges = data.get("edges", [])
            valid_edges = []
            exercise_nodes_from_edges = []

            for item in edges:
                if not isinstance(item, dict):
                    valid_edges.append(item)
                    continue

                # 仅当它是“节点形态”时才从 edges 迁回 nodes；
                # 若同时包含 type/source/target，说明本质是边，不应当转成节点。
                has_edge_shape = all(k in item for k in ("type", "source", "target"))
                is_exercise_node = item.get("label") == "Exercise" and not has_edge_shape

                if is_exercise_node:
                    exercise_nodes_from_edges.append(item)
                else:
                    valid_edges.append(item)

            if exercise_nodes_from_edges:
                data["nodes"].extend(exercise_nodes_from_edges)

            data["edges"] = valid_edges

            return data
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"响应内容: {response[:500]}...")
            return {"nodes": [], "edges": []}


def create_llm_client(**kwargs) -> LLMClient:
    """创建 OpenAI 客户端"""
    return OpenAIClient(**kwargs)


if __name__ == "__main__":
    import sys

    try:
        client = create_llm_client()
        print("成功创建 openai 客户端")

        test_prompt = "请用JSON格式返回，包含nodes和edges两个字段，nodes和edges都为空数组。"
        print(f"\n测试prompt: {test_prompt}")
        response = client.generate(test_prompt)
        print(f"\n响应: {response}")

        parsed = client.parse_response(response)
        print(f"\n解析结果: {json.dumps(parsed, ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
