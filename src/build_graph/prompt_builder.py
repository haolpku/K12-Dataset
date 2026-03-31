#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt构建模块
根据section内容填充prompt模板
"""

from typing import Any


class PromptBuilder:
    """Prompt构建器"""

    def __init__(self, prompt_template_path: str):
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()

    def build_prompt(self, section: Any) -> str:
        """根据section内容构建完整prompt。"""
        return self.template.replace("{{Section_Markdown_Content}}", section.content)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python prompt_builder.py <prompt模板文件>")
        sys.exit(1)

    prompt_template_path = sys.argv[1]
    builder = PromptBuilder(prompt_template_path)

    class _S:
        content = "示例章节内容"

    prompt = builder.build_prompt(_S())
    print("生成的Prompt:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
