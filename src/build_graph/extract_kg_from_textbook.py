#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从教材提取知识图谱的主pipeline
"""

import json
import os
import re
import sys
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# 确保本脚本无论从哪里启动都能导入同目录模块
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# 确保使用UTF-8编码
if sys.platform != 'win32':
    import locale
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # 尝试设置locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            pass

# 设置标准输出编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from dataclasses import dataclass, field
from prompt_builder import PromptBuilder
from llm_client import create_llm_client, LLMClient


@dataclass
class Section:
    chapter_num: str
    chapter_title: str
    section_num: str
    section_title: str
    content: str
    start_line: int = 0
    end_line: int = 0
    source_meta: Dict[str, Any] = field(default_factory=dict)
    file_name: str = ""


class KnowledgeGraphExtractor:
    """知识图谱提取器"""
    
    def __init__(self, llm_client: LLMClient, prompt_builder: PromptBuilder,
                 grade: str, publisher: str, subject: str,
                 book_prefix: Optional[str] = None):
        """
        初始化提取器
        
        Args:
            llm_client: LLM客户端
            prompt_builder: Prompt构建器
            grade: 年级
            publisher: 出版社
            subject: 学科
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.grade = grade
        self.publisher = publisher
        self.subject = subject
        # 教材编号前缀（如 math_7a_rjb），用于生成全局唯一的节点ID
        self.book_prefix = book_prefix
        if not self.book_prefix or not str(self.book_prefix).strip():
            raise ValueError("book_prefix 不能为空")
        
        # 存储所有提取的节点和边
        self.all_nodes: List[Dict[str, Any]] = []
        self.all_edges: List[Dict[str, Any]] = []
        
        # 节点和边的ID生成器
        self.node_id_counter = 0
        self.edge_id_counter = 0
        self.node_id_map: Dict[str, str] = {}  # 用于去重
        
        # 日志文件路径（在提取流程中根据 output_dir 设置）
        self.log_path: Optional[str] = None
    
    def _generate_node_id(self, node_type: str, name: str) -> str:
        """
        生成节点ID
        
        Args:
            node_type: 节点类型（Concept, Skill, Experiment, Exercise）
            name: 节点名称
            
        Returns:
            节点ID
        """
        key = f"{node_type}:{name}"
        if key not in self.node_id_map:
            self.node_id_counter += 1
            self.node_id_map[key] = f"{node_type}_{self.node_id_counter}"
        return self.node_id_map[key]
    
    def _safe_print(self, message: str):
        """安全打印，处理编码问题，并同步写入日志文件（如果已配置）"""
        # 终端输出
        try:
            print(message)
        except UnicodeEncodeError:
            # 如果编码失败，尝试使用ASCII替代
            print(message.encode('ascii', 'replace').decode('ascii'))
        
        # 追加到日志文件
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as lf:
                    lf.write(message + "\n")
            except Exception:
                # 日志写入失败不影响主流程
                pass

    def _build_source_metadata(self, section_record: Dict[str, Any]) -> Dict[str, str]:
        """根据 sections_index.json 的 *_num/*_title 键构建 source 元数据。"""
        source_meta: Dict[str, str] = {}
        for key, value in section_record.items():
            if not key.endswith("_num"):
                continue
            if value is None or str(value).strip() == "":
                continue
            level = key[:-4]  # chapter_num -> chapter
            source_meta[f"source_{level}"] = str(value)
            title_key = f"{level}_title"
            title_val = section_record.get(title_key)
            if title_val is not None and str(title_val).strip() != "":
                source_meta[f"source_{level}_title"] = str(title_val)
        return source_meta
    
    def _process_section(self, section: Section, section_index: int, 
                        total_sections: int) -> Dict[str, Any]:
        """
        处理单个section，提取知识图谱
        
        Args:
            section: Section对象
            section_index: section索引（从0开始）
            total_sections: 总section数
            
        Returns:
            包含nodes、edges、section_info和raw_response的字典（raw_response 仅用于写入 raw.txt）
        """
        section_info = f"[{section_index + 1}/{total_sections}] Processing: Chapter {section.chapter_num} {section.chapter_title}"
        self._safe_print(f"\n{section_info}")
        source_fields = dict(section.source_meta or {})
        
        # 构建prompt
        prompt = self.prompt_builder.build_prompt(section)
        
        # 调用LLM
        try:
            self._safe_print("  Calling LLM...")
            response = self.llm_client.generate(prompt)
            
            # 保存原始响应用于调试
            if len(response) > 0:
                self._safe_print(f"  LLM response received (length: {len(response)} chars)")
            else:
                self._safe_print("  WARNING: Empty LLM response!")
            
            # 解析响应
            self._safe_print("  Parsing response...")
            kg_data = self.llm_client.parse_response(response)
            
            nodes = kg_data.get("nodes", [])
            edges = kg_data.get("edges", [])
            
            self._safe_print(f"  Extracted {len(nodes)} nodes, {len(edges)} edges")
            
            # 如果解析失败，打印部分响应用于调试
            if len(nodes) == 0 and len(edges) == 0:
                self._safe_print("  WARNING: No nodes or edges extracted!")
                self._safe_print(f"  Response preview (first 500 chars): {response[:500]}")
            
            # 为节点生成ID并添加元数据
            processed_nodes = []
            # 使用节点的编号 id 作为主索引键（与你在 prompt 中约定的 cpt_*/skl_*/exp_*/exe_* 一致）
            node_id_key_to_internal_id = {}
            
            for node in nodes:
                # 节点类型从 label 字段读取（Concept/Skill/Experiment/Exercise）
                node_type = node.get("label")
                
                # 如果 label 缺失，根据 id 前缀推断节点类型（兜底逻辑）
                if not node_type:
                    node_id = node.get("id", "")
                    if node_id.startswith("cpt_"):
                        node_type = "Concept"
                    elif node_id.startswith("skl_"):
                        node_type = "Skill"
                    elif node_id.startswith("exp_"):
                        node_type = "Experiment"
                    elif node_id.startswith("exe_"):
                        node_type = "Exercise"
                    else:
                        # 最后兜底：默认为 Concept
                        node_type = "Concept"
                
                # 约定：LLM 必须输出 id（cpt_*/skl_*/exp_*/exe_*），否则退回用 name/stem 兜底
                node_key = node.get("id") or node.get("name") or node.get("stem") or f"node_{len(processed_nodes)}"
                
                # 生成内部唯一ID（用于 Neo4j 等），与 edge 使用的编号解耦
                # 按用户约定规则构造：
                #  - Concept:   {book_prefix}_cpt{n}
                #  - Skill:     {book_prefix}_skl{n}
                #  - Experiment:{book_prefix}_exp{n}
                #  - Exercise:  {book_prefix}_ch{chapter}_t{n}
                # 其中 n 为当前 chapter 内该类型节点的序号
                # 为简单起见，这里从 node_key 中提取尾部数字作为 n，如果没有数字则用遍历顺序
                m = re.search(r'(\d+)$', str(node_key))
                if m:
                    idx = int(m.group(1))
                else:
                    idx = len([n for n in processed_nodes if n["label"] == node_type]) + 1
                
                chapter_code = f"ch{section.chapter_num}"
                
                prefix = self.book_prefix
                if node_type == "Concept":
                    internal_id = f"{prefix}_cpt{idx}"
                elif node_type == "Skill":
                    internal_id = f"{prefix}_skl{idx}"
                elif node_type == "Experiment":
                    internal_id = f"{prefix}_exp{idx}"
                elif node_type == "Exercise":
                    internal_id = f"{prefix}_{chapter_code}_t{idx}"
                else:
                    # 兜底
                    internal_id = self._generate_node_id(node_type, node_key)
                node_id_key_to_internal_id[node_key] = internal_id
                
                # 构建节点数据
                processed_node = {
                    "id": internal_id,
                    "label": node_type,
                    "properties": {
                        **node,  # 包含所有原始属性
                        "name": node.get("name") or node.get("stem") or node_key,
                        **source_fields
                    }
                }
                processed_nodes.append(processed_node)
            
            # 处理边
            processed_edges = []
            
            def _normalize_ref(ref: Any) -> Optional[str]:
                """将 edge 中的 source/target 引用规范化为节点编号（与节点 id 对齐的字符串）"""
                if ref is None:
                    return None
                if isinstance(ref, str):
                    return ref
                # 如果是dict，优先取 id，其次 name/stem/label
                if isinstance(ref, dict):
                    return (
                        ref.get("id")
                        or ref.get("name")
                        or ref.get("stem")
                        or ref.get("label")
                    )
                # 如果是列表，取第一个元素递归处理
                if isinstance(ref, list) and ref:
                    return _normalize_ref(ref[0])
                # 其他类型，退化为字符串
                return str(ref)
            
            for edge in edges:
                raw_source = edge.get("source")
                raw_target = edge.get("target")
                source_name = _normalize_ref(raw_source)
                target_name = _normalize_ref(raw_target)
                edge_type = edge.get("type", "relates_to")
                
                # 查找内部节点ID（基于规范化后的编号 key）
                source_id = node_id_key_to_internal_id.get(source_name)
                target_id = node_id_key_to_internal_id.get(target_name)
                
                if source_id and target_id:
                    processed_edge = {
                        "source": source_id,
                        "target": target_id,
                        "type": edge_type,
                        "properties": {
                            **{k: v for k, v in edge.items() 
                               if k not in ["source", "target", "type"]},
                            **source_fields
                        }
                    }
                    processed_edges.append(processed_edge)
                else:
                    self._safe_print(f"  WARNING: Cannot find nodes for edge: {raw_source} -> {raw_target} (normalized: {source_name} -> {target_name})")
            
            return {
                "nodes": processed_nodes,
                "edges": processed_edges,
                "section_info": {
                    "chapter": section.chapter_num,
                    "chapter_title": section.chapter_title,
                    "section": section.section_num,
                    "section_title": section.section_title
                },
                "raw_response": response
            }
            
        except Exception as e:
            self._safe_print(f"  ERROR: Failed to process section: {e}")
            import traceback
            traceback.print_exc()
            # 出错时也返回原始响应（如果有），方便排查
            raw_resp = locals().get("response", "")
            return {"nodes": [], "edges": [], "section_info": None, "raw_response": raw_resp}
    
    def extract_from_sections_dir(self, sections_dir: str, 
                                  start_chapter: Optional[int] = None,
                                  end_chapter: Optional[int] = None,
                                  chapters: Optional[List[int]] = None,
                                  save_intermediate: bool = True,
                                  output_dir: str = "output") -> Dict[str, Any]:
        """
        从已分割的章节目录提取知识图谱
        
        Args:
            sections_dir: 章节文件目录（包含 ch1.md, ch2.md 等和 sections_index.json）
            start_chapter: 起始章节编号（从1开始），如果为None则从第一个开始
            end_chapter: 结束章节编号（包含），如果为None则处理到最后
            chapters: 指定要处理的章节编号列表（如[2,4,6]），优先级高于start_chapter和end_chapter
            save_intermediate: 是否保存中间结果
            output_dir: 输出目录
            
        Returns:
            包含所有节点和边的字典
        """
        # 创建输出目录
        if save_intermediate:
            os.makedirs(output_dir, exist_ok=True)
        
        # 设置并初始化日志文件路径（每次运行覆盖旧日志）
        self.log_path = os.path.join(output_dir, "run.log")
        try:
            with open(self.log_path, "w", encoding="utf-8") as lf:
                lf.write(f"Knowledge graph extraction log\nSections directory: {sections_dir}\n\n")
        except Exception:
            # 如果无法创建日志文件，仅在终端输出
            self.log_path = None
        
        # 读取索引文件
        index_path = os.path.join(sections_dir, "sections_index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        sections_list = index_data.get("sections", [])
        # 去重：按 (chapter_num, section_num, filename) 去重，避免重复处理
        # 注意：同一个 chapter 可能有多个 section 文件（如 ch5_s1.md, ch5_s2.md）
        seen_items = set()
        unique_chapters = []
        for ch in sections_list:
            ch_num = ch.get("chapter_num", "")
            section_num = ch.get("section_num", "")
            filename = ch.get("file", "")
            # 使用 (chapter_num, section_num, filename) 作为唯一键
            key = (ch_num, section_num, filename)
            if key not in seen_items:
                seen_items.add(key)
                unique_chapters.append(ch)
        
        # 按章节编号和 section 编号排序
        def sort_key(x):
            try:
                ch_num = int(x.get("chapter_num", 0))
            except (ValueError, TypeError):
                ch_num = 0
            section_num = x.get("section_num", "")
            try:
                sec_num = int(section_num) if section_num else 0
            except (ValueError, TypeError):
                sec_num = 0
            return (ch_num, sec_num)
        
        unique_chapters.sort(key=sort_key)
        
        self._safe_print(f"Found {len(unique_chapters)} chapters in {sections_dir}")
        
        # 过滤章节范围
        # 如果指定了chapters列表，优先使用它；否则使用start_chapter和end_chapter
        if chapters is not None:
            # 将chapters列表转换为字符串集合以便比较
            chapters_set = {str(ch) for ch in chapters}
            self._safe_print(f"指定处理章节: {chapters}")
            unique_chapters = [ch for ch in unique_chapters if ch.get("chapter_num", "") in chapters_set]
            self._safe_print(f"过滤后剩余 {len(unique_chapters)} 个章节")
        else:
            # 使用start_chapter和end_chapter过滤
            if start_chapter is not None:
                unique_chapters = [ch for ch in unique_chapters if int(ch.get("chapter_num", 0)) >= start_chapter]
            if end_chapter is not None:
                unique_chapters = [ch for ch in unique_chapters if int(ch.get("chapter_num", 0)) <= end_chapter]
        
        self._safe_print(f"Processing {len(unique_chapters)} chapters")
        
        # 处理每个章节
        all_results = []
        for i, ch_info in enumerate(unique_chapters):
            chapter_num = ch_info.get("chapter_num")
            chapter_title = ch_info.get("chapter_title", "")
            filename = ch_info.get("file")
            
            if not filename:
                self._safe_print(f"  WARNING: Skipping chapter {chapter_num} (no filename)")
                continue
            
            # 读取章节文件内容
            chapter_path = os.path.join(sections_dir, filename)
            if not os.path.exists(chapter_path):
                self._safe_print(f"  WARNING: Chapter file not found: {chapter_path}")
                continue
            
            with open(chapter_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 移除文件开头的标题行（如果存在），并尝试提取 section 信息
            # 格式可能是: "# 第1章 准备课\n\n" 或类似
            lines = content.split('\n')
            content_start = 0
            section_num = ch_info.get("section_num")
            section_title = ch_info.get("section_title")
            
            # 尝试从文件名中提取 section 编号（如 ch5_s1.md -> s1）
            filename_match = re.search(r'_s(\d+)\.md$', filename)
            if filename_match:
                section_num = filename_match.group(1)
            
            # 跳过 chapter 标题行
            if lines and lines[0].startswith('# 第') and '章' in lines[0]:
                content_start = 1
                while content_start < len(lines) and not lines[content_start].strip():
                    content_start += 1
                
                # 尝试从内容中提取 section 标题（查找第一个非 chapter 标题的 # 标题）
                for i in range(content_start, min(content_start + 20, len(lines))):
                    line = lines[i].strip()
                    if line.startswith('#'):
                        # 检查是否是 chapter 标题（包含"第X章"）
                        if '第' in line and '章' in line:
                            continue
                        # 提取 section 标题
                        section_title = line.lstrip('#').strip()
                        # 如果还没有 section_num，尝试从标题中提取（如 "6和7" -> "1"）
                        if not section_num and section_title:
                            # 这里可以根据实际需要调整提取逻辑
                            # 暂时使用文件名中的编号
                            pass
                        break
            
            content = '\n'.join(lines[content_start:])
            
            # 创建 Section 对象
            section = Section(
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                section_num=section_num,
                section_title=section_title,
                content=content,
                start_line=0,
                end_line=len(content.split('\n')),
                source_meta=self._build_source_metadata(ch_info),
                file_name=filename or ""
            )
            
            # 处理章节
            result = self._process_section(section, i, len(unique_chapters))
            
            # 先拿到原始响应（只写 raw.txt，不写入 json）
            raw_response = result.pop("raw_response", "")
            base_name = os.path.splitext(filename)[0] if filename else f"ch{chapter_num}"

            # 保存中间结果（不包含 raw_response 字段）
            if save_intermediate and result.get("section_info"):
                json_path = os.path.join(output_dir, base_name + ".json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                self._safe_print(f"  Saved intermediate result: {os.path.basename(json_path)}")

            # 无论解析成功与否，都保存一份原始 LLM 输出（可能为空）
            os.makedirs(output_dir, exist_ok=True)
            raw_path = os.path.join(output_dir, base_name + "_llm_raw.txt")
            with open(raw_path, 'w', encoding='utf-8') as f_raw:
                f_raw.write(raw_response if isinstance(raw_response, str) else str(raw_response))
            
            # 累积节点和边
            self.all_nodes.extend(result["nodes"])
            self.all_edges.extend(result["edges"])
            all_results.append(result)
        
        # 构建最终结果
        final_result = {
            "metadata": {
                "sections_dir": sections_dir,
                "grade": self.grade,
                "publisher": self.publisher,
                "subject": self.subject,
                "total_chapters_processed": len(all_results),
                "total_nodes": len(self.all_nodes),
                "total_edges": len(self.all_edges)
            },
            "nodes": self.all_nodes,
            "edges": self.all_edges
        }
        
        # 保存最终结果
        output_file = os.path.join(output_dir, "knowledge_graph.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        self._safe_print(f"\nKnowledge graph extraction completed!")
        self._safe_print(f"  Total nodes: {len(self.all_nodes)}")
        self._safe_print(f"  Total edges: {len(self.all_edges)}")
        self._safe_print(f"  Results saved to: {output_file}")
        
        return final_result


def main():
    parser = argparse.ArgumentParser(
        description="从 out_sections 目录提取知识图谱",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
:
  python extract_kg_from_textbook.py \
    --sections-dir out_sections \
    --prompt src/build_graph/prompt_math.txt \
    --model gpt-4o \
    --grade "必修二" \
    --publisher "人民教育出版社" \
    --subject "数学" \
    --book-prefix "math_g8c08_rjb"

  # 只处理第 8~9 章
  python extract_kg_from_textbook.py \
    --sections-dir out_sections \
    --prompt src/build_graph/prompt_math.txt \
    --start 8 \
    --end 9 \
    --grade "必修二" \
    --publisher "人民教育出版社" \
    --subject "数学"
        """
    )

    parser.add_argument(
        "--sections-dir",
        required=True,
        help="已分割的章节目录路径（必须包含 sections_index.json 和章节 md 文件）"
    )
    parser.add_argument("--prompt", required=True, help="prompt模板文件路径")
    parser.add_argument(
        "--model",
        default=None,
        help="模型名称 (默认: gpt-4)"
    )
    parser.add_argument("--grade", required=True, help="年级，如 '一年级'")
    parser.add_argument("--publisher", required=True, help="出版社，如 '人民教育出版社'")
    parser.add_argument("--subject", required=True, help="学科，如 '数学'")
    parser.add_argument("--book-prefix", default="", help="教材编号前缀，如 'math_7a_rjb'，用于生成全局唯一节点ID")
    parser.add_argument("--start", type=int, default=None, help="起始章节编号 (默认: 从第一个开始)")
    parser.add_argument("--end", type=int, default=None, help="结束章节编号（包含） (默认: 处理所有)")
    parser.add_argument("--chapters", type=str, default=None, help="指定要处理的章节编号列表，用逗号分隔（如 '2,4,6'），优先级高于 --start 和 --end")
    parser.add_argument("--output-dir", default="output", help="输出目录 (默认: output)")
    parser.add_argument("--no-intermediate", action="store_true", help="不保存中间结果")
    parser.add_argument("--api-key", default=None, help="API密钥（如果未设置环境变量）")
    parser.add_argument("--base-url", default=None, help="API基础URL（仅OpenAI）")

    args = parser.parse_args()

    prompt_path = args.prompt
    if not os.path.exists(args.sections_dir):
        print(f"ERROR: Sections directory not found: {args.sections_dir}")
        sys.exit(1)
    if not os.path.exists(prompt_path):
        print(f"ERROR: Prompt file not found: {prompt_path}")
        sys.exit(1)

    if args.model is None:
        args.model = "gpt-4"

    if not args.book_prefix or not str(args.book_prefix).strip():
        print("ERROR: --book-prefix 不能为空（例如 math_rjb_1a）")
        sys.exit(1)

    try:
        llm_kwargs = {"model": args.model}
        if args.api_key:
            llm_kwargs["api_key"] = args.api_key
        if args.base_url:
            llm_kwargs["base_url"] = args.base_url

        print(f"Creating openai client (model: {args.model})...")
        llm_client = create_llm_client(**llm_kwargs)
        print("LLM client created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create LLM client: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        prompt_builder = PromptBuilder(prompt_path)
    except Exception as e:
        print(f"ERROR: Failed to load prompt template: {e}")
        sys.exit(1)

    extractor = KnowledgeGraphExtractor(
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        grade=args.grade,
        publisher=args.publisher,
        subject=args.subject,
        book_prefix=args.book_prefix,
    )

    try:
        chapters_list = None
        if args.chapters:
            try:
                chapters_list = [int(ch.strip()) for ch in args.chapters.split(',') if ch.strip()]
                print(f"指定处理章节: {chapters_list}")
            except ValueError as e:
                print(f"ERROR: --chapters 参数格式错误，应为逗号分隔的数字（如 '2,4,6'）: {e}")
                sys.exit(1)

        extractor.extract_from_sections_dir(
            sections_dir=args.sections_dir,
            start_chapter=args.start,
            end_chapter=args.end,
            chapters=chapters_list,
            save_intermediate=not args.no_intermediate,
            output_dir=args.output_dir,
        )
    except KeyboardInterrupt:
        print("\n\nUser interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

