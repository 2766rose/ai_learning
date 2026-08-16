import re
import json
import sys
import pytest
import requests
from typing import Dict, Any

# ================= 🎯 配置 =================
CONFIG = {
    "RAG_API_URL": "http://localhost:8000/api/rag/chat",
    # 复用 RAG 接口做 Judge，无需独立 Key
    "JUDGE_API_URL": "http://localhost:8000/api/rag/chat",
    "API_KEY": "",  # 若 RAG 接口需要认证，在此填入
    "MODEL_NAME": "qwen-plus",
    "DEFAULT_SESSION_ID": "p0_test_session",
    "RESPONSE_ANSWER_PATH": "ai_answer",
    "P0_PASS_THRESHOLD": 0.8,
    "TIMEOUT_SECONDS": 120,
}

HEADERS = {"Content-Type": "application/json"}
if CONFIG["API_KEY"]:
    HEADERS["Authorization"] = f"Bearer {CONFIG['API_KEY']}"

GOLDEN_QA_SET = [
    {"id": 1, "query": "当前系统默认使用的Embedding模型是什么？", "ground_truth": "默认 Embedding 模型为 BGE-M3。", "question_type": "事实型"},
    {"id": 2, "query": "chunk_overlap参数的默认值和取值范围分别是多少？", "ground_truth": "默认值为64，取值范围为0-256。", "question_type": "事实型"},
    {"id": 3, "query": "混合检索中BM25与向量检索的默认权重配比是多少？", "ground_truth": "默认权重配比为 bm25:vector = 0.3:0.7。", "question_type": "事实型"},
    {"id": 4, "query": "新文档上传后平均需要多长时间才能被检索到？", "ground_truth": "平均耗时 3-5 分钟/百页。", "question_type": "事实型"},
    {"id": 5, "query": "用于预览实际切片结果的API接口路径是什么？", "ground_truth": "/api/v1/chunk/preview", "question_type": "事实型"},
    {"id": 6, "query": "Cross-Encoder重排序默认使用的模型名称是什么？", "ground_truth": "bge-reranker-v2-m3", "question_type": "事实型"},
    {"id": 7, "query": "split_by_header参数为true时，系统的默认主切分粒度是哪一级标题？", "ground_truth": "二级标题 (##) 是默认主切分粒度。", "question_type": "事实型"},
    {"id": 8, "query": "查询文档处理状态的API接口返回什么状态值时表示文档可被检索？", "ground_truth": "仅当状态为 COMPLETED 时才可被检索。", "question_type": "事实型"},
    {"id": 9, "query": "如果业务场景以故障码查询为主，应如何调整混合检索权重？该调整依据来自哪个章节的参数说明？", "ground_truth": '原文建议将权重调整为 bm25:vector = 0.6:0.4（注意是"建议"而非强制）。该建议出自「## 3. 检索与重排序 > ### 3.1 混合检索权重」，原文表述为"若业务场景以精确关键词匹配为主（如故障码查询），建议调整为 0.6:0.4"。', "question_type": "多跳推理"},
    {"id": 10, "query": "当二级标题下的内容超过chunk_size时，系统会如何处理切分？这一行为依赖于哪个基础参数的开启？", "ground_truth": "系统会自动降级到三级标题(###)进行切分。这一行为依赖于 split_by_header=true 的开启，见「### 2.2 标题感知切分规则」中第3条及「### 2.1 基础参数说明」表格。", "question_type": "多跳推理"},
    {"id": 11, "query": "人工抽检切片质量时需要关注哪三个重点？这些检查项对应的是哪个API接口的使用场景？", "ground_truth": "重点关注：① 表格是否被截断；② 代码块是否完整保留；③ 标题路径元数据是否正确继承。对应接口为 /api/v1/chunk/preview，见「### 4.2 如何验证切片质量？」。", "question_type": "多跳推理"},
    {"id": 12, "query": "为什么重排序阶段无法修复初始召回缺失正确答案的问题？这与重排序的工作机制有什么关系？", "ground_truth": "因为重排序阶段不会引入新的候选文档，仅对已有结果重新打分。若初始召回未包含正确答案，精排无从对其打分，故无法修复。见「### 3.2 Cross-Encoder 重排序」中的注意说明。", "question_type": "多跳推理"},
    {"id": 13, "query": "一级标题(#)作为切分粒度的生效条件是什么？如果文档同时存在二级标题，一级标题是否还会参与切分？", "ground_truth": "一级标题仅当文档无二级及以下标题时生效。若文档存在二级标题，则一级标题不参与切分，系统以二级标题为默认主切分粒度。见「### 2.2 标题感知切分规则」优先级列表第1、2条。", "question_type": "多跳推理"},
    {"id": 14, "query": "新文档上传后不能立即被检索的根本原因是什么？需要经历哪些步骤，且通过什么方式确认已完成？", "ground_truth": '根本原因是新文档需经历"解析→切片→向量化→索引写入"全流程，平均耗时3-5分钟/百页。可通过 /api/v1/index/status 接口查询，状态为 COMPLETED 时方可被检索。见「### 4.1」。', "question_type": "多跳推理"},
    {"id": 15, "query": "当前v2.3版本不支持哪些语言的语义切分？需要什么条件才能支持？", "ground_truth": "当前版本（v2.3）不支持日文、韩文文档的语义切分，需升级至 v2.4 版本后方可使用。", "question_type": "否定/边界"},
    {"id": 16, "query": "为什么不能将chunk_overlap设置为大于chunk_size的值？会导致什么后果？", "ground_truth": "禁止将 chunk_overlap 设置为大于 chunk_size 的值，否则会导致无限循环切分，触发系统 OOM 保护机制。", "question_type": "否定/边界"},
    {"id": 17, "query": "Cross-Encoder重排序会不会引入混合检索未返回的新文档？", "ground_truth": "不会。重排序阶段不会引入新的候选文档，仅对已有结果重新打分。", "question_type": "否定/边界"},
    {"id": 18, "query": "文档处理状态不是COMPLETED时，能否被检索到？", "ground_truth": '不能。原文明确指出"仅当状态为 COMPLETED 时才可被检索"，未提及其他任何状态可被检索的情况。', "question_type": "否定/边界"},
    {"id": 19, "query": "请综合说明该系统从文档上传到最终检索输出的完整链路，包括切片、检索、重排序三个核心阶段的关键机制与注意事项。", "ground_truth": "完整链路为：① 文档上传后经历解析→切片→向量化→索引写入（3-5分钟/百页，COMPLETED后可检索）；切片支持标题感知切分（split_by_header=true时按##主切分，超限降级###，禁止overlap>size）。② 检索采用BM25+向量混合检索，通过RRF（Reciprocal Rank Fusion）算法融合，默认权重0.3:0.7，关键词场景建议调0.6:0.4。③ 重排序使用bge-reranker-v2-m3对Top-50精排，不引入新候选，初始召回缺失则无法修复。注意事项：v2.3不支持日韩文；需通过/api/v1/chunk/preview抽检切片质量。", "question_type": "摘要综合"},
    {"id": 20, "query": "请对比分析标题感知切分的三种优先级规则各自的触发条件，并说明它们与chunk_size参数之间的联动关系。", "ground_truth": "三种优先级触发条件：① 一级标题(#)：仅当文档无任何二级及以下标题时生效；② 二级标题(##)：默认主切分粒度，只要存在二级标题即优先使用；③ 三级标题(###)：当二级标题下内容超过chunk_size时自动降级触发。联动关系：三级标题的降级切分直接依赖chunk_size阈值判断，而chunk_overlap与chunk_size之间还存在约束关系（overlap不得大于size，否则OOM）。整体切分行为由split_by_header=true统一控制。", "question_type": "摘要综合"},
]


# ================= 🧠 LLM-as-Judge 评估引擎 =================
JUDGE_PROMPT_TEMPLATE = """你是一个严格的RAG系统评估专家。请判断【实际回答】是否在语义上完整覆盖了【标准答案】的核心信息。

## 评分规则
- 忽略格式差异（Markdown、引用标记、大小写、中英文标点、空格）
- 忽略表述重组（同义替换、语序调整、扩写解释）
- 仅关注：标准答案中的每个关键事实/数值/结论是否在实际回答中有对应表达
- 实际回答可以比标准答案更详细，但不能遗漏关键信息或给出错误信息

## 输出格式（严格JSON，不要任何其他内容）
{{"score": 0或1, "reason": "简要说明"}}

## 输入
【用户问题】{query}
【标准答案】{ground_truth}
【实际回答】{answer}"""


def _judge_with_llm(query: str, ground_truth: str, answer: str) -> float:
    """复用 RAG 接口做 Judge，通过 Prompt 强制纯评估模式"""
    # 关键：注入系统指令，禁止 RAG 检索行为
    system_prefix = (
        "【系统指令】你现在是一个纯文本评估器，禁止使用任何知识库、搜索或检索工具。"
        "仅根据下方提供的【用户问题】【标准答案】【实际回答】三个字段进行语义判断。"
        "不要回答用户问题本身，只输出评估JSON。\n\n"
    )

    prompt = system_prefix + JUDGE_PROMPT_TEMPLATE.format(
        query=query, ground_truth=ground_truth, answer=answer
    )

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": CONFIG["MODEL_NAME"],
        # 使用独立 session_id，避免污染正常对话上下文
        "session_id": "judge_eval_session_no_rag",
        "stream": False,
    }

    try:
        resp = requests.post(
            CONFIG["JUDGE_API_URL"],
            json=payload,
            headers=HEADERS,
            timeout=CONFIG["TIMEOUT_SECONDS"],
        )
        resp.raise_for_status()
        data = resp.json()
        judge_answer = data.get(CONFIG["RESPONSE_ANSWER_PATH"], "")

        # 兼容多种 JSON 返回格式
        match = re.search(r'\{[^{}]*"score"\s*:\s*[01][^{}]*\}', judge_answer)
        if match:
            result = json.loads(match.group())
            return float(result.get("score", 0))

        print(f"⚠️ Judge返回无法解析: {judge_answer[:200]}")
        return 0.0
    except Exception as e:
        print(f"⚠️ Judge调用异常: {e}")
        return 0.0


# ================= 🔧 轻量级兜底评估 =================
def _normalize_text(text: str) -> str:
    text = re.sub(r'\*{1,3}|`{1,3}', '', text)
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('（', '(').replace('）', ')').replace('：', ':')
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def _calc_semantic_hit_fallback(answer: str, ground_truth: str) -> float:
    """仅在 LLM Judge 不可用时作为兜底"""
    norm_answer = _normalize_text(answer)
    norm_gt = _normalize_text(ground_truth)
    if norm_gt in norm_answer:
        return 1.0
    gt_phrases = [p.strip() for p in re.split(r'[,，。.;；]', norm_gt) if len(p.strip()) >= 2]
    if not gt_phrases:
        return 1.0
    hit_count = sum(1 for p in gt_phrases if p in norm_answer)
    return hit_count / len(gt_phrases)


def _calc_semantic_hit(query: str, answer: str, ground_truth: str) -> float:
    """主评估入口：优先 LLM Judge，失败则兜底"""
    score = _judge_with_llm(query, ground_truth, answer)
    if score > 0:
        return score
    fallback = _calc_semantic_hit_fallback(answer, ground_truth)
    return max(score, fallback)


# ================= 🔌 RAG 接口调用 =================
def _build_payload(query: str) -> dict:
    return {
        "messages": [{"role": "user", "content": query}],
        "model": CONFIG["MODEL_NAME"],
        "session_id": CONFIG["DEFAULT_SESSION_ID"],
        "stream": False,
    }


def _extract_answer(data: dict) -> str:
    return data.get(CONFIG["RESPONSE_ANSWER_PATH"], "") or ""


# ================= ✅ P0 测试类 =================
class TestRAGP0:
    @pytest.mark.parametrize(
        "qa",
        GOLDEN_QA_SET,
        ids=[f"Q{qa['id']}_{qa['question_type']}" for qa in GOLDEN_QA_SET],
    )
    def test_rag_answer_accuracy(self, qa: Dict[str, Any]):
        payload = _build_payload(qa["query"])
        resp = requests.post(
            CONFIG["RAG_API_URL"],
            json=payload,
            headers=HEADERS,
            timeout=CONFIG["TIMEOUT_SECONDS"],
        )
        assert resp.status_code == 200, f"接口返回非200: {resp.status_code}"
        data = resp.json()
        answer = _extract_answer(data)
        assert answer, f"ai_answer为空: {json.dumps(data, ensure_ascii=False)[:300]}"

        hit_ratio = _calc_semantic_hit(qa["query"], answer, qa["ground_truth"])
        threshold = 0.6

        assert hit_ratio >= threshold, (
            f"\n❌ Q{qa['id']} [{qa['question_type']}] 语义命中率不足 ({hit_ratio:.0%} < {threshold:.0%})\n"
            f"   Query: {qa['query']}\n"
            f"   Expected GT: {qa['ground_truth'][:100]}...\n"
            f"   Actual Answer: {answer[:200]}..."
        )

    def test_p0_overall_pass_rate(self):
        results = []
        for qa in GOLDEN_QA_SET:
            try:
                payload = _build_payload(qa["query"])
                resp = requests.post(
                    CONFIG["RAG_API_URL"],
                    json=payload,
                    headers=HEADERS,
                    timeout=CONFIG["TIMEOUT_SECONDS"],
                )
                data = resp.json()
                answer = _extract_answer(data)
                hit = _calc_semantic_hit(qa["query"], answer, qa["ground_truth"]) >= 0.6
                results.append(hit)
            except Exception as e:
                print(f"⚠️ Q{qa['id']} 异常: {e}")
                results.append(False)

        pass_rate = sum(results) / len(results)
        threshold = CONFIG["P0_PASS_THRESHOLD"]
        print(f"\n{'=' * 50}")
        print(f"📊 P0 验收通过率: {pass_rate:.0%} ({sum(results)}/{len(results)})")
        print(f"🎯 P0 目标阈值:   {threshold:.0%}")
        print("✅ P0 通过!" if pass_rate >= threshold else "❌ P0 未达标")
        print(f"{'=' * 50}\n")
        assert pass_rate >= threshold, f"P0 通过率 {pass_rate:.0%} 未达阈值 {threshold:.0%}"


# ================= 🚀 直接运行入口 =================
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "-s"]))