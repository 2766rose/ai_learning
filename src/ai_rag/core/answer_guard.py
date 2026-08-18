# -*- coding: utf-8 -*-
"""Anti-hallucination guard decision (pure function, unit-testable)."""
import re

REFUSAL_MESSAGE = "抱歉，知识库中未找到与您问题相关的信息。"

# 公司类问题关键词：命中说明是"企业内部知识"类问题，未检索到时需要拒绝
COMPANY_HINT = (
    "报销", "制度", "比例", "公积金", "考勤", "请假", "差旅",
    "设备", "故障", "培训", "福利", "工资", "年假", "病假",
    "加班", "年终奖", "休假", "补贴", "体检", "红线", "兼职",
    "试用", "转正", "绩效", "住宿", "高铁", "通讯费", "放假",
    "竞业", "持股", "入职", "薪酬", "五险一金", "餐补", "招待",
    "礼品", "公司", "员工手册", "手册", "OA", "审批", "申请",
)


def should_refuse(answer: str, kb_had_content: bool, other_tool_content: bool, query: str = "") -> bool:
    """Return True when a company-flavored question gets an ungrounded digit-bearing claim."""
    if not answer:
        return False
    if kb_had_content or other_tool_content:
        return False
    if "未找到" in answer or "没有找到" in answer:
        return False
    if not re.search(r"\d", answer):
        return False
    # 只拦截公司类问题的编造；常识/通识类问题允许模型用自身知识回答
    if query and not any(k in query for k in COMPANY_HINT):
        return False
    return True
