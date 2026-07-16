"""
圆桌会议 Pipe for Open WebUI
多模型圆桌讨论：2轮 + 主持人总结
直连各家模型API，不走OWUI中转，避免死锁

支持提供商（大小写不敏感，中英文都认）：
  TX / 腾讯  → 腾讯云混元
  BL / 百炼  → 百炼云（阿里DashScope）
  SS / 书生  → 书生 InternLM

用法示例：

  第一次讨论：
    为什么docker可以有-d参数而酒馆没有？
    【参会模型】
    TX:hy3
    BL:kimi-k2.6
    BL:glm-5.2
    【主持人】SS:internlm2.5-latest

  继续讨论（同一批模型，不用重写列表）：
    继续讨论:请从安全角度重新审视

  续议（新议题，自动带上次纪要）：
    如何把方案落地实施？
    【参会模型】
    TX:hy3
    BL:deepseek-v4-pro
    【主持人】SS:internlm2.5-latest
    续议
"""

import requests
import concurrent.futures
import re
from pydantic import BaseModel, Field
from typing import Generator

TITLE = "圆桌会议"


class Pipe:
    class Valves(BaseModel):
        # 百炼云（阿里DashScope）
        BAILIAN_API_KEY: str = Field(
            default="", description="百炼云API Key",
            json_schema_extra={"format": "password"},
        )
        BAILIAN_BASE_URL: str = Field(
            default="", description="百炼云Base URL，如 https://ws-4ma5uq91cq7f3lbz.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        )
        # 腾讯云
        TENCENT_API_KEY: str = Field(
            default="", description="腾讯云API Key",
            json_schema_extra={"format": "password"},
        )
        TENCENT_BASE_URL: str = Field(
            default="", description="腾讯云Base URL，如 https://tokenhub.tencentmaas.cn/v1"
        )
        # 书生InternLM
        INTERNLM_API_KEY: str = Field(
            default="", description="书生InternLM API Key",
            json_schema_extra={"format": "password"},
        )
        INTERNLM_BASE_URL: str = Field(
            default="", description="书生Base URL，如 https://chat.intern-ai.org.cn/api/v1"
        )
        TEMPERATURE: float = Field(default=0.8, description="温度")
        MAX_TOKENS: int = Field(default=2000, description="每次发言最大token")

    def __init__(self):
        self.valves = self.Valves()

    # 前缀映射（大小写不敏感，支持中英文）
    _PREFIX_MAP = {
        "tx": ("tencent", "TENCENT"),
        "腾讯": ("tencent", "TENCENT"),
        "bl": ("bailian", "BAILIAN"),
        "百炼": ("bailian", "BAILIAN"),
        "ss": ("internlm", "INTERNLM"),
        "书生": ("internlm", "INTERNLM"),
    }

    _PROVIDER_NAME = {
        "tencent": "腾讯云",
        "bailian": "百炼云",
        "internlm": "书生",
    }

    def _resolve_provider(self, model_id: str):
        """根据 前缀:模型ID 解析API提供商。
        返回 (base_url, api_key, actual_model, provider_key) 或 None。
        """
        v = self.valves
        if ":" not in model_id:
            return None
        prefix, actual_model = model_id.split(":", 1)
        prefix_clean = prefix.strip()
        prefix_lower = prefix_clean.lower()

        if prefix_lower in self._PREFIX_MAP:
            provider_key, valve_key = self._PREFIX_MAP[prefix_lower]
        elif prefix_clean in self._PREFIX_MAP:
            provider_key, valve_key = self._PREFIX_MAP[prefix_clean]
        else:
            return None

        base_url = getattr(v, f"{valve_key}_BASE_URL")
        api_key = getattr(v, f"{valve_key}_API_KEY")
        return base_url, api_key, actual_model, provider_key

    def _parse_participant(self, line: str):
        """解析一行参会模型。
        格式：前缀:模型ID  或  前缀:模型ID@显示名
        返回 dict 或 None。
        """
        line = line.strip()
        if not line:
            return None

        label = ""
        if "@" in line:
            model_part, label = line.rsplit("@", 1)
            label = label.strip()
            model_id = model_part.strip()
        else:
            model_id = line

        info = self._resolve_provider(model_id)
        if info is None:
            return None

        base_url, api_key, actual_model, provider_key = info
        if not label:
            label = actual_model

        return {
            "label": label,
            "model": model_id,           # 完整的 前缀:模型ID（用于显示和解析）
            "actual_model": actual_model,  # 实际发给API的ID
            "base_url": base_url,
            "api_key": api_key,
            "provider": provider_key,
        }

    def _call_model(self, p, prompt, system_prompt="") -> str:
        """直连模型API调用。p 是 participant dict。"""
        url = f"{p['base_url'].rstrip('/')}/chat/completions"
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if p["api_key"]:
            headers["Authorization"] = f"Bearer {p['api_key']}"

        resp = requests.post(
            url,
            json={
                "model": p["actual_model"],
                "messages": msgs,
                "stream": False,
                "temperature": self.valves.TEMPERATURE,
                "max_tokens": self.valves.MAX_TOKENS,
            },
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _parallel_call(self, participants, prompt, system_prompt):
        """并发调用多个模型，按完成顺序返回结果。"""
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(participants)
        ) as executor:
            futures = {}
            for p in participants:
                future = executor.submit(self._call_model, p, prompt, system_prompt)
                futures[future] = p
            for future in concurrent.futures.as_completed(futures):
                p = futures[future]
                try:
                    text = future.result()
                except Exception as e:
                    text = f"[调用失败: {e}]"
                results.append((p["label"], p["model"], text))
        return results

    def _extract_prev_round(self, messages):
        """从对话历史中提取上一轮圆桌会议的信息。
        返回 {"participants": [...], "moderator": {...}, "transcript": "...", "minutes": "..."} 或 None。
        """
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if "圆桌会议" not in content:
                continue

            # 解析参与者：从 ### {label}（{model}） 提取
            participants = []
            seen_models = set()
            for match in re.finditer(r'###\s+(.+?)（(.+?)）', content):
                label = match.group(1).strip()
                model_id = match.group(2).strip()
                if model_id in seen_models:
                    continue
                seen_models.add(model_id)
                p = self._parse_participant(model_id)
                if p:
                    participants.append(p)

            if not participants:
                continue

            # 解析主持人：从 **主持人：** xxx 提取
            moderator = None
            mod_match = re.search(r'\*\*主持人[：:]\*\*\s*(.+)', content)
            if mod_match:
                mod_str = mod_match.group(1).strip()
                moderator = self._parse_participant(mod_str)
            if not moderator:
                moderator = participants[0]

            # 提取纪要
            minutes = ""
            if "【纪要】" in content:
                minutes = content.split("【纪要】", 1)[1].strip()
            elif "## 主持人总结" in content:
                minutes = content.split("## 主持人总结", 1)[1].strip()

            return {
                "participants": participants,
                "moderator": moderator,
                "transcript": content,
                "minutes": minutes,
            }
        return None

    def pipe(self, body: dict, __user__: dict) -> Generator[str, None, None]:
        messages = body.get("messages", [])
        user_msg = messages[-1].get("content", "") if messages else ""
        if not user_msg:
            yield "请输入议题。"
            return

        user_stripped = user_msg.strip()

        # ===== 模式一：继续讨论 =====
        if user_stripped.startswith("继续讨论"):
            rest = user_stripped[4:]  # 去掉"继续讨论"
            rest = re.sub(r'^[：:\s]+', '', rest)  # 去掉开头的冒号/空白
            continue_topic = rest.strip()

            prev = self._extract_prev_round(messages)
            if prev is None:
                yield "未找到上一轮圆桌会议记录，无法继续讨论。\n请重新发起议题并指定参会模型。"
                return

            participants = prev["participants"]
            moderator = prev["moderator"]
            prev_transcript = prev["transcript"]

            yield f"# 圆桌会议（继续讨论）\n\n"
            yield f"**参与者：** {', '.join(p['label'] for p in participants)}\n"
            yield f"**主持人：** {moderator['model']}\n"
            if continue_topic:
                yield f"**方向：** {continue_topic}\n"
            yield f"---\n\n"

            R1_PROMPT = f"上一轮讨论记录：\n{prev_transcript}\n\n"
            if continue_topic:
                R1_PROMPT += f"请围绕以下方向继续深入讨论：\n{continue_topic}\n\n"
            else:
                R1_PROMPT += "请继续深入讨论，提出新的观点或补充之前的不足。\n"
            R1_PROMPT += "用中文，控制在400字以内。"

            R1_SYS = "你是一场圆桌会议的参与者。上一轮讨论已经结束，现在进入新一轮。请直接发表观点，用中文。"

            yield f"## 继续讨论\n\n"
            r1_results = self._parallel_call(participants, R1_PROMPT, R1_SYS)
            for label, model, text in r1_results:
                yield f"### {label}（{model}）\n\n{text}\n\n"

            r1_transcript = "\n\n".join(
                f"【{p['label']}（{p['model']}）】\n{text}"
                for p, (_, _, text) in zip(participants, r1_results)
            )

            yield f"## 点评与表态\n\n"
            R2_PROMPT = (
                f"上一轮讨论记录：\n{prev_transcript}\n\n"
                f"本轮发言记录：\n{r1_transcript}\n\n"
                "请结合两轮讨论，点评其他参与者的观点，明确表态：\n"
                "1. 你同意谁的观点？为什么？\n"
                "2. 你反对谁的观点？为什么？\n"
                "3. 你要补充或修改自己的观点吗？\n"
                "用中文，控制在400字以内。"
            )
            R2_SYS = "你是圆桌会议参与者，现在进入点评环节。请认真点评他人观点并表态，用中文。"

            r2_results = self._parallel_call(participants, R2_PROMPT, R2_SYS)
            for label, model, text in r2_results:
                yield f"### {label}（{model}）\n\n{text}\n\n"

            r2_transcript = "\n\n".join(
                f"【{p['label']}（{p['model']}）】\n{text}"
                for p, (_, _, text) in zip(participants, r2_results)
            )

            full_transcript = (
                f"上轮讨论：\n{prev_transcript}\n\n"
                f"本轮第一轮：\n{r1_transcript}\n\n"
                f"本轮点评：\n{r2_transcript}"
            )

            yield f"## 主持人总结\n\n"
            MOD_PROMPT = (
                f"议题方向：{continue_topic or '继续深入讨论'}\n\n"
                f"完整讨论记录：\n{full_transcript}\n\n"
                "你是会议主持人，请归纳总结：\n"
                "1. 找出观点一致的阵营（≥2人观点趋同合并为一个方案）\n"
                "2. 坚持独立观点的人单独成方案\n"
                "3. 每个方案标注：支持者、核心观点、风险点、备选路径\n"
                "4. 至少给出2个方案，如果只有1个共识也要给出2条执行路径\n"
                "5. 末尾输出【纪要】标签，包含简明纪要供下次会议参考\n"
                "用中文。"
            )
            MOD_SYS = "你是圆桌会议主持人/参谋。你负责归纳多方观点，整理方案矩阵，不替用户做决定。用中文。"

            try:
                mod_text = self._call_model(moderator, MOD_PROMPT, MOD_SYS)
                yield mod_text
            except Exception as e:
                yield f"[主持人调用失败: {e}]"
            return

        # ===== 模式二/三：正常模式 / 续议模式 =====
        is_resume = bool(re.search(r'续议[：:]?\s*$', user_stripped)) or '【续议】' in user_stripped

        topic_raw = user_msg
        if is_resume:
            topic_raw = topic_raw.replace('【续议】', '')
            topic_raw = re.sub(r'续议[：:]?\s*$', '', topic_raw.strip())

        topic = topic_raw
        moderator_str = ""
        prev_minutes = ""

        mod_marker = "【主持人】"
        mod_marker2 = "【支持人】"  # 错别字容错
        models_marker = "【参会模型】"
        minutes_marker = "【上次纪要】"

        # 提取主持人
        for mm in [mod_marker, mod_marker2]:
            if mm in topic:
                parts = topic.split(mm, 1)
                topic = parts[0]
                rest = parts[1]
                if minutes_marker in rest:
                    m_parts = rest.split(minutes_marker, 1)
                    moderator_str = m_parts[0].strip()
                    prev_minutes = m_parts[1].strip()
                else:
                    moderator_str = rest.strip()
                break
        else:
            if minutes_marker in topic:
                parts = topic.split(minutes_marker, 1)
                topic = parts[0].strip()
                prev_minutes = parts[1].strip()

        # 续议模式：从对话历史自动提取上次纪要
        if is_resume and not prev_minutes:
            prev = self._extract_prev_round(messages)
            if prev:
                prev_minutes = prev["minutes"] or prev["transcript"]

        # 提取参会模型
        participants = []
        parse_errors = []
        if models_marker in topic:
            parts = topic.split(models_marker, 1)
            topic = parts[0].strip()
            rest = parts[1]
            for mm in [mod_marker, mod_marker2, minutes_marker]:
                if mm in rest:
                    rest = rest.split(mm)[0]
            for line in rest.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                p = self._parse_participant(line)
                if p:
                    participants.append(p)
                else:
                    parse_errors.append(
                        f"❌ 无法解析 `{line}`，请用 前缀:模型ID 格式（如 TX:hy3）\n"
                    )
        else:
            yield (
                "请用以下格式发消息：\n\n"
                "你的议题\n"
                "【参会模型】\n"
                "TX:hy3\n"
                "BL:kimi-k2.6\n"
                "SS:internlm2.5-latest\n"
                "【主持人】SS:internlm2.5-latest\n"
            )
            return

        if parse_errors:
            for e in parse_errors:
                yield e
            yield "\n支持的前缀：TX/腾讯、BL/百炼、SS/书生\n"
            return

        topic = topic.strip()
        if not topic:
            yield "请输入议题。"
            return
        if not participants:
            yield "请至少指定一个参会模型。"
            return

        # 解析主持人
        moderator = None
        if moderator_str:
            moderator = self._parse_participant(moderator_str)
        if not moderator:
            moderator = participants[0]
            yield f"⚠️ 未指定主持人，默认使用 {moderator['label']}\n\n"

        # 验证 API Key 和 Base URL
        errors = []
        seen_providers = set()
        for p in participants:
            if p["provider"] in seen_providers:
                continue
            seen_providers.add(p["provider"])
            pname = self._PROVIDER_NAME[p["provider"]]
            if not p["api_key"]:
                errors.append(f"❌ {pname} API Key 未配置\n")
            if not p["base_url"]:
                errors.append(f"❌ {pname} Base URL 未配置\n")
        if moderator["provider"] not in seen_providers:
            pname = self._PROVIDER_NAME[moderator["provider"]]
            if not moderator["api_key"]:
                errors.append(f"❌ 主持人（{pname}）API Key 未配置\n")
            if not moderator["base_url"]:
                errors.append(f"❌ 主持人（{pname}）Base URL 未配置\n")
        if errors:
            for e in errors:
                yield e
            yield "\n请在 Pipe 设置中配置对应的 API Key 和 Base URL。\n"
            return

        # === 会议头 ===
        yield f"# 圆桌会议\n\n"
        yield f"**议题：** {topic}\n"
        yield f"**参与者：** {', '.join(p['label'] for p in participants)}\n"
        yield f"**主持人：** {moderator['model']}\n"
        if prev_minutes:
            yield f"**参考：** 上次会议纪要\n"
        yield f"---\n\n"

        R1_PROMPT = f"议题：{topic}\n\n请直接发表你的观点，简洁有力，不要重复议题。"
        if prev_minutes:
            R1_PROMPT = (
                f"议题：{topic}\n\n"
                f"上次纪要：\n{prev_minutes}\n\n"
                "请在上次讨论基础上发表你的观点。"
            )

        R1_SYS = "你是一场圆桌会议的参与者。请直接发表观点，用中文回答，控制在300字以内。"

        # === 第一轮 ===
        yield f"## 第一轮：各自观点\n\n"
        r1_results = self._parallel_call(participants, R1_PROMPT, R1_SYS)
        for label, model, text in r1_results:
            yield f"### {label}（{model}）\n\n{text}\n\n"

        r1_transcript = "\n\n".join(
            f"【{p['label']}（{p['model']}）】\n{text}"
            for p, (_, _, text) in zip(participants, r1_results)
        )

        # === 第二轮 ===
        yield f"## 第二轮：点评与表态\n\n"
        R2_PROMPT = (
            f"议题：{topic}\n\n"
            f"第一轮发言记录：\n{r1_transcript}\n\n"
            "请点评其他参与者的观点，明确表态：\n"
            "1. 你同意谁的观点？为什么？\n"
            "2. 你反对谁的观点？为什么？\n"
            "3. 你要补充或修改自己的观点吗？\n"
            "用中文，控制在400字以内。"
        )
        R2_SYS = "你是圆桌会议参与者，现在进入第二轮。请认真点评他人观点并表态，用中文。"

        r2_results = self._parallel_call(participants, R2_PROMPT, R2_SYS)
        for label, model, text in r2_results:
            yield f"### {label}（{model}）\n\n{text}\n\n"

        r2_transcript = "\n\n".join(
            f"【{p['label']}（{p['model']}）】\n{text}"
            for p, (_, _, text) in zip(participants, r2_results)
        )

        full_transcript = f"第一轮：\n{r1_transcript}\n\n第二轮：\n{r2_transcript}"

        # === 主持人总结 ===
        yield f"## 主持人总结\n\n"
        MOD_PROMPT = (
            f"议题：{topic}\n\n"
            f"讨论记录：\n{full_transcript}\n\n"
            "你是会议主持人，请归纳总结：\n"
            "1. 找出观点一致的阵营（≥2人观点趋同合并为一个方案）\n"
            "2. 坚持独立观点的人单独成方案\n"
            "3. 每个方案标注：支持者、核心观点、风险点、备选路径\n"
            "4. 至少给出2个方案，如果只有1个共识也要给出2条执行路径\n"
            "5. 末尾输出【纪要】标签，包含简明纪要供下次会议参考\n"
            "用中文。"
        )
        MOD_SYS = "你是圆桌会议主持人/参谋。你负责归纳多方观点，整理方案矩阵，不替用户做决定。用中文。"

        try:
            mod_text = self._call_model(moderator, MOD_PROMPT, MOD_SYS)
            yield mod_text
        except Exception as e:
            yield f"[主持人调用失败: {e}]"
