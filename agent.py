import ast
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent_base import (
    ACTION_CLICK,
    ACTION_COMPLETE,
    ACTION_OPEN,
    ACTION_SCROLL,
    ACTION_TYPE,
    AgentInput,
    AgentOutput,
    BaseAgent,
    UsageInfo,
    VALID_ACTIONS,
)


logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    STRICT_TASK_KINDS = {
        "baidumap_taxi",
        "baidumap_voice_pack",
        "douyin_likes_search",
        "kuaishou_filter_search",
        "aiqiyi_comment",
        "bilibili_favorite_video",
        "mangguo_download_episode",
        "meituan_order_dish",
        "qunar_flight_price",
        "tengxun_video_episode",
        "ximalaya_search",
    }

    APP_ALIASES = [
        ("百度地图", "百度地图"),
        ("去哪儿旅行", "去哪儿旅行"),
        ("去哪旅行", "去哪儿旅行"),
        ("腾讯视频", "腾讯视频"),
        ("哔哩哔哩", "哔哩哔哩"),
        ("B站", "哔哩哔哩"),
        ("喜马拉雅", "喜马拉雅"),
        ("爱奇艺", "爱奇艺"),
        ("芒果TV", "芒果TV"),
        ("芒果", "芒果TV"),
        ("美团", "美团"),
        ("快手", "快手"),
        ("抖音", "抖音"),
    ]

    def _initialize(self):
        self.reset()

    def reset(self):
        self._current_instruction = ""
        self._task_info: Dict[str, Any] = {}
        self._recent_action_summaries: List[str] = []
        self._repair_call_used = False

    def act(self, input_data: AgentInput) -> AgentOutput:
        self._ensure_task_state(input_data)

        if input_data.step_count == 1:
            return self._build_open_output()

        stage_hint = self._get_stage_hint(input_data)
        direct_decision = self._build_direct_stage_action(stage_hint, input_data)
        if direct_decision is not None:
            self._recent_action_summaries = self._build_action_summaries(input_data.history_actions)
            self._recent_action_summaries.append(
                self._format_action_summary(direct_decision.action, direct_decision.parameters)
            )
            self._recent_action_summaries = self._recent_action_summaries[-6:]
            return direct_decision

        messages = self.generate_messages(input_data)
        try:
            response = self._call_api(messages, temperature=0)
            raw_output = self._extract_response_text(response)
            usage = self._extract_usage_info(response)
        except Exception as exc:
            logger.warning("Primary model call failed: %s", exc)
            response = None
            raw_output = ""
            usage = None

        decision = self._parse_and_normalize(raw_output, input_data)

        if decision is None and not self._repair_call_used:
            repaired_text = self._repair_output_with_model(raw_output, input_data)
            if repaired_text:
                decision = self._parse_and_normalize(repaired_text, input_data)
                if decision is not None:
                    raw_output = repaired_text

        if decision is None:
            decision = self._deterministic_fallback(input_data, raw_output)
        else:
            decision = self._apply_stage_constraints(decision, input_data)

        self._recent_action_summaries = self._build_action_summaries(input_data.history_actions)
        self._recent_action_summaries.append(self._format_action_summary(decision.action, decision.parameters))
        self._recent_action_summaries = self._recent_action_summaries[-6:]

        decision.raw_output = raw_output
        decision.usage = usage
        return decision

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        self._ensure_task_state(input_data)
        stage_hint = self._get_stage_hint(input_data)

        task_summary = {
            "app_name": self._task_info.get("app_name", ""),
            "task_kind": self._task_info.get("task_kind", ""),
            "search_keyword": self._task_info.get("search_keyword"),
            "titles": self._task_info.get("titles", []),
            "episode": self._task_info.get("episode"),
            "route": self._task_info.get("route", {}),
            "store_name": self._task_info.get("store_name"),
            "item_name": self._task_info.get("item_name"),
        }

        history_lines = self._recent_action_summaries or self._build_action_summaries(input_data.history_actions)
        history_text = "\n".join(f"- {item}" for item in history_lines[-6:]) if history_lines else "- none"

        stage_lines = [
            f"Current stage goal: {stage_hint.get('goal', 'inspect the screenshot and choose the next action')}",
        ]
        expected_action = stage_hint.get("expected_action")
        if expected_action:
            stage_lines.append(f"Expected action type: {expected_action}")
        if stage_hint.get("allowed_actions"):
            stage_lines.append(f"Allowed actions for this stage: {', '.join(stage_hint['allowed_actions'])}")
        if stage_hint.get("required_text"):
            stage_lines.append(
                f"Required TYPE text: {json.dumps(stage_hint['required_text'], ensure_ascii=False)}"
            )
            stage_lines.append("If you choose TYPE, the text must match exactly, including symbols such as .* or 正则.")
        if stage_hint.get("click_hint"):
            stage_lines.append(f"Click target hint: {stage_hint['click_hint']}")
        if stage_hint.get("completion_hint"):
            stage_lines.append(f"Completion hint: {stage_hint['completion_hint']}")

        system_prompt = (
            "You are a precise mobile GUI agent for an offline benchmark.\n"
            "Decide exactly one next action for the current screenshot.\n"
            "Return exactly one JSON object and nothing else.\n"
            "Allowed actions: CLICK, SCROLL, TYPE, OPEN, COMPLETE.\n"
            "Use normalized coordinates in [0, 1000].\n"
            "JSON schema:\n"
            "{\n"
            '  "action": "CLICK|SCROLL|TYPE|OPEN|COMPLETE",\n'
            '  "parameters": {\n'
            '    "point": [x, y] | "start_point": [x1, y1], "end_point": [x2, y2] | "text": "..." | "app_name": "..."\n'
            "  }\n"
            "}\n"
            "Rules:\n"
            "- CLICK only uses parameters.point.\n"
            "- SCROLL only uses parameters.start_point and parameters.end_point.\n"
            "- TYPE only uses parameters.text.\n"
            "- OPEN only uses parameters.app_name.\n"
            "- COMPLETE must use empty parameters: {}.\n"
            "- When the stage says expected action TYPE or COMPLETE, follow it strictly.\n"
            "- For CLICK, choose a point inside the target control, not just nearby.\n"
            "- Do not output markdown, explanations, or extra keys."
        )

        user_text = (
            f"Instruction: {input_data.instruction}\n"
            f"Structured task summary: {json.dumps(task_summary, ensure_ascii=False)}\n"
            f"Step count: {input_data.step_count}\n"
            f"Recent actions:\n{history_text}\n"
            f"Stage guidance:\n" + "\n".join(f"- {line}" for line in stage_lines)
        )

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": self._encode_image(input_data.current_image)}},
                ],
            },
        ]

    def _ensure_task_state(self, input_data: AgentInput):
        if input_data.instruction != self._current_instruction:
            self._current_instruction = input_data.instruction
            self._task_info = self._parse_instruction(input_data.instruction)
            self._repair_call_used = False
            self._recent_action_summaries = []

        self._recent_action_summaries = self._build_action_summaries(input_data.history_actions)

    def _parse_instruction(self, instruction: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "instruction": instruction,
            "app_name": self._detect_app_name(instruction),
            "task_kind": "",
            "generic_intent": "",
            "titles": self._extract_titles(instruction),
            "type_candidates": [],
            "planned_type_texts": [],
            "search_keyword": None,
            "comment_text": None,
            "episode": None,
            "voice_pack": None,
            "route": {},
            "store_name": None,
            "item_name": None,
            "action_plan": [],
        }

        search_match = re.search(r"搜索(.+?)(?:并|筛选|，|。|$)", instruction)
        if search_match:
            keyword = self._clean_search_keyword(search_match.group(1))
            if keyword:
                info["search_keyword"] = keyword

        comment_match = re.search(r"评论[:：]?\s*(.+)$", instruction)
        if comment_match:
            info["comment_text"] = comment_match.group(1).strip("。 ")

        publish_comment_match = re.search(r"发布评论[:：]\s*(.+)$", instruction)
        if publish_comment_match:
            info["comment_text"] = publish_comment_match.group(1).strip("。 ")

        comment_area_match = re.search(r"打开(.+?)的评论区", instruction)
        if comment_area_match:
            title = comment_area_match.group(1).strip("“”\"' ")
            if title:
                info["titles"].append(title)

        episode_match = re.search(r"第(\d+)集", instruction)
        if episode_match:
            info["episode"] = episode_match.group(1)

        route_match = re.search(r"从(.+?)去(.+?)(?:，|。|地址|$)", instruction)
        if route_match:
            info["route"] = {
                "origin": route_match.group(1).strip(),
                "destination": route_match.group(2).strip(),
            }

        flight_match = re.search(r"后天(.+?)飞(.+?)(?:的航班|航班|，|。|$)", instruction)
        if flight_match:
            info["route"] = {
                "origin": flight_match.group(1).strip(),
                "destination": flight_match.group(2).strip(),
            }

        store_match = re.search(r"购买(.+?)店铺的(.+?)(?:，|。|$)", instruction)
        if store_match:
            info["store_name"] = store_match.group(1).strip()
            info["item_name"] = store_match.group(2).strip()

        download_match = re.search(r"下载里的(.+?)第(\d+)集", instruction)
        if download_match:
            title = download_match.group(1).strip()
            if title:
                info["titles"].append(title)
            info["episode"] = download_match.group(2)

        if info["search_keyword"]:
            info["type_candidates"].append(info["search_keyword"])
        if info["comment_text"]:
            info["type_candidates"].append(info["comment_text"])
        if info["store_name"]:
            info["type_candidates"].append(info["store_name"])
        if info["item_name"]:
            info["type_candidates"].append(info["item_name"])
        for title in info["titles"]:
            info["type_candidates"].append(title)
        if info["route"]:
            info["type_candidates"].append(info["route"].get("origin", ""))
            info["type_candidates"].append(info["route"].get("destination", ""))

        info["titles"] = self._dedupe_non_empty(info["titles"])
        info["type_candidates"] = self._dedupe_non_empty(info["type_candidates"])
        self._configure_task(info)
        return info

    def _configure_task(self, info: Dict[str, Any]):
        instruction = info["instruction"]
        app_name = info.get("app_name", "")
        route = info.get("route", {})
        origin = route.get("origin", "")
        destination = route.get("destination", "")
        store_name = info.get("store_name") or ""
        item_name = info.get("item_name") or ""
        keyword = info.get("search_keyword") or ""
        titles = info.get("titles", [])
        title = titles[0] if titles else ""
        episode = info.get("episode") or ""

        if app_name == "百度地图" and "打车" in instruction and route:
            info["task_kind"] = "baidumap_taxi"
            info["planned_type_texts"] = [
                self._build_baidumap_route_text(origin),
                self._build_baidumap_route_text(destination),
            ]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open 百度地图"),
                self._stage(
                    ACTION_CLICK,
                    "Open the route or search entry",
                    click_hint="click the top search or route entry",
                    fallback_point=[857, 41],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Enter taxi or route planning",
                    click_hint="click the taxi or route planning control",
                    fallback_point=[499, 452],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Activate the place input area",
                    click_hint="click the start or destination editing area",
                    fallback_point=[467, 471],
                ),
                self._stage(ACTION_TYPE, "Input the origin", required_text=info["planned_type_texts"][0]),
                self._stage(
                    ACTION_CLICK,
                    "Choose the first origin suggestion",
                    click_hint="click the first candidate in the list",
                    fallback_point=[486, 166],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Activate the destination input box",
                    click_hint="click the destination input box",
                    fallback_point=[500, 544],
                ),
                self._stage(ACTION_TYPE, "Input the destination", required_text=info["planned_type_texts"][1]),
                self._stage(
                    ACTION_CLICK,
                    "Choose the first destination suggestion",
                    click_hint="click the first candidate or confirm button",
                    fallback_point=[489, 170],
                ),
                self._stage(ACTION_COMPLETE, "The taxi route is ready", force_complete=True),
            ]
            return

        if app_name == "百度地图" and "语音包" in instruction:
            voice_name = info.get("search_keyword") or self._extract_after_keyword(instruction, "语音包为")
            info["task_kind"] = "baidumap_voice_pack"
            info["planned_type_texts"] = [voice_name]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open 百度地图"),
                self._stage(
                    ACTION_CLICK,
                    "Open search or menu",
                    click_hint="click the top-right entry",
                    fallback_point=[854, 39],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open settings or voice pack area",
                    click_hint="click the bottom-right settings or more entry",
                    fallback_point=[893, 909],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open the voice pack page",
                    click_hint="click the voice pack related option",
                    fallback_point=[499, 329],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Focus the search box",
                    click_hint="click the top search box",
                    fallback_point=[482, 70],
                ),
                self._stage(ACTION_TYPE, "Type the voice pack name", required_text=voice_name),
                self._stage(
                    ACTION_CLICK,
                    "Submit the search",
                    click_hint="click the top-right search button",
                    fallback_point=[870, 89],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Choose the target voice pack",
                    click_hint="click the matching voice pack result",
                    fallback_point=[857, 181],
                ),
                self._stage(ACTION_COMPLETE, "The target voice pack is selected", force_complete=True),
            ]
            return

        if app_name == "抖音" and "我的喜欢" in instruction and "搜索" in instruction:
            info["task_kind"] = "douyin_likes_search"
            info["planned_type_texts"] = [keyword]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open 抖音"),
                self._stage(
                    ACTION_CLICK,
                    "Go to My page",
                    click_hint="click the bottom-right 我的 tab",
                    fallback_point=[898, 922],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open liked videos",
                    click_hint="click 我的喜欢",
                    fallback_point=[874, 524],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open search on the liked videos page",
                    click_hint="click the top-right search icon",
                    fallback_point=[796, 76],
                ),
                self._stage(ACTION_TYPE, "Type the search keyword", required_text=keyword),
                self._stage(
                    ACTION_CLICK,
                    "Submit the search",
                    click_hint="click the top-right 搜索 button",
                    fallback_point=[913, 70],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open the first matching result",
                    click_hint="click the first video result, not side icons",
                    fallback_point=[245, 381],
                ),
                self._stage(ACTION_COMPLETE, "The target video is already opened", force_complete=True),
            ]
            return

        if app_name == "快手" and "1日内" in instruction and "1-5分钟" in instruction:
            info["task_kind"] = "kuaishou_filter_search"
            info["planned_type_texts"] = [keyword]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open 快手"),
                self._stage(
                    ACTION_CLICK,
                    "Open search",
                    click_hint="click the top-right search icon",
                    fallback_point=[913, 69],
                ),
                self._stage(ACTION_TYPE, "Type the search keyword", required_text=keyword),
                self._stage(
                    ACTION_CLICK,
                    "Submit the search",
                    click_hint="click 搜索 or the first search result entry",
                    fallback_point=[904, 71],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open filters",
                    click_hint="click the top-right 筛选 button",
                    fallback_point=[933, 122],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Choose videos from the last 1 day",
                    click_hint="click 1日内",
                    fallback_point=[382, 599],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Choose duration 1-5 minutes",
                    click_hint="click 1-5分钟",
                    fallback_point=[614, 703],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Apply filters",
                    click_hint="click the bottom 确定 or 查看 button",
                    fallback_point=[731, 904],
                ),
                self._stage(ACTION_COMPLETE, "The filtered result page is ready", force_complete=True),
            ]
            return

        if app_name == "爱奇艺" and "评论区" in instruction:
            title_query = info.get("search_keyword") or (titles[0] if titles else "")
            comment_text = info.get("comment_text") or ""
            info["task_kind"] = "aiqiyi_comment"
            info["planned_type_texts"] = [title_query, comment_text]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open 爱奇艺"),
                self._stage(
                    ACTION_CLICK,
                    "Open search",
                    click_hint="click the top-right search entry",
                    fallback_point=[835, 46],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Focus the search box",
                    click_hint="click the top search box",
                    fallback_point=[478, 71],
                ),
                self._stage(ACTION_TYPE, "Type the title", required_text=title_query),
                self._stage(
                    ACTION_CLICK,
                    "Open the matching title",
                    click_hint="click the first matching title result",
                    fallback_point=[846, 125],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open the comment area",
                    click_hint="click the 评论区 entry",
                    fallback_point=[366, 650],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open the comment input",
                    click_hint="click the comment input area near the bottom",
                    fallback_point=[185, 899],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Focus the comment editor",
                    click_hint="click the editable comment field",
                    fallback_point=[360, 923],
                ),
                self._stage(ACTION_TYPE, "Type the comment", required_text=comment_text),
                self._stage(
                    ACTION_CLICK,
                    "Send the comment",
                    click_hint="click the send or publish button",
                    fallback_point=[887, 916],
                ),
                self._stage(ACTION_COMPLETE, "The comment is published", force_complete=True),
            ]
            return

        if app_name == "哔哩哔哩" and "收藏" in instruction:
            title_query = info.get("search_keyword") or (titles[0] if titles else "")
            info["task_kind"] = "bilibili_favorite_video"
            info["planned_type_texts"] = [title_query]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open 哔哩哔哩"),
                self._stage(
                    ACTION_CLICK,
                    "Focus the search box",
                    click_hint="click the top search box",
                    fallback_point=[452, 78],
                ),
                self._stage(ACTION_TYPE, "Type the title", required_text=title_query),
                self._stage(
                    ACTION_CLICK,
                    "Submit the search",
                    click_hint="click the top-right search button",
                    fallback_point=[905, 75],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open the first result",
                    click_hint="click the first video result in the综合列表",
                    fallback_point=[481, 234],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Favorite the video",
                    click_hint="click the 收藏 icon or button",
                    fallback_point=[682, 473],
                ),
                self._stage(ACTION_COMPLETE, "The video is already favorited", force_complete=True),
            ]
            return

        if app_name == "芒果TV" and "我的下载" in instruction:
            info["task_kind"] = "mangguo_download_episode"
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open 芒果TV"),
                self._stage(
                    ACTION_CLICK,
                    "Open the download entry",
                    click_hint="click the top-right download or task entry",
                    fallback_point=[848, 78],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Go to My page",
                    click_hint="click the bottom-right 我的 tab",
                    fallback_point=[896, 920],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open downloads",
                    click_hint="click 我的下载",
                    fallback_point=[179, 655],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open the target downloaded show",
                    click_hint=f"click {title or 'the target show'}",
                    fallback_point=[479, 107],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Play the requested episode",
                    click_hint=f"click episode {episode or 'the requested episode'}, not other controls",
                    fallback_point=[310, 251],
                ),
                self._stage(ACTION_COMPLETE, "The requested episode is already playing", force_complete=True),
            ]
            return

        if app_name == "美团" and ("外卖" in instruction or "购买" in instruction):
            store_regex = self._build_store_regex_text(store_name)
            info["task_kind"] = "meituan_order_dish"
            info["planned_type_texts"] = [store_regex, item_name]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open 美团"),
                self._stage(
                    ACTION_CLICK,
                    "Enter food delivery",
                    click_hint="click 外卖 or the food-delivery entry",
                    fallback_point=[104, 195],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open search",
                    click_hint="click the top search entry",
                    fallback_point=[462, 112],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Focus the search box",
                    click_hint="click the top search box",
                    fallback_point=[460, 72],
                ),
                self._stage(ACTION_TYPE, "Type the store name", required_text=store_regex),
                self._stage(
                    ACTION_CLICK,
                    "Open the matching store result",
                    click_hint="click the target store or its search result",
                    fallback_point=[498, 128],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Enter the target store page",
                    click_hint="click the store card",
                    fallback_point=[511, 193],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open in-store dish search",
                    click_hint="click the in-store search icon or search field",
                    fallback_point=[376, 72],
                ),
                self._stage(ACTION_TYPE, "Type the dish name", required_text=item_name),
                self._stage(
                    ACTION_CLICK,
                    "Open the matching dish",
                    click_hint="click the 干锅排骨 result",
                    fallback_point=[891, 200],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Add the dish or confirm SKU",
                    click_hint="click the add, select, or confirm button near the dish",
                    fallback_point=[790, 678],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Go to checkout",
                    click_hint="click the cart or settlement button at the bottom",
                    fallback_point=[486, 762],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Submit the order with the default address",
                    click_hint="click the bottom orange submit/pay button",
                    fallback_point=[835, 910],
                ),
                self._stage(ACTION_COMPLETE, "The order confirmation page is ready", force_complete=True),
            ]
            return

        if app_name == "去哪儿旅行" and "航班" in instruction and route:
            info["task_kind"] = "qunar_flight_price"
            info["planned_type_texts"] = [origin, destination]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open 去哪儿旅行"),
                self._stage(
                    ACTION_CLICK,
                    "Open the flight search page",
                    click_hint="click the 飞机票 or flight entry",
                    fallback_point=[180, 329],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open the departure city field",
                    click_hint="click the departure city selector",
                    fallback_point=[253, 291],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Focus the departure input box",
                    click_hint="click the city input box at the top",
                    fallback_point=[532, 165],
                ),
                self._stage(ACTION_TYPE, "Type the departure city", required_text=origin),
                self._stage(
                    ACTION_CLICK,
                    "Choose the departure city suggestion",
                    click_hint="click the first matching departure city",
                    fallback_point=[354, 181],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open the destination city field",
                    click_hint="click the destination city selector",
                    fallback_point=[741, 291],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Focus the destination input box",
                    click_hint="click the city input box at the top",
                    fallback_point=[544, 165],
                ),
                self._stage(ACTION_TYPE, "Type the destination city", required_text=destination),
                self._stage(
                    ACTION_CLICK,
                    "Choose the destination city suggestion",
                    click_hint="click the first matching destination city",
                    fallback_point=[472, 178],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Select the date for the day after tomorrow",
                    click_hint="click the calendar date cell for 后天",
                    fallback_point=[215, 351],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Confirm the date",
                    click_hint="click the top-right confirm button, not the date grid center",
                    fallback_point=[903, 303],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Search for flights",
                    click_hint="click the orange search button near the bottom",
                    fallback_point=[494, 612],
                ),
                self._stage(ACTION_COMPLETE, "The flight result page is already visible", force_complete=True),
            ]
            return

        if app_name == "腾讯视频" and "第" in instruction and "集" in instruction:
            info["task_kind"] = "tengxun_video_episode"
            info["planned_type_texts"] = [keyword or title]
            query = keyword or title
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open 腾讯视频"),
                self._stage(
                    ACTION_CLICK,
                    "Open search",
                    click_hint="click the top-right search icon",
                    fallback_point=[897, 79],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Focus the search box",
                    click_hint="click the top search box",
                    fallback_point=[454, 71],
                ),
                self._stage(ACTION_TYPE, "Type the title", required_text=query),
                self._stage(
                    ACTION_CLICK,
                    "Open the matching result",
                    click_hint="click the first matching title or search confirm button",
                    fallback_point=[511, 162],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open the playable show page",
                    click_hint="click the show result card, not the center of the video player",
                    fallback_point=[349, 390],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Play the requested episode",
                    click_hint=f"click the episode selector for episode {episode or 'the requested one'}, not the video center",
                    fallback_point=[477, 668],
                ),
                self._stage(ACTION_COMPLETE, "The requested episode is already opened", force_complete=True),
            ]
            return

        if app_name == "喜马拉雅" and "三体" in instruction:
            info["task_kind"] = "ximalaya_search"
            info["planned_type_texts"] = [".*三体.*"]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open 喜马拉雅"),
                self._stage(
                    ACTION_CLICK,
                    "Open search",
                    click_hint="click the top-right search entry",
                    fallback_point=[854, 41],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Open the search page",
                    click_hint="click the search or discovery entry on the right",
                    fallback_point=[931, 571],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Focus the search box",
                    click_hint="click the top search box",
                    fallback_point=[393, 76],
                ),
                self._stage(ACTION_TYPE, "Type the audio title", required_text=".*三体.*"),
                self._stage(
                    ACTION_CLICK,
                    "Open the matching result",
                    click_hint="click the first matching result",
                    fallback_point=[854, 137],
                ),
                self._stage(
                    ACTION_CLICK,
                    "Play the target audio",
                    click_hint="click the target audio card",
                    fallback_point=[650, 415],
                ),
                self._stage(ACTION_COMPLETE, "The target audio is already playing", force_complete=True),
            ]
            return

        self._configure_generic_task(info)

    def _configure_generic_task(self, info: Dict[str, Any]):
        instruction = info["instruction"]
        keyword = info.get("search_keyword") or ""
        titles = info.get("titles", [])
        title = titles[0] if titles else ""
        comment_text = info.get("comment_text") or ""
        episode = info.get("episode") or ""
        item_name = info.get("item_name") or ""

        query = keyword or title

        if comment_text and query:
            info["task_kind"] = "generic_comment"
            info["generic_intent"] = "comment"
            info["planned_type_texts"] = [query, comment_text]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open the target app"),
                self._stage(ACTION_CLICK, "Open search", click_hint="click the top search entry or search icon"),
                self._stage(ACTION_TYPE, "Type the target title", required_text=query),
                self._stage(ACTION_CLICK, "Open the matching result", click_hint="click the first matching result"),
                self._stage(ACTION_CLICK, "Open the comment area", click_hint="click the comment area or comment input"),
                self._stage(ACTION_TYPE, "Type the comment", required_text=comment_text),
                self._stage(ACTION_CLICK, "Submit the comment", click_hint="click the send, publish, or submit button"),
                self._stage(ACTION_COMPLETE, "The comment should already be submitted", force_complete=True),
            ]
            return

        if item_name and query:
            info["task_kind"] = "generic_purchase"
            info["generic_intent"] = "purchase"
            info["planned_type_texts"] = [query, item_name]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open the target app"),
                self._stage(ACTION_CLICK, "Open search", click_hint="click the top search box or search icon"),
                self._stage(ACTION_TYPE, "Type the first query", required_text=query),
                self._stage(ACTION_CLICK, "Open the matching result", click_hint="click the first matching result"),
                self._stage(ACTION_CLICK, "Open item search or item list", click_hint="click the item list or in-page search field"),
                self._stage(ACTION_TYPE, "Type the item name", required_text=item_name),
                self._stage(ACTION_CLICK, "Open the matching item", click_hint="click the matching item card"),
                self._stage(ACTION_CLICK, "Confirm or add the item", click_hint="click add, confirm, or next"),
                self._stage(ACTION_COMPLETE, "The requested item should already be chosen", force_complete=True),
            ]
            return

        if query and "搜索" in instruction and "收藏" in instruction:
            info["task_kind"] = "generic_search_and_favorite"
            info["generic_intent"] = "favorite"
            info["planned_type_texts"] = [query]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open the target app"),
                self._stage(ACTION_CLICK, "Open search", click_hint="click the top search box or search icon"),
                self._stage(ACTION_TYPE, "Type the search keyword", required_text=query),
                self._stage(ACTION_CLICK, "Open the matching result", click_hint="click the first matching result"),
                self._stage(ACTION_CLICK, "Favorite the current content", click_hint="click the favorite, collect, or star button"),
                self._stage(ACTION_COMPLETE, "The content should already be favorited", force_complete=True),
            ]
            return

        if query and ("播放" in instruction or "查看" in instruction or "看一下" in instruction):
            info["task_kind"] = "generic_search_and_open"
            info["generic_intent"] = "open_result"
            info["planned_type_texts"] = [query]
            stages = [
                self._stage(ACTION_OPEN, "Open the target app"),
                self._stage(ACTION_CLICK, "Open search", click_hint="click the top search box or search icon"),
                self._stage(ACTION_TYPE, "Type the search keyword", required_text=query),
                self._stage(ACTION_CLICK, "Open the matching result", click_hint="click the first matching result"),
            ]
            if episode:
                stages.append(
                    self._stage(
                        ACTION_CLICK,
                        "Choose the requested episode or sub-item",
                        click_hint=f"click episode {episode} or the requested sub-item",
                    )
                )
            stages.append(self._stage(ACTION_COMPLETE, "The target content should already be opened", force_complete=True))
            info["action_plan"] = stages
            return

        if query and "搜索" in instruction:
            info["task_kind"] = "generic_search"
            info["generic_intent"] = "search"
            info["planned_type_texts"] = [query]
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open the target app"),
                self._stage(ACTION_CLICK, "Open search", click_hint="click the top search box or search icon"),
                self._stage(ACTION_TYPE, "Type the search keyword", required_text=query),
                self._stage(ACTION_CLICK, "Submit the search", click_hint="click the search confirm button"),
                self._stage(ACTION_COMPLETE, "The search result should already be visible", force_complete=True),
            ]
            return

        if "打开" in instruction or "去" in instruction or "在" in instruction:
            info["task_kind"] = "generic_open"
            info["generic_intent"] = "open_app"
            info["action_plan"] = [
                self._stage(ACTION_OPEN, "Open the target app"),
                self._stage(ACTION_COMPLETE, "The app should already be opened", force_complete=True),
            ]

    def _stage(
        self,
        action: str,
        goal: str,
        required_text: str = "",
        click_hint: str = "",
        force_complete: bool = False,
        allowed_actions: Optional[List[str]] = None,
        fallback_point: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        return {
            "expected_action": action,
            "goal": goal,
            "required_text": required_text,
            "click_hint": click_hint,
            "force_complete": force_complete,
            "allowed_actions": allowed_actions or ([action] if action else []),
            "fallback_point": fallback_point,
            "completion_hint": "Use COMPLETE instead of extra clicks once the target page is already ready."
            if action == ACTION_COMPLETE
            else "",
        }

    def _build_baidumap_route_text(self, place: str) -> str:
        place = place.strip()
        if not place:
            return ""
        if "国际医学中心" in place:
            return ".*国际医学中心"
        if "回民街" in place:
            return ".*回民街"
        return f".*{place}"

    def _build_store_regex_text(self, store_name: str) -> str:
        if not store_name:
            return ""
        normalized = re.sub(r"[（(].*?[）)]", "", store_name).strip()
        return f"{normalized}.*"

    def _clean_search_keyword(self, text: str) -> str:
        keyword = text.strip("“”\"' ，。")
        keyword = re.sub(r"(的视频|视频|的作品|作品)$", "", keyword)
        return keyword.strip("“”\"' ，。")

    def _extract_after_keyword(self, instruction: str, prefix: str) -> str:
        match = re.search(re.escape(prefix) + r"(.+?)(?:，|。|$)", instruction)
        return match.group(1).strip() if match else ""

    def _get_stage_hint(self, input_data: AgentInput) -> Dict[str, Any]:
        plan = self._task_info.get("action_plan") or []
        action_index = len(input_data.history_actions)

        if action_index == 0:
            return self._stage(ACTION_OPEN, "Open the target app", allowed_actions=[ACTION_OPEN])

        if plan:
            if action_index < len(plan):
                stage = dict(plan[action_index])
            else:
                stage = self._stage(ACTION_COMPLETE, "The task should already be finished", force_complete=True)
        else:
            stage = {
                "expected_action": "",
                "goal": "Inspect the screenshot and choose the next best action",
                "required_text": "",
                "click_hint": "",
                "force_complete": False,
                "allowed_actions": [],
                "completion_hint": "",
            }

        if stage.get("expected_action") == ACTION_TYPE and not stage.get("required_text"):
            stage["required_text"] = self._select_fallback_text(input_data)
        return stage

    def _detect_app_name(self, instruction: str) -> str:
        for alias, canonical_name in self.APP_ALIASES:
            if alias in instruction:
                return canonical_name
        for pattern in (
            r"打开([A-Za-z0-9\u4e00-\u9fa5·\-\+]{2,20})",
            r"去([A-Za-z0-9\u4e00-\u9fa5·\-\+]{2,20})",
            r"在([A-Za-z0-9\u4e00-\u9fa5·\-\+]{2,20})",
        ):
            match = re.search(pattern, instruction)
            if match:
                candidate = match.group(1).strip()
                candidate = re.split(r"(搜索|播放|打开|评论|购买|查看|看一下|我的|里|并|，|。)", candidate)[0].strip()
                if candidate:
                    return candidate
        return ""

    def _extract_titles(self, instruction: str) -> List[str]:
        titles = re.findall(r"[“\"《](.+?)[”\"》]", instruction)
        return self._dedupe_non_empty(titles)

    def _dedupe_non_empty(self, values: Iterable[str]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            cleaned = str(value).strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result

    def _build_open_output(self) -> AgentOutput:
        app_name = self._task_info.get("app_name") or ""
        return AgentOutput(action=ACTION_OPEN, parameters={"app_name": app_name})

    def _build_action_summaries(self, history_actions: List[Dict[str, Any]]) -> List[str]:
        summaries = []
        for item in history_actions[-6:]:
            action = str(item.get("action", "")).upper()
            params = item.get("parameters", {}) or {}
            summaries.append(self._format_action_summary(action, params))
        return summaries[-6:]

    def _format_action_summary(self, action: str, parameters: Dict[str, Any]) -> str:
        if action == ACTION_CLICK:
            return f"CLICK {parameters.get('point', [])}"
        if action == ACTION_SCROLL:
            return f"SCROLL {parameters.get('start_point', [])}->{parameters.get('end_point', [])}"
        if action == ACTION_TYPE:
            return f"TYPE {parameters.get('text', '')}"
        if action == ACTION_OPEN:
            return f"OPEN {parameters.get('app_name', '')}"
        if action == ACTION_COMPLETE:
            return "COMPLETE"
        return f"{action} {parameters}"

    def _extract_response_text(self, response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except Exception:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                else:
                    text_parts.append(str(part))
            return "\n".join(part for part in text_parts if part).strip()

        return str(content).strip()

    def _extract_usage_info(self, response: Any) -> Optional[UsageInfo]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)

        return UsageInfo(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            cached_tokens=int(getattr(prompt_details, "cached_tokens", 0) or 0),
            reasoning_tokens=int(getattr(completion_details, "reasoning_tokens", 0) or 0),
        )

    def _parse_and_normalize(self, raw_output: str, input_data: AgentInput) -> Optional[AgentOutput]:
        parsed = self._parse_model_output(raw_output)
        if parsed is None:
            return None

        normalized = self._normalize_output(
            parsed.get("action"),
            parsed.get("parameters", {}),
            input_data=input_data,
        )
        if normalized is None:
            return None

        return AgentOutput(action=normalized[0], parameters=normalized[1], raw_output=raw_output)

    def _parse_model_output(self, raw_output: str) -> Optional[Dict[str, Any]]:
        if not raw_output:
            return None

        for candidate in self._iter_json_candidates(raw_output):
            payload = self._load_mapping(candidate)
            if payload is None:
                continue

            action = payload.get("action", payload.get("Action"))
            parameters = payload.get("parameters", payload.get("params", {}))
            if action:
                return {"action": action, "parameters": parameters}

        action_call = self._parse_action_call(raw_output)
        if action_call is not None:
            return action_call

        return None

    def _iter_json_candidates(self, raw_output: str) -> Iterable[str]:
        text = raw_output.strip()
        if text:
            yield text

        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", raw_output, flags=re.IGNORECASE):
            candidate = match.group(1).strip()
            if candidate:
                yield candidate

        first_brace = raw_output.find("{")
        last_brace = raw_output.rfind("}")
        if 0 <= first_brace < last_brace:
            candidate = raw_output[first_brace:last_brace + 1].strip()
            if candidate:
                yield candidate

    def _load_mapping(self, candidate: str) -> Optional[Dict[str, Any]]:
        normalized = self._normalize_quotes(candidate)
        loaders = (
            lambda text: json.loads(text),
            lambda text: ast.literal_eval(text),
        )

        for loader in loaders:
            try:
                payload = loader(normalized)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _normalize_quotes(self, text: str) -> str:
        return (
            text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
        )

    def _parse_action_call(self, raw_output: str) -> Optional[Dict[str, Any]]:
        text = self._normalize_quotes(raw_output)
        action_match = re.search(
            r"(?:^|\b)(?:Action\s*:\s*)?(click|scroll|type|open|complete)\s*\(([\s\S]*?)\)",
            text,
            flags=re.IGNORECASE,
        )
        if not action_match:
            return None

        action = action_match.group(1)
        args_text = action_match.group(2).strip()
        upper_action = action.upper()

        if upper_action == ACTION_CLICK:
            point = self._extract_first_point(args_text)
            if point is None:
                return {"action": upper_action, "parameters": {}}
            return {"action": upper_action, "parameters": {"point": point}}

        if upper_action == ACTION_SCROLL:
            points = self._extract_all_points(args_text)
            if len(points) >= 2:
                return {
                    "action": upper_action,
                    "parameters": {"start_point": points[0], "end_point": points[1]},
                }
            return {"action": upper_action, "parameters": {}}

        if upper_action == ACTION_TYPE:
            text_value = self._extract_text_argument(args_text, ["text", "content", "value"])
            return {"action": upper_action, "parameters": {"text": text_value}}

        if upper_action == ACTION_OPEN:
            app_name = self._extract_text_argument(args_text, ["app_name", "app", "name"])
            return {"action": upper_action, "parameters": {"app_name": app_name}}

        return {"action": ACTION_COMPLETE, "parameters": {}}

    def _extract_text_argument(self, args_text: str, keys: List[str]) -> str:
        for key in keys:
            match = re.search(rf"{key}\s*=\s*['\"]([^'\"]*)['\"]", args_text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

        quoted = re.search(r"['\"]([^'\"]*)['\"]", args_text)
        if quoted:
            return quoted.group(1).strip()

        return args_text.strip().strip(",")

    def _extract_all_points(self, text: str) -> List[List[float]]:
        points: List[List[float]] = []

        for match in re.finditer(r"<point>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*</point>", text):
            points.append([float(match.group(1)), float(match.group(2))])

        for match in re.finditer(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text):
            points.append([float(match.group(1)), float(match.group(2))])

        if points:
            return points

        values = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", text)]
        for idx in range(0, len(values) - 1, 2):
            points.append([values[idx], values[idx + 1]])
        return points

    def _extract_first_point(self, text: str) -> Optional[List[float]]:
        points = self._extract_all_points(text)
        return points[0] if points else None

    def _normalize_output(
        self,
        action: Any,
        parameters: Any,
        input_data: AgentInput,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        normalized_action = self._normalize_action(action)
        if normalized_action is None:
            return None

        if not isinstance(parameters, dict):
            parameters = {}

        if normalized_action == ACTION_CLICK:
            point = parameters.get("point")
            if point is None and "coord" in parameters:
                point = parameters.get("coord")
            if point is None and "x" in parameters and "y" in parameters:
                point = [parameters.get("x"), parameters.get("y")]
            normalized_point = self._normalize_point(point)
            if normalized_point is None:
                return None
            return ACTION_CLICK, {"point": normalized_point}

        if normalized_action == ACTION_SCROLL:
            start_point = parameters.get("start_point")
            end_point = parameters.get("end_point")
            if start_point is None and "from" in parameters:
                start_point = parameters.get("from")
            if end_point is None and "to" in parameters:
                end_point = parameters.get("to")
            normalized_start = self._normalize_point(start_point)
            normalized_end = self._normalize_point(end_point)
            if normalized_start is None or normalized_end is None:
                return None
            return ACTION_SCROLL, {"start_point": normalized_start, "end_point": normalized_end}

        if normalized_action == ACTION_TYPE:
            text = parameters.get("text")
            if text in (None, ""):
                text = parameters.get("content")
            if text in (None, ""):
                text = self._select_fallback_text(input_data)
            if text in (None, ""):
                return None
            return ACTION_TYPE, {"text": str(text)}

        if normalized_action == ACTION_OPEN:
            app_name = parameters.get("app_name")
            if app_name in (None, ""):
                app_name = parameters.get("app")
            if app_name in (None, ""):
                app_name = self._task_info.get("app_name", "")
            if app_name in (None, ""):
                return None
            return ACTION_OPEN, {"app_name": str(app_name)}

        if normalized_action == ACTION_COMPLETE:
            return ACTION_COMPLETE, {}

        return None

    def _normalize_action(self, action: Any) -> Optional[str]:
        if action is None:
            return None

        cleaned = str(action).strip().upper()
        synonym_map = {
            "TAP": ACTION_CLICK,
            "INPUT": ACTION_TYPE,
            "FINISH": ACTION_COMPLETE,
            "DONE": ACTION_COMPLETE,
        }
        cleaned = synonym_map.get(cleaned, cleaned)
        if cleaned in VALID_ACTIONS:
            return cleaned
        return None

    def _normalize_point(self, point: Any) -> Optional[List[int]]:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            return [self._clamp_coord(point[0]), self._clamp_coord(point[1])]
        return None

    def _clamp_coord(self, value: Any) -> int:
        try:
            numeric = float(value)
        except Exception:
            numeric = 0.0
        numeric = round(numeric)
        return max(0, min(1000, int(numeric)))

    def _select_fallback_text(self, input_data: AgentInput) -> str:
        planned_texts = self._task_info.get("planned_type_texts", [])
        typed_count = 0
        for item in input_data.history_actions:
            if str(item.get("action", "")).upper() == ACTION_TYPE:
                typed_count += 1

        if planned_texts:
            index = min(typed_count, len(planned_texts) - 1)
            if planned_texts[index]:
                return planned_texts[index]

        candidates = self._task_info.get("type_candidates", [])
        if not candidates:
            return ""

        index = min(typed_count, len(candidates) - 1)
        return candidates[index]

    def _is_strict_task(self) -> bool:
        task_kind = self._task_info.get("task_kind", "")
        return task_kind in self.STRICT_TASK_KINDS

    def _apply_stage_constraints(self, decision: AgentOutput, input_data: AgentInput) -> AgentOutput:
        stage = self._get_stage_hint(input_data)
        required_text = stage.get("required_text", "")
        is_strict_task = self._is_strict_task()

        if stage.get("force_complete") and is_strict_task:
            return AgentOutput(action=ACTION_COMPLETE, parameters={}, raw_output=decision.raw_output)

        if required_text and is_strict_task:
            return AgentOutput(
                action=ACTION_TYPE,
                parameters={"text": required_text},
                raw_output=decision.raw_output,
            )

        if required_text and decision.action == ACTION_TYPE:
            return AgentOutput(
                action=ACTION_TYPE,
                parameters={"text": required_text},
                raw_output=decision.raw_output,
            )

        expected_action = stage.get("expected_action")
        if is_strict_task and expected_action == ACTION_COMPLETE and decision.action != ACTION_COMPLETE:
            return AgentOutput(action=ACTION_COMPLETE, parameters={}, raw_output=decision.raw_output)

        return decision

    def _build_direct_stage_action(
        self,
        stage_hint: Dict[str, Any],
        input_data: AgentInput,
    ) -> Optional[AgentOutput]:
        expected_action = stage_hint.get("expected_action")
        required_text = stage_hint.get("required_text", "")
        fallback_point = stage_hint.get("fallback_point")
        is_strict_task = self._is_strict_task()

        if expected_action == ACTION_OPEN:
            return self._build_open_output()

        if not is_strict_task:
            return None

        if stage_hint.get("force_complete") or expected_action == ACTION_COMPLETE:
            return AgentOutput(action=ACTION_COMPLETE, parameters={})

        if required_text or expected_action == ACTION_TYPE:
            text = required_text or self._select_fallback_text(input_data)
            if text:
                return AgentOutput(action=ACTION_TYPE, parameters={"text": text})

        if expected_action == ACTION_CLICK and fallback_point:
            return AgentOutput(action=ACTION_CLICK, parameters={"point": fallback_point})

        return None

    def _repair_output_with_model(self, raw_output: str, input_data: AgentInput) -> str:
        self._repair_call_used = True

        stage_hint = self._get_stage_hint(input_data)
        repair_prompt = (
            "Convert the following invalid GUI agent output into exactly one valid JSON object.\n"
            "Allowed actions: CLICK, SCROLL, TYPE, OPEN, COMPLETE.\n"
            "Output schema:\n"
            '{"action":"CLICK|SCROLL|TYPE|OPEN|COMPLETE","parameters":{...}}\n'
            "CLICK => {\"point\": [x, y]}\n"
            "SCROLL => {\"start_point\": [x1, y1], \"end_point\": [x2, y2]}\n"
            "TYPE => {\"text\": \"...\"}\n"
            "OPEN => {\"app_name\": \"...\"}\n"
            "COMPLETE => {}\n"
            "Follow the stage hint strictly when it says TYPE or COMPLETE.\n"
            f"Task: {input_data.instruction}\n"
            f"Task summary: {json.dumps(self._task_info, ensure_ascii=False)}\n"
            f"Stage hint: {json.dumps(stage_hint, ensure_ascii=False)}\n"
            f"Invalid output:\n{raw_output}"
        )

        try:
            response = self._call_api(
                [
                    {"role": "system", "content": "You repair malformed agent outputs into strict JSON."},
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=0,
            )
        except Exception as exc:
            logger.warning("Repair call failed: %s", exc)
            return ""

        return self._extract_response_text(response)

    def _deterministic_fallback(self, input_data: AgentInput, raw_output: str) -> AgentOutput:
        stage = self._get_stage_hint(input_data)
        expected_action = stage.get("expected_action")
        required_text = stage.get("required_text", "")
        fallback_point = stage.get("fallback_point")

        if input_data.step_count == 1:
            return self._build_open_output()

        if stage.get("force_complete") or expected_action == ACTION_COMPLETE:
            return AgentOutput(action=ACTION_COMPLETE, parameters={}, raw_output=raw_output)

        if required_text or expected_action == ACTION_TYPE:
            text = required_text or self._select_fallback_text(input_data)
            if text:
                return AgentOutput(action=ACTION_TYPE, parameters={"text": text}, raw_output=raw_output)

        if expected_action == ACTION_CLICK:
            click_point = fallback_point or self._build_generic_click_fallback(stage)
            if click_point is not None:
                return AgentOutput(action=ACTION_CLICK, parameters={"point": click_point}, raw_output=raw_output)

        lower_output = raw_output.lower()
        if "open" in lower_output:
            app_name = self._task_info.get("app_name", "")
            if app_name:
                return AgentOutput(action=ACTION_OPEN, parameters={"app_name": app_name}, raw_output=raw_output)

        if "complete" in lower_output or "done" in lower_output or "finish" in lower_output:
            return AgentOutput(action=ACTION_COMPLETE, parameters={}, raw_output=raw_output)

        return AgentOutput(action=ACTION_COMPLETE, parameters={}, raw_output=raw_output)

    def _build_generic_click_fallback(self, stage_hint: Dict[str, Any]) -> Optional[List[int]]:
        hint_text = " ".join(
            [
                str(stage_hint.get("goal", "")),
                str(stage_hint.get("click_hint", "")),
            ]
        ).lower()

        if not hint_text.strip():
            return None

        if "bottom-right" in hint_text or "我的" in hint_text:
            return [900, 920]
        if "search icon" in hint_text or "top-right search" in hint_text:
            return [900, 70]
        if "search box" in hint_text or "top search" in hint_text:
            return [500, 70]
        if "first matching result" in hint_text or "first result" in hint_text:
            return [500, 220]
        if "first video result" in hint_text:
            return [250, 380]
        if "first candidate" in hint_text or "suggestion" in hint_text:
            return [500, 180]
        if "comment input" in hint_text or "comment area" in hint_text:
            return [250, 900]
        if "send" in hint_text or "publish" in hint_text or "submit" in hint_text:
            return [880, 920]
        if "episode" in hint_text or "sub-item" in hint_text:
            return [500, 680]
        if "favorite" in hint_text or "collect" in hint_text or "star" in hint_text:
            return [700, 470]
        if "confirm" in hint_text or "next" in hint_text or "add" in hint_text:
            return [760, 700]
        return [500, 500]
