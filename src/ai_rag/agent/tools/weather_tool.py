# src/ai_rag/agent/tools/weather_tool.py
"""天气查询工具：通过免费 wttr.in 服务获取实时天气（无需 API Key）"""
import logging
import urllib.parse

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 常见中文城市名 → 英文（wttr.in 对中文名解析不稳定，优先用英文名）
_CITY_MAP = {
    "北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou", "深圳": "Shenzhen",
    "杭州": "Hangzhou", "成都": "Chengdu", "武汉": "Wuhan", "南京": "Nanjing",
    "重庆": "Chongqing", "西安": "Xian", "天津": "Tianjin", "苏州": "Suzhou",
    "长沙": "Changsha", "青岛": "Qingdao", "厦门": "Xiamen", "郑州": "Zhengzhou",
    "沈阳": "Shenyang", "大连": "Dalian", "昆明": "Kunming", "哈尔滨": "Harbin",
    "济南": "Jinan", "福州": "Fuzhou", "合肥": "Hefei", "宁波": "Ningbo",
    "无锡": "Wuxi", "东莞": "Dongguan", "佛山": "Foshan", "石家庄": "Shijiazhuang",
    "香港": "Hong Kong", "台北": "Taipei",
}

_WEATHER_CLIENT = httpx.AsyncClient(timeout=20.0, follow_redirects=True)


@tool("get_weather")
async def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气：天气现象、气温、体感温度、湿度、风力、云量、今日最高/最低温。
    当用户询问天气、气温、会不会下雨、风力、空气质量等问题时，必须调用此工具。
    参数 city 为城市名，例如"上海"、"北京"、"杭州"。
    """
    city = (city or "").strip()
    if not city:
        return "查询失败：缺少城市名称，请提供要查询天气的城市。"

    target = _CITY_MAP.get(city, city)
    url = f"https://wttr.in/{urllib.parse.quote(target)}?format=j1&lang=zh"
    try:
        resp = await _WEATHER_CLIENT.get(url, headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("[Tool:get_weather] 请求失败 city=%s error=%s", city, e)
        return "查询失败：天气服务暂时不可用，请稍后重试。"

    try:
        cc = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0].get("areaName", [{}])[0].get("value", city)
        desc = (cc.get("weatherDesc", [{}])[0].get("value") or "未知").strip()
        temp = cc.get("temp_C", "?")
        feels = cc.get("FeelsLikeC", "?")
        hum = cc.get("humidity", "?")
        wind_dir = cc.get("winddir16Point", "")
        wind_km = cc.get("windspeedKmph", "?")
        cloud = cc.get("cloudcover", "?")
        precip = cc.get("precipMM", "?")
        obs_time = cc.get("localObsDateTime", "")
        today = data.get("weather", [{}])[0]
        tmax = today.get("maxtempC", "?")
        tmin = today.get("mintempC", "?")
        lines = [
            f"{area}当前天气：{desc}，气温 {temp}°C（体感 {feels}°C），湿度 {hum}%，",
            f"{wind_dir}风 {wind_km}km/h，云量 {cloud}%，降水量 {precip}mm。",
            f"今日预报：{tmin}°C ~ {tmax}°C。",
        ]
        if obs_time:
            lines.append(f"数据更新时间：{obs_time}。")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("[Tool:get_weather] 解析失败 city=%s error=%s", city, e)
        return "查询失败：天气数据解析异常，请稍后重试。"