"""
JD-Agent 示例技能：天气查询
==============================
提供一个天气查询的能力包。

此文件 + SKILL.md 构成了一个完整的 Skill。
"""

from agent.skills.base import Skill, SkillAbility
from agent.tools.base import Tool


class GetWeatherTool(Tool):
    """获取指定城市的天气信息"""

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "获取指定城市的实时天气信息，包含天气状况、气温、湿度等"

    @property
    def parameters(self) -> dict:
        return {
            "city": {
                "type": "string",
                "description": "城市名称，如：北京、上海、广州",
                "required": True,
            },
        }

    async def execute(self, city: str, **kwargs) -> str:
        import random
        weathers = ["晴朗", "多云", "小雨", "阴天", "晴转多云", "阵雨", "大风"]
        temps = {"北京": 26, "上海": 28, "广州": 31, "深圳": 30, "杭州": 27, "成都": 25, "武汉": 29}
        temp = temps.get(city, random.randint(18, 34))
        weather = random.choice(weathers)
        humidity = random.randint(40, 85)
        wind = random.choice(["1级", "2级", "3级", "4级"])
        return (
            f"🌆 {city} 今日天气\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"☀️ 天气: {weather}\n"
            f"🌡️  气温: {temp}°C\n"
            f"💧 湿度: {humidity}%\n"
            f"🌬️  风力: {wind}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"建议：{'适宜户外活动' if weather in ('晴朗', '多云') else '出门请带伞'}"
        )


class GetForecastTool(Tool):
    """获取未来几天的天气预报"""

    @property
    def name(self) -> str:
        return "get_forecast"

    @property
    def description(self) -> str:
        return "获取指定城市未来3-7天的天气预报"

    @property
    def parameters(self) -> dict:
        return {
            "city": {
                "type": "string",
                "description": "城市名称",
                "required": True,
            },
            "days": {
                "type": "integer",
                "description": "预报天数（1-7），默认3",
                "required": False,
            },
        }

    async def execute(self, city: str, days: int = 3, **kwargs) -> str:
        import random
        from datetime import datetime, timedelta

        lines = [f"📅 {city} 未来 {days} 天天气预报\n"]
        for i in range(min(days, 7)):
            d = (datetime.now() + timedelta(days=i + 1)).strftime("%m-%d")
            w = random.choice(["☀️ 晴", "⛅ 多云", "🌧️ 雨", "🌦️ 阵雨"])
            t_high = random.randint(22, 35)
            t_low = t_high - random.randint(5, 12)
            lines.append(f"  {d} {w} {t_low}~{t_high}°C")
        return "\n".join(lines)


class GetAirQualityTool(Tool):
    """获取空气质量信息"""

    @property
    def name(self) -> str:
        return "get_air_quality"

    @property
    def description(self) -> str:
        return "获取指定城市的空气质量指数（AQI）和污染等级"

    @property
    def parameters(self) -> dict:
        return {
            "city": {
                "type": "string",
                "description": "城市名称",
                "required": True,
            },
        }

    async def execute(self, city: str, **kwargs) -> str:
        import random
        aqi = random.randint(20, 180)
        if aqi <= 50:
            level, advice = "优 ✅", "空气质量令人满意，可正常外出"
        elif aqi <= 100:
            level, advice = "良 👍", "空气质量可接受，敏感人群注意防护"
        elif aqi <= 150:
            level, advice = "轻度污染 ⚠️", "敏感人群减少户外活动"
        else:
            level, advice = "中度污染 ❌", "建议减少外出，佩戴口罩"
        return (
            f"🏭 {city} 空气质量\n"
            f"  AQI: {aqi}\n"
            f"  等级: {level}\n"
            f"  建议: {advice}"
        )


class WeatherSkill(Skill):
    """
    天气技能：完整的气象信息能力包。
    
    包含三个 Tool：
      - get_weather:    实时天气
      - get_forecast:   天气预报
      - get_air_quality:空气质量
    """

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "天气查询技能：获取实时天气、天气预报和空气质量信息"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def categories(self) -> list[str]:
        return ["信息获取", "生活服务"]

    @property
    def abilities(self) -> list[SkillAbility]:
        return [
            SkillAbility(
                name="get_weather",
                description="获取指定城市的实时天气",
                keywords=["天气", "气温", "温度", "多少度", "冷", "热", "暖和"],
                examples=[
                    "今天天气怎么样？",
                    "上海多少度？",
                    "北京今天冷不冷？",
                ],
                tool=GetWeatherTool(),
            ),
            SkillAbility(
                name="get_forecast",
                description="获取未来几天的天气预报",
                keywords=["预报", "未来", "明天", "后天", "这周", "周末", "下周"],
                examples=[
                    "明天北京会下雨吗？",
                    "这周末适合出去玩吗？",
                    "未来三天的天气预报",
                ],
                tool=GetForecastTool(),
            ),
            SkillAbility(
                name="get_air_quality",
                description="获取空气质量指数",
                keywords=["空气质量", "AQI", "雾霾", "污染", "pm2.5", "pm10"],
                examples=[
                    "北京的空气质量怎么样？",
                    "今天有雾霾吗？",
                ],
                tool=GetAirQualityTool(),
            ),
        ]


# ─── 导出实例（SkillLoader 自动发现需要）───
skill_instance = WeatherSkill()
