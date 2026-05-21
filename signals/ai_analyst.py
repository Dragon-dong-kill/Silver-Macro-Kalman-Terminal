
import json
from datetime import datetime

import pandas as pd
import streamlit as st


AI_PROMPT_TEMPLATE = (
    "你是一位专业白银期货基本面分析师。请根据以下最新市场数据，"
    "输出一份JSON格式的基本面研判。\n\n"
    "=== 当前价格数据 ===\n"
    "SI=F 白银期货最新报价: {silver_price} 美元/盎司\n"
    "美元指数 DXY: {dxy}\n"
    "十年期美债收益率 TNX: {tnx}%\n"
    "实际利率 DFII10: {real_rate}%\n"
    "盈亏平衡通胀率 T10YIE: {breakeven}%\n"
    "VIX 恐慌指数: {vix}\n"
    "黄金 GC=F: {gold} 美元/盎司\n"
    "金银比: {gold_silver_ratio:.1f}\n"
    "铜 HG=F: {copper} 美元/磅\n"
    "标普500 GSPC: {sp500}\n\n"
    "=== 近期新闻标题 ===\n"
    "{news_titles}\n\n"
    "=== 技术面参考 ===\n"
    "卡尔曼滤波价格: {kalman_price}\n"
    "隐藏动量: {velocity:.3f}\n"
    "ADX趋势强度: {adx:.1f}\n"
    "宏观因子Z-Score - 情绪:{z_sentiment:.2f} 美元:{z_dxy:.2f} 利率:{z_rate:.2f}\n\n"
    "请严格输出以下JSON(不要多余文字):\n"
    '{{"direction": "bullish|neutral|bearish",'
    '"confidence": 0.0-1.0,'
    '"score": -100到100的整数,'
    '"key_drivers": "关键驱动因素简述(中文,50字内)",'
    '"risk_factors": "主要风险因素(中文,50字内)",'
    '"short_term_view": "短期1-3天展望(中文,30字内)",'
    '"macro_assessment": "宏观环境评估(中文,40字内)"}}'
)


def _build_analysis_prompt(
    macro: pd.DataFrame,
    filtered: pd.DataFrame,
    news_titles: str,
) -> str:
    latest_macro = macro.iloc[-1]
    latest_filtered = filtered.iloc[-1]

    silver_price = latest_filtered.get("observed", "N/A")
    kalman_price = latest_filtered.get("kalman_price", "N/A")
    velocity = float(latest_filtered.get("velocity", 0))
    adx = float(latest_filtered.get("adx", 0)) if pd.notna(latest_filtered.get("adx")) else 0

    dxy = latest_macro.get("dxy", "N/A")
    tnx = latest_macro.get("tnx", "N/A")
    real_rate = latest_macro.get("real_rate", "N/A")
    breakeven = latest_macro.get("breakeven", "N/A")
    vix = latest_macro.get("vix", "N/A")
    gold = latest_macro.get("gold", "N/A")
    copper = latest_macro.get("copper", "N/A")
    sp500 = latest_macro.get("sp500", "N/A")

    gold_silver_ratio = 0.0
    if (
        pd.notna(latest_macro.get("gold", None))
        and pd.notna(silver_price)
        and isinstance(silver_price, (int, float))
        and float(silver_price) > 0
    ):
        gold_silver_ratio = float(latest_macro["gold"]) / float(silver_price)

    z_sentiment = float(latest_macro.get("z_sentiment", 0))
    z_dxy = float(latest_macro.get("z_dxy", 0))
    z_rate = float(latest_macro.get("z_rate", 0))

    return AI_PROMPT_TEMPLATE.format(
        silver_price=silver_price,
        dxy=dxy,
        tnx=tnx,
        real_rate=real_rate,
        breakeven=breakeven,
        vix=vix,
        gold=gold,
        gold_silver_ratio=gold_silver_ratio,
        copper=copper,
        sp500=sp500,
        news_titles=news_titles[:2000],
        kalman_price=kalman_price,
        velocity=velocity,
        adx=adx,
        z_sentiment=z_sentiment,
        z_dxy=z_dxy,
        z_rate=z_rate,
    )


def _call_deepseek(prompt: str, api_key: str) -> dict | None:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的白银期货基本面分析师，只输出JSON。当前日期: " + datetime.now().strftime("%Y-%m-%d")},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception:
        return None


def run_ai_analysis(macro, filtered, news, api_key):
    if not api_key or api_key.strip() == "":
        return _build_fallback_analysis(macro, filtered)

    news_titles = ""
    if news is not None and not news.empty and "title" in news.columns:
        recent = news.head(15)
        titles = recent["title"].tolist()
        news_titles = "\n".join(f"- {t}" for t in titles if isinstance(t, str))

    prompt = _build_analysis_prompt(macro, filtered, news_titles)
    result = _call_deepseek(prompt, api_key.strip())

    if result is None:
        return _build_fallback_analysis(macro, filtered)

    return {
        "source": "DeepSeek AI",
        "direction": str(result.get("direction", "neutral")),
        "confidence": float(result.get("confidence", 0.5)),
        "score": int(result.get("score", 0)),
        "key_drivers": str(result.get("key_drivers", "多因子模型评估")),
        "risk_factors": str(result.get("risk_factors", "需关注宏观数据变动")),
        "short_term_view": str(result.get("short_term_view", "观望")),
        "macro_assessment": str(result.get("macro_assessment", "宏观环境中性")),
    }


def _build_fallback_analysis(macro, filtered):
    latest_macro = macro.iloc[-1]
    latest_filtered = filtered.iloc[-1]

    z_s = float(latest_macro.get("z_sentiment", 0))
    z_d = float(latest_macro.get("z_dxy", 0))
    z_r = float(latest_macro.get("z_rate", 0))
    z_v = float(latest_macro.get("z_vix", 0)) if pd.notna(latest_macro.get("z_vix")) else 0
    gold_silver = float(latest_macro.get("z_gold_silver", 0)) if pd.notna(latest_macro.get("z_gold_silver")) else 0
    copper_z = float(latest_macro.get("z_copper", 0)) if pd.notna(latest_macro.get("z_copper")) else 0

    composite = z_s * 0.25 - z_d * 0.25 - z_r * 0.20 - z_v * 0.10 + gold_silver * 0.10 + copper_z * 0.10

    if composite > 0.5:
        direction = "bullish"
        short_view = "基本面因子偏多，关注价格突破信号"
    elif composite < -0.5:
        direction = "bearish"
        short_view = "基本面因子偏空，警惕下行风险"
    else:
        direction = "neutral"
        short_view = "基本面多空交织，震荡整理概率大"

    drivers_parts = []
    if z_s > 0.3:
        drivers_parts.append("新闻情绪偏正面")
    elif z_s < -0.3:
        drivers_parts.append("新闻情绪偏负面")
    if z_d > 0.3:
        drivers_parts.append("美元走强构成压力")
    elif z_d < -0.3:
        drivers_parts.append("美元走弱利好白银")
    if z_r > 0.3:
        drivers_parts.append("实际利率上升利空")
    elif z_r < -0.3:
        drivers_parts.append("实际利率下降利好")
    if z_v > 0.5:
        drivers_parts.append("VIX上升市场恐慌")
    drivers = "；".join(drivers_parts) if drivers_parts else "各因子处于中性区间"

    risks_parts = []
    if abs(z_v) > 1:
        risks_parts.append("VIX波动异常")
    if abs(z_d) > 1.5:
        risks_parts.append("美元极端波动")
    risks = "；".join(risks_parts) if risks_parts else "暂无显著极端风险"

    score = int(composite * 40)
    score = max(-100, min(100, score))

    return {
        "source": "本地多因子模型",
        "direction": direction,
        "confidence": round(min(abs(composite) / 2.0, 0.85), 2),
        "score": score,
        "key_drivers": drivers,
        "risk_factors": risks,
        "short_term_view": short_view,
        "macro_assessment": f"情绪{Z_s:+.1f} 美元Z_{z_d:+.1f} 利率Z_{z_r:+.1f}",
    }
