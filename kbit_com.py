#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Telegram bot with menu-based coin analysis and simple Bithumb trading."""

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd
import requests
from telegram import (
    Update,
    InputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

try:
    import pybithumb
except Exception:  # pragma: no cover - dependency may be missing at runtime
    pybithumb = None

BOT_TOKEN = "8216690986:AAHCxs_o5nXyOcbd6Sr9ooJhLgs5tcQ7024"
KST = dt.timezone(dt.timedelta(hours=9))

UPBIT_MIN_URL = "https://api.upbit.com/v1/candles/minutes/{unit}"


@dataclass
class TradeConfig:
    api_key: str = ""
    api_secret: str = ""
    percent: float = 0.0
    buy_time: str = ""
    sell_time: str = ""
    job_buy: Optional[object] = None
    job_sell: Optional[object] = None

trade_cfg = TradeConfig()

# ----- Conversation state keys -----
STATE = "state"
ANALYZE_TICKERS = "ANALYZE_TICKERS"
ANALYZE_DAYS = "ANALYZE_DAYS"
ANALYZE_UNIT = "ANALYZE_UNIT"
API_KEY = "API_KEY"
API_SECRET = "API_SECRET"
SETTRADE_PERCENT = "SETTRADE_PERCENT"
SETTRADE_BUY = "SETTRADE_BUY"
SETTRADE_SELL = "SETTRADE_SELL"

# ----- Utility functions -----
def parse_time_str(t: str) -> dt.time:
    if len(t) not in (3, 4):
        raise ValueError("HHMM format required")
    t = t.zfill(4)
    return dt.time(int(t[:2]), int(t[2:]), tzinfo=KST)


def fetch_candles(market: str, start: dt.datetime, end: dt.datetime, unit: int) -> pd.DataFrame:
    url = UPBIT_MIN_URL.format(unit=unit)
    results: List[Dict[str, object]] = []
    to = end
    while to > start:
        count = min(200, int((to - start).total_seconds() / 60 / unit))
        if count <= 0:
            break
        params = {
            "market": market,
            "to": to.strftime("%Y-%m-%d %H:%M:%S"),
            "count": count,
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if not data:
            break
        for item in data:
            t = dt.datetime.strptime(item["candle_date_time_kst"], "%Y-%m-%dT%H:%M:%S")
            t = t.replace(tzinfo=KST)
            if t < start:
                continue
            results.append({"time": t, "price": float(item["trade_price"])})
        last = data[-1]
        to = dt.datetime.strptime(last["candle_date_time_kst"], "%Y-%m-%dT%H:%M:%S")
        to = to.replace(tzinfo=KST) - dt.timedelta(minutes=unit)
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values("time", inplace=True)
    return df


async def perform_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE, tickers: List[str], days: int, unit: int) -> None:
    summary_lines = []
    for t in tickers:
        market = t if t.startswith("KRW-") else f"KRW-{t}"
        rows = []
        for i in range(days):
            d = dt.datetime.now(KST).date() - dt.timedelta(days=i)
            start_night = dt.datetime.combine(d - dt.timedelta(days=1), dt.time(21, 0, tzinfo=KST))
            end_night = dt.datetime.combine(d, dt.time(7, 0, tzinfo=KST))
            start_morn = dt.datetime.combine(d, dt.time(7, 0, tzinfo=KST))
            end_morn = dt.datetime.combine(d, dt.time(12, 0, tzinfo=KST))
            df_night = fetch_candles(market, start_night, end_night, unit)
            df_morn = fetch_candles(market, start_morn, end_morn, unit)
            if df_night.empty or df_morn.empty:
                continue
            min_row = df_night.loc[df_night["price"].idxmin()]
            max_row = df_morn.loc[df_morn["price"].idxmax()]
            rows.append({
                "date": d.isoformat(),
                "night_time": min_row["time"].strftime("%H:%M"),
                "night_price": min_row["price"],
                "morning_time": max_row["time"].strftime("%H:%M"),
                "morning_price": max_row["price"],
            })
        df = pd.DataFrame(rows)
        if df.empty:
            summary_lines.append(f"{t}: no data")
            continue
        filename = f"analysis_{t}_{days}d_{unit}m.xlsx"
        df.to_excel(filename, index=False)
        summary_lines.append(f"{t}: 저장 {filename} (rows={len(df)})")
        await update.message.reply_document(InputFile(filename))
    if summary_lines:
        await update.message.reply_text("\n".join(summary_lines))


# ----- Menu and conversation handlers -----
async def cmd_bts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("코인 분석", callback_data="analyze")],
        [InlineKeyboardButton("API 설정", callback_data="setapi")],
        [InlineKeyboardButton("매매 설정", callback_data="settrade")],
        [InlineKeyboardButton("매매 시작", callback_data="start")],
        [InlineKeyboardButton("매매 중지", callback_data="stop")],
        [InlineKeyboardButton("도움말", callback_data="help")],
    ]
    await update.message.reply_text(
        "메뉴를 선택하세요",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def menu_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "analyze":
        await query.message.reply_text("티커를 입력하세요 (예: MANA,ADA)")
        context.user_data[STATE] = ANALYZE_TICKERS
    elif data == "setapi":
        await query.message.reply_text("API 키를 입력하세요")
        context.user_data[STATE] = API_KEY
    elif data == "settrade":
        await query.message.reply_text("매매 비중을 입력하세요 (예: 50)")
        context.user_data[STATE] = SETTRADE_PERCENT
    elif data == "start":
        await cmd_starttrade(update, context)
    elif data == "stop":
        await cmd_stoptrade(update, context)
    elif data == "help":
        await cmd_help(update, context)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = context.user_data.get(STATE)
    text = update.message.text.strip()
    if state == ANALYZE_TICKERS:
        context.user_data["tickers"] = [t.strip().upper() for t in text.split(',') if t.strip()]
        await update.message.reply_text("조회 기간(일)을 입력하세요")
        context.user_data[STATE] = ANALYZE_DAYS
    elif state == ANALYZE_DAYS:
        context.user_data["days"] = int(text)
        await update.message.reply_text("분봉 단위를 입력하세요 (1-240)")
        context.user_data[STATE] = ANALYZE_UNIT
    elif state == ANALYZE_UNIT:
        tickers = context.user_data.pop("tickers")
        days = context.user_data.pop("days")
        context.user_data.pop(STATE, None)
        await perform_analysis(update, context, tickers, days, int(text))
    elif state == API_KEY:
        context.user_data["api_key"] = text
        await update.message.reply_text("API 시크릿을 입력하세요")
        context.user_data[STATE] = API_SECRET
    elif state == API_SECRET:
        trade_cfg.api_key = context.user_data.pop("api_key")
        trade_cfg.api_secret = text
        context.user_data.pop(STATE, None)
        await update.message.reply_text("API 설정 완료")
    elif state == SETTRADE_PERCENT:
        trade_cfg.percent = float(text)
        await update.message.reply_text("매입 시간을 입력하세요 (예: 1900)")
        context.user_data[STATE] = SETTRADE_BUY
    elif state == SETTRADE_BUY:
        trade_cfg.buy_time = text
        await update.message.reply_text("매각 시간을 입력하세요 (예: 2330)")
        context.user_data[STATE] = SETTRADE_SELL
    elif state == SETTRADE_SELL:
        trade_cfg.sell_time = text
        context.user_data.pop(STATE, None)
        await update.message.reply_text("매매 설정 완료")


# ----- Trade functions -----
def execute_buy(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not pybithumb or not trade_cfg.api_key:
        return
    # TODO: implement actual buy logic
    context.bot.send_message(context.job.chat_id, "[모의] 매수 실행")


def execute_sell(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not pybithumb or not trade_cfg.api_key:
        return
    # TODO: implement actual sell logic
    context.bot.send_message(context.job.chat_id, "[모의] 매도 실행")


async def cmd_starttrade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if trade_cfg.job_buy or trade_cfg.job_sell:
        await update.effective_message.reply_text("이미 실행 중입니다")
        return
    try:
        buy_t = parse_time_str(trade_cfg.buy_time)
        sell_t = parse_time_str(trade_cfg.sell_time)
    except Exception as e:
        await update.effective_message.reply_text(f"시간 설정 오류: {e}")
        return
    trade_cfg.job_buy = context.job_queue.run_daily(execute_buy, buy_t, chat_id=update.effective_chat.id)
    trade_cfg.job_sell = context.job_queue.run_daily(execute_sell, sell_t, chat_id=update.effective_chat.id)
    await update.effective_message.reply_text("자동 매매 시작")


async def cmd_stoptrade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    for job in (trade_cfg.job_buy, trade_cfg.job_sell):
        if job:
            job.schedule_removal()
    trade_cfg.job_buy = trade_cfg.job_sell = None
    await update.effective_message.reply_text("자동 매매 중지")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "1. /bts 명령으로 메뉴를 연 후 원하는 기능을 선택하세요.\n"
        "2. 각 기능은 안내에 따라 정보를 입력하면 됩니다."
    )


# ----- Main -----
def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("bts", cmd_bts))
    app.add_handler(CallbackQueryHandler(menu_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CommandHandler("help", cmd_help))
    app.run_polling()


if __name__ == "__main__":
    main()
