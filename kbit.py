#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Telegram bot for coin analysis and simple Bithumb trading."""

import os
import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd
import requests
from telegram import Update, InputFile
from telegram.ext import (
    Application, CommandHandler, ContextTypes, JobQueue
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

# ----- Utility functions -----
def parse_time_str(t: str) -> dt.time:
    if len(t) not in (3,4):
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


async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/analyze <tickers> <days> <unit>"""
    if len(context.args) != 3:
        await update.message.reply_text("Usage: /analyze TICKERS DAYS UNIT\nExample: /analyze MANA,ADA 30 60")
        return
    tickers = [t.strip().upper() for t in context.args[0].split(',') if t.strip()]
    days = int(context.args[1])
    unit = int(context.args[2])
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
        summary_lines.append(
            f"{t}: 저장 {filename} (rows={len(df)})")
        await update.message.reply_document(InputFile(filename))
    if summary_lines:
        await update.message.reply_text("\n".join(summary_lines))


async def cmd_setapi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /setapi KEY SECRET")
        return
    trade_cfg.api_key, trade_cfg.api_secret = context.args
    await update.message.reply_text("API key set")


async def cmd_settrade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) != 3:
        await update.message.reply_text("Usage: /settrade PERCENT BUYTIME SELLTIME")
        return
    trade_cfg.percent = float(context.args[0])
    trade_cfg.buy_time = context.args[1]
    trade_cfg.sell_time = context.args[2]
    await update.message.reply_text("Trade config updated")


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
        await update.message.reply_text("Already running")
        return
    try:
        buy_t = parse_time_str(trade_cfg.buy_time)
        sell_t = parse_time_str(trade_cfg.sell_time)
    except Exception as e:
        await update.message.reply_text(f"시간 설정 오류: {e}")
        return
    trade_cfg.job_buy = context.job_queue.run_daily(execute_buy, buy_t, chat_id=update.effective_chat.id)
    trade_cfg.job_sell = context.job_queue.run_daily(execute_sell, sell_t, chat_id=update.effective_chat.id)
    await update.message.reply_text("Auto trade started")


async def cmd_stoptrade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    for job in (trade_cfg.job_buy, trade_cfg.job_sell):
        if job:
            job.schedule_removal()
    trade_cfg.job_buy = trade_cfg.job_sell = None
    await update.message.reply_text("Auto trade stopped")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/analyze TICKERS DAYS UNIT\n"
        "/setapi KEY SECRET\n"
        "/settrade PERCENT BUYTIME SELLTIME\n"
        "/starttrade\n/stoptrade"
    )


def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("setapi", cmd_setapi))
    app.add_handler(CommandHandler("settrade", cmd_settrade))
    app.add_handler(CommandHandler("starttrade", cmd_starttrade))
    app.add_handler(CommandHandler("stoptrade", cmd_stoptrade))
    app.add_handler(CommandHandler("help", cmd_help))
    app.run_polling()


if __name__ == "__main__":
    main()
