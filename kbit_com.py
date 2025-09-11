#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Telegram bot with menu-based coin analysis and simple Bithumb trading."""

# ----- install dependencies & prepare environment -----
import importlib
import os
import subprocess
import sys

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "requests": "requests",
    "openpyxl": "openpyxl",
    "python-dotenv": "dotenv",
    "python-telegram-bot": "telegram",
    "pybithumb": "pybithumb",
}

for pkg, mod in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError:  # pragma: no cover - installation branch
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

from dotenv import load_dotenv

load_dotenv()

# Default env vars (can be overridden externally)
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "8216690986:AAHCxs_o5nXyOcbd6Sr9ooJh")
os.environ.setdefault("BITHUMB_API_KEY", "YOUR_API_KEY")
os.environ.setdefault("BITHUMB_API_SECRET", "YOUR_API_SECRET")

import argparse
import datetime as dt
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
KST = dt.timezone(dt.timedelta(hours=9))

UPBIT_MIN_URL = "https://api.upbit.com/v1/candles/minutes/{unit}"

logging.basicConfig(level=logging.INFO)


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

# ----- HOD analysis utilities -----
@dataclass
class FetchConfig:
    market: str = "KRW-ETC"
    unit: int = 60
    days: int = 300
    pause: float = 0.13


def kst_to_utc_str(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=KST)
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fetch_candles(cfg: FetchConfig) -> pd.DataFrame:
    target_rows = cfg.days * (24 * 60 // cfg.unit) + 200
    url = UPBIT_MIN_URL.format(unit=cfg.unit)
    frames: List[pd.DataFrame] = []
    to_cursor: Optional[str] = None
    collected = 0
    while collected < target_rows:
        params = {"market": cfg.market, "count": 200}
        if to_cursor:
            params["to"] = to_cursor
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        df = pd.DataFrame(data)
        df = df.iloc[::-1].reset_index(drop=True)
        frames.append(
            df[["candle_date_time_kst", "opening_price", "high_price", "low_price", "trade_price"]].copy()
        )
        collected += len(df)
        oldest_kst = pd.to_datetime(df["candle_date_time_kst"].iloc[0])
        to_cursor = kst_to_utc_str(oldest_kst - dt.timedelta(minutes=cfg.unit))
        time.sleep(cfg.pause)
        if len(frames) > 4000:
            break
    if not frames:
        return pd.DataFrame(columns=["ts_kst", "open", "high", "low", "close"])
    out = pd.concat(frames, ignore_index=True).rename(
        columns={
            "candle_date_time_kst": "ts_kst",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
        }
    )
    out["ts_kst"] = pd.to_datetime(out["ts_kst"])
    out = out.sort_values("ts_kst").reset_index(drop=True)
    cutoff = out["ts_kst"].max() - pd.Timedelta(days=cfg.days)
    out = out[out["ts_kst"] >= cutoff].reset_index(drop=True)
    return out


def parse_window(spec: str) -> Tuple[int, int]:
    spec = spec.strip()
    if "-" not in spec:
        raise ValueError("시간창은 '시작-끝' 형식이어야 합니다. 예: 22-7")
    a, b = spec.split("-", 1)
    start = int(a)
    end = int(b)
    if not (0 <= start <= 23 and 0 <= end <= 23):
        raise ValueError("시간은 0~23 사이 정수여야 합니다.")
    return start, end


def mask_window_hour(df: pd.DataFrame, col: str, start: int, end: int) -> pd.DataFrame:
    h = df[col]
    if start <= end:
        return df[(h >= start) & (h <= end)]
    return df[(h >= start) | (h <= end)]


def find_best_hours(
    df: pd.DataFrame, night_win: Tuple[int, int], morning_win: Tuple[int, int]
) -> Tuple[int, int, float, float]:
    tmp = df.copy()
    tmp["hour"] = tmp["ts_kst"].dt.hour
    night = mask_window_hour(tmp, "hour", *night_win)
    morning = mask_window_hour(tmp, "hour", *morning_win)
    if night.empty or morning.empty:
        raise RuntimeError("시간대 필터 결과가 비어 있습니다.")
    buy_series = night.groupby("hour")["low"].mean()
    sell_series = morning.groupby("hour")["high"].mean()
    buy_hour = int(buy_series.idxmin())
    sell_hour = int(sell_series.idxmax())
    return buy_hour, sell_hour, float(buy_series.min()), float(sell_series.max())


def simulate_daily(df: pd.DataFrame, buy_hour: int, sell_hour: int, initial: float = 1_000_000.0) -> Tuple[float, int]:
    d = df.copy().sort_values("ts_kst")
    d["date"] = d["ts_kst"].dt.date
    d["hour"] = d["ts_kst"].dt.hour
    dates = sorted(d["date"].unique())
    capital = float(initial)
    trades = 0
    for i in range(len(dates) - 1):
        today, next_day = dates[i], dates[i + 1]
        buy_row = d[(d["date"] == today) & (d["hour"] == buy_hour)]
        sell_row = d[(d["date"] == next_day) & (d["hour"] == sell_hour)]
        if buy_row.empty or sell_row.empty:
            continue
        buy_price = float(buy_row.iloc[0]["close"])
        sell_price = float(sell_row.iloc[0]["close"])
        if buy_price <= 0:
            continue
        qty = capital / buy_price
        capital = qty * sell_price
        trades += 1
    return round(capital, 2), trades


def to_excel(df: pd.DataFrame, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(outpath, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Raw")


def analyze_hod(
    market: str,
    days: int,
    unit: int,
    night_spec: str,
    morning_spec: str,
    save_raw: Optional[str] = None,
) -> str:
    df = fetch_candles(FetchConfig(market=market, unit=unit, days=days))
    if df.empty:
        return "[ERROR] 데이터 수집 실패"
    night = parse_window(night_spec)
    morning = parse_window(morning_spec)
    buy_hour, sell_hour, buy_mean_low, sell_mean_high = find_best_hours(df, night, morning)
    final_capital, trades = simulate_daily(df, buy_hour, sell_hour, initial=1_000_000.0)
    total_ret = (final_capital / 1_000_000.0 - 1.0) * 100.0
    lines = [
        f"[INFO] 수집: {market}, 최근 {days}일, {unit}분봉",
        f"[RESULT] 매수 시간대(야간 {night_spec} 중 평균 저가 최저): {buy_hour}시 (평균 저가 {buy_mean_low:.4f})",
        f"[RESULT] 매도 시간대(아침 {morning_spec} 중 평균 고가 최고): {sell_hour}시 (평균 고가 {sell_mean_high:.4f})",
        f"[SIM] 거래 횟수: {trades}",
        f"[SIM] 초기 1,000,000원 → 최종 {final_capital:,.2f}원 (누적 수익률 {total_ret:.2f}%)",
    ]
    if save_raw:
        to_excel(df, Path(save_raw))
        lines.append(f"[SAVE] Raw 저장: {save_raw}")
    return "\n".join(lines)


def run_hod_cli(market: str, days: int, unit: int, night_spec: str, morning_spec: str, save_raw: Optional[str] = None):
    print(analyze_hod(market, days, unit, night_spec, morning_spec, save_raw=save_raw))


def prompt_input() -> Tuple[str, int, int, str, str]:
    m = input("마켓(예: KRW-ETC, KRW-BTC, KRW-ADA): ").strip().upper()
    if not m.startswith("KRW-"):
        m = f"KRW-{m}"
    while True:
        s = input("최근 N일 (기본 300): ").strip()
        if not s:
            d = 300
            break
        if s.isdigit() and int(s) > 0:
            d = int(s)
            break
        print("양의 정수를 입력하세요.")
    u = input("분봉 단위(기본 60): ").strip()
    unit = int(u) if (u.isdigit() and int(u) > 0) else 60
    night = input("야간창(기본 22-7): ").strip() or "22-7"
    morning = input("아침창(기본 7-12): ").strip() or "7-12"
    return m, d, unit, night, morning

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


def fetch_candles_range(market: str, start: dt.datetime, end: dt.datetime, unit: int) -> pd.DataFrame:
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
            df_night = fetch_candles_range(market, start_night, end_night, unit)
            df_morn = fetch_candles_range(market, start_morn, end_morn, unit)
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


async def cmd_hod(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    market = args[0] if len(args) > 0 else "KRW-ETC"
    days = int(args[1]) if len(args) > 1 else 300
    unit = int(args[2]) if len(args) > 2 else 60
    night = args[3] if len(args) > 3 else "22-7"
    morning = args[4] if len(args) > 4 else "7-12"
    try:
        result = analyze_hod(market, days, unit, night, morning)
    except Exception as e:
        result = f"[ERROR] {e}"
    await update.message.reply_text(result)


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
        "1. /start 또는 /bts 명령으로 메뉴를 연 후 원하는 기능을 선택하세요.\n"
        "2. 각 기능은 안내에 따라 정보를 입력하면 됩니다.\n"
        "3. /hod [티커] [일수] [분봉] [야간] [아침] 명령으로 HOD 분석을 실행합니다."
    )


# ----- Main -----
def start_bot() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler(["bts", "start"], cmd_bts))
    app.add_handler(CallbackQueryHandler(menu_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("hod", cmd_hod))
    logging.info("Bot started. Waiting for commands...")
    app.run_polling()


def main() -> None:
    ap = argparse.ArgumentParser(description="Telegram bot and HOD analysis")
    ap.add_argument("--bot", action="store_true", help="Run Telegram bot")
    ap.add_argument("--market", type=str, help="마켓 티커 (예: KRW-ETC)")
    ap.add_argument("--days", type=int, help="최근 N일 (기본 300)")
    ap.add_argument("--unit", type=int, default=60, help="분봉 단위(기본 60)")
    ap.add_argument("--night", type=str, default="22-7", help="야간창 (기본 22-7)")
    ap.add_argument("--morning", type=str, default="7-12", help="아침창 (기본 7-12)")
    ap.add_argument("--save-raw", type=str, default=None, help="Raw 저장 경로(excel). 지정 시 저장")
    ap.add_argument("--interactive", action="store_true", help="프롬프트로 입력 받기")
    args = ap.parse_args()
    if args.bot or not (args.market and args.days) and not args.interactive:
        start_bot()
        return
    if args.interactive or not (args.market and args.days):
        market, days, unit, night, morning = prompt_input()
    else:
        market = args.market.upper()
        if not market.startswith("KRW-"):
            market = "KRW-" + market
        days = args.days if args.days else 300
        unit = args.unit
        night = args.night
        morning = args.morning
    run_hod_cli(market, days, unit, night, morning, save_raw=args.save_raw)


if __name__ == "__main__":
    main()
