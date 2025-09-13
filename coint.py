#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
텔레그램 - 우분투 - 코인 분석기 (Upbit 공용 API 기반)
기능:
  1) 변동성 상위 Top10 (기간: 200/300일 등)
  2) 거래액(원화) Top10 (기간: 200/300일 등)
  3) 일일단가 수익률 비교 백테스트:
     - 종목 (예: KRW-BTC)
     - 조회기간(일)
     - 매입 시간대 (기본: 19:00~익일 01:00)
     - 매각 시간대 (기본: 익일 08:00~익일 12:00)
     - 분봉(1/3/5/10/15/30/60/240)
     - 기간 동안 평균적으로 '가장 싼 시간(매입)'과 '가장 비싼 시간(매도)'을 각 시간대에서 찾고,
       매일 100만원을 동일 시간에 사고/팔았다면 결과(고정 예산 합산 vs 매일 재투자 복리)를 계산
명령:
  /bts  : 메뉴 시작 (선택형)
  /start: 안내

주의:
  - Upbit 공개 API 사용(무인증), 과도한 호출 방지.
  - 시간대는 Asia/Seoul 기준.
"""

import asyncio
import datetime as dt
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import aiohttp
import pandas as pd
import numpy as np
import pytz

from telegram import (
    Update, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove
)
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, ConversationHandler,
    CallbackQueryHandler, MessageHandler, filters
)

# ========= 통합 설정 (여기만 수정하면 됨) =========
TOKEN = "8216690986:AAHCxs_o5nXyOcbd6Sr9ooJhLgs5tcQ7024"  # 요청하신 대로 하드코딩
TZ = pytz.timezone("Asia/Seoul")

# 텔레그램 접근 제어: 특정 사용자만 사용하려면 ID를 적으세요. 비우면 전체 허용.
ALLOWED_USER_IDS: List[int] = []  # 예: [5517670242]

# Upbit API
UPBIT_BASE = "https://api.upbit.com"
API_REQUEST_DELAY = 0.15  # seconds between Upbit API calls

# 기본 백테스트 설정
DEFAULT_BUY_WINDOW = ("19:00", "01:00")  # (시작, 끝) 끝이 익일 가능
DEFAULT_SELL_WINDOW = ("08:00", "12:00")  # (시작, 끝) 동일/익일 가능
DEFAULT_INTERVAL_MIN = 5
DEFAULT_BUDGET = 1_000_000  # 원
MAX_CANDLE_COUNT = 200  # Upbit 분봉/일봉 요청 최대 200

# ========== 내부 상태키 ==========
(
    MENU,                # 루트 메뉴
    Q1_PERIOD,           # 기능1: 변동성 기간
    Q2_PERIOD,           # 기능2: 거래액 기간
    Q3_SYMBOL,           # 기능3: 종목
    Q3_PERIOD,           # 기능3: 기간
    Q3_BUY_WINDOW,       # 기능3: 매입 시간대
    Q3_SELL_WINDOW,      # 기능3: 매도 시간대
    Q3_INTERVAL          # 기능3: 분봉
) = range(8)

# ========== 유틸 ==========
def user_allowed(user_id: int) -> bool:
    return (not ALLOWED_USER_IDS) or (user_id in ALLOWED_USER_IDS)

def parse_hhmm(s: str) -> dt.time:
    return dt.datetime.strptime(s.strip(), "%H:%M").time()

def time_range_minutes(start: dt.time, end: dt.time) -> List[int]:
    """
    분 단위 '하루 내 offset(0~1439)' 배열 생성. 종료가 익일이면 그 구간을 이어붙인다.
    """
    def to_min(t: dt.time) -> int:
        return t.hour * 60 + t.minute

    s = to_min(start)
    e = to_min(end)
    if s <= e:
        return list(range(s, e + 1))
    else:
        return list(range(s, 24*60)) + list(range(0, e + 1))

def localize(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        return TZ.localize(ts)
    return ts.astimezone(TZ)

def kst_now() -> dt.datetime:
    return dt.datetime.now(tz=TZ)

# ========== Upbit API ==========
async def http_get_json(session: aiohttp.ClientSession, url: str, params: dict=None):
    """GET 요청을 보내고 JSON을 반환한다.
    429(Too Many Requests) 발생 시 잠시 대기 후 최대 5회까지 재시도한다.
    """
    for _ in range(5):
        async with session.get(url, params=params, timeout=30) as resp:
            if resp.status == 429:
                await asyncio.sleep(1)
                continue
            resp.raise_for_status()
            return await resp.json()
    raise RuntimeError("Too Many Requests: repeated 429 responses")

async def fetch_markets(session: aiohttp.ClientSession) -> List[str]:
    url = f"{UPBIT_BASE}/v1/market/all"
    data = await http_get_json(session, url, {"isDetails": "true"})
    markets = [d["market"] for d in data if d["market"].startswith("KRW-")]
    return markets

async def fetch_daily_candles(session: aiohttp.ClientSession, market: str, count: int) -> pd.DataFrame:
    url = f"{UPBIT_BASE}/v1/candles/days"
    data = await http_get_json(session, url, {"market": market, "count": min(count, MAX_CANDLE_COUNT)})
    # 최근 -> 과거 순으로 내려옴
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # 표준화
    df["date_kst"] = pd.to_datetime(df["candle_date_time_kst"])
    df["date_kst"] = df["date_kst"].dt.tz_localize("Asia/Seoul")
    df.rename(columns={
        "opening_price": "open",
        "trade_price": "close",
        "high_price": "high",
        "low_price": "low",
        "candle_acc_trade_price": "value_krw",
        "candle_acc_trade_volume": "volume"
    }, inplace=True)
    keep = ["date_kst", "open", "close", "high", "low", "value_krw", "volume"]
    return df[keep].sort_values("date_kst")

async def fetch_minute_candles(
    session: aiohttp.ClientSession, market: str, unit: int, end_time_kst: dt.datetime, need: int
) -> pd.DataFrame:
    """
    분봉 데이터를 끝 시각 기준으로 MAX_CANDLE_COUNT씩 당겨오며 need개 이상 확보.
    Upbit는 최근→과거 순으로 반환하므로 적절히 정렬.
    """
    url = f"{UPBIT_BASE}/v1/candles/minutes/{unit}"
    dfs = []
    remain = need
    cursor = end_time_kst

    while remain > 0:
        params = {
            "market": market,
            "count": min(MAX_CANDLE_COUNT, remain),
            "to": cursor.strftime("%Y-%m-%d %H:%M:%S")
        }
        data = await http_get_json(session, url, params)
        if not data:
            break
        df = pd.DataFrame(data)
        df["time_kst"] = pd.to_datetime(df["candle_date_time_kst"])
        df["time_kst"] = df["time_kst"].dt.tz_localize("Asia/Seoul")
        df.rename(columns={
            "trade_price": "close"
        }, inplace=True)
        dfs.append(df[["time_kst", "close"]])
        cursor = df["time_kst"].iloc[-1] - dt.timedelta(seconds=unit*60)  # 다음 요청 anchor
        remain -= len(df)
    if not dfs:
        return pd.DataFrame(columns=["time_kst", "close"])
    out = pd.concat(dfs, ignore_index=True)
    return out.sort_values("time_kst")

async def iter_markets_daily(session: aiohttp.ClientSession, period_days: int):
    """Yield (market, daily_df) for KRW markets with enough history."""
    markets = await fetch_markets(session)
    min_days = period_days
    for m in markets:
        df = await fetch_daily_candles(session, m, period_days)
        if len(df) >= min_days:
            yield m, df
        await asyncio.sleep(API_REQUEST_DELAY)

# ========== 계산 로직 ==========
async def calc_top_volatility(period_days: int) -> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        rows: List[Dict[str, float]] = []
        async for m, df in iter_markets_daily(session, period_days):
            oc_vol = (df["close"] - df["open"]).abs() / df["open"] * 100.0
            rows.append({"market": m, "mean_oc_volatility_pct": float(oc_vol.mean())})
        if not rows:
            return pd.DataFrame(columns=["market", "mean_oc_volatility_pct"])
        out = pd.DataFrame(rows)
        out = out.sort_values("mean_oc_volatility_pct", ascending=False).head(10).reset_index(drop=True)
        return out

async def calc_top_value(period_days: int) -> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        rows: List[Dict[str, float]] = []
        async for m, df in iter_markets_daily(session, period_days):
            rows.append({"market": m, "mean_daily_value_krw": float(df["value_krw"].mean())})
        if not rows:
            return pd.DataFrame(columns=["market", "mean_daily_value_krw"])
        out = pd.DataFrame(rows)
        out = out.sort_values("mean_daily_value_krw", ascending=False).head(10).reset_index(drop=True)
        return out

def _minutes_of_day(ts: pd.Timestamp) -> int:
    kst = ts.tz_convert("Asia/Seoul")
    return kst.hour * 60 + kst.minute

def _window_to_minutes(window: Tuple[str, str]) -> List[int]:
    s = parse_hhmm(window[0])
    e = parse_hhmm(window[1])
    return time_range_minutes(s, e)

async def backtest_intraday(
    market: str,
    period_days: int,
    buy_window: Tuple[str, str],
    sell_window: Tuple[str, str],
    interval_min: int,
    budget_krw: int = DEFAULT_BUDGET
) -> Dict:
    """
    1) 기간 동안 '분봉 close'를 일자별로 분해
    2) buy_window 내 각 '분 단위'의 평균가격을 계산해 최저 평균 분(min_buy)을 선택
    3) sell_window 내 각 '분 단위'의 평균가격을 계산해 최고 평균 분(min_sell)을 선택
    4) 각 날짜별로 해당 시점 가격을 사용하여 일일 매수/매도 수익률 계산
       - 고정예산(매일 100만원 별도) 합산
       - 재투자(복리) 누적
    """
    end = kst_now()
    need_minutes = period_days * int(24*60/interval_min) + 400  # 버퍼
    async with aiohttp.ClientSession() as session:
        df = await fetch_minute_candles(session, market, interval_min, end, need_minutes)
    if df.empty:
        return {"error": "분봉 데이터를 가져오지 못했습니다."}

    # 분-오프셋(0~1439) 계산
    df["min_of_day"] = df["time_kst"].apply(lambda x: _minutes_of_day(pd.Timestamp(x)))
    df["date"] = df["time_kst"].dt.tz_convert("Asia/Seoul").dt.date

    # 기간 제한
    first_day = (end - dt.timedelta(days=period_days)).date()
    df = df[df["date"] >= first_day]
    if df.empty:
        return {"error": "해당 기간의 분봉 데이터가 없습니다."}

    # 윈도우 후보 분 리스트
    buy_candidates = _window_to_minutes(buy_window)
    sell_candidates = _window_to_minutes(sell_window)

    # 각 분(분해능은 interval_min에 의존)으로 샘플 매핑
    # interval_min에 맞춰 분을 반올림
    def round_to_interval(m):
        return (m // interval_min) * interval_min

    df["bucket_min"] = df["min_of_day"].apply(round_to_interval)

    # 평균가격 by bucket
    buy_avg = (
        df[df["bucket_min"].isin(buy_candidates)]
        .groupby("bucket_min")["close"]
        .mean()
        .sort_values(ascending=True)
    )
    sell_avg = (
        df[df["bucket_min"].isin(sell_candidates)]
        .groupby("bucket_min")["close"]
        .mean()
        .sort_values(ascending=False)
    )

    if buy_avg.empty or sell_avg.empty:
        return {"error": "지정한 시간대에 데이터가 충분하지 않습니다."}

    best_buy_min = int(buy_avg.index[0])
    best_sell_min = int(sell_avg.index[0])

    # 일자별 해당 시점 가격 추출
    buy_price_by_day = (
        df[df["bucket_min"] == best_buy_min]
        .groupby("date")["close"].last()
    )
    sell_price_by_day = (
        df[df["bucket_min"] == best_sell_min]
        .groupby("date")["close"].last()
    )

    # 매도는 '익일'로 가정 → 날짜+1
    sell_price_by_day.index = [d + dt.timedelta(days=1) for d in sell_price_by_day.index]

    # 교집합 날짜로 수익률 산출
    days = sorted(set(buy_price_by_day.index).intersection(set(sell_price_by_day.index)))
    results = []
    capital = budget_krw  # 재투자 시작자본
    cum_capitals = []     # 재투자 궤적

    for d in days:
        bp = float(buy_price_by_day.get(d, np.nan))
        sp = float(sell_price_by_day.get(d, np.nan))
        if np.isnan(bp) or np.isnan(sp) or bp <= 0:
            continue
        ratio = sp / bp
        # 고정 예산: 매일 100만원을 별개로 투자하여 누적 이익 집계
        daily_gain = budget_krw * (ratio - 1.0)
        results.append({"date": d, "buy": bp, "sell": sp, "ratio": ratio, "daily_gain": daily_gain})
        # 재투자(복리)
        capital *= ratio
        cum_capitals.append(capital)

    if not results:
        return {"error": "매수/매도 시점이 겹치는 일자가 충분하지 않습니다."}

    res_df = pd.DataFrame(results).sort_values("date")
    total_fixed = budget_krw * len(res_df) + res_df["daily_gain"].sum()
    # 재투자: 초기가격 budget_krw → 마지막 capital
    reinvest_final = budget_krw if not cum_capitals else cum_capitals[-1]

    return {
        "market": market,
        "period_days": period_days,
        "interval_min": interval_min,
        "best_buy_min": best_buy_min,
        "best_sell_min": best_sell_min,
        "n_trades": len(res_df),
        "fixed_total_krw": int(round(total_fixed)),
        "reinvest_final_krw": int(round(reinvest_final)),
        "buy_time_str": f"{best_buy_min//60:02d}:{best_buy_min%60:02d}",
        "sell_time_str": f"{best_sell_min//60:02d}:{best_sell_min%60:02d}",
        "sample": res_df.tail(5).to_string(index=False)  # 최근 5일 샘플
    }

# ========== 텔레그램 핸들러 ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_allowed(update.effective_user.id):
        return
    await update.message.reply_text(
        "안녕하세요! /bts 를 입력하면 코인 분석 메뉴가 열립니다.",
        reply_markup=ReplyKeyboardRemove()
    )

async def bts_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_allowed(update.effective_user.id):
        return ConversationHandler.END
    kb = [
        [InlineKeyboardButton("1) 변동성 Top10", callback_data="MENU_VOL")],
        [InlineKeyboardButton("2) 거래액 Top10", callback_data="MENU_VAL")],
        [InlineKeyboardButton("3) 일일단가 수익률 비교 (백테스트)", callback_data="MENU_BT")],
        [InlineKeyboardButton("닫기", callback_data="MENU_END")],
    ]
    if update.message:
        await update.message.reply_text("원하는 작업을 선택하세요:", reply_markup=InlineKeyboardMarkup(kb))
    else:
        await update.callback_query.edit_message_text("원하는 작업을 선택하세요:", reply_markup=InlineKeyboardMarkup(kb))
    return MENU

# ---- 메뉴 분기
async def on_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_allowed(update.effective_user.id):
        return ConversationHandler.END
    q = update.callback_query
    await q.answer()
    data = q.data
    if data == "MENU_VOL":
        await q.edit_message_text("변동성 조회기간을 입력하시오 (예: 200 또는 300)")
        return Q1_PERIOD
    elif data == "MENU_VAL":
        await q.edit_message_text("거래액 조회기간을 입력하시오 (예: 200 또는 300)")
        return Q2_PERIOD
    elif data == "MENU_BT":
        await q.edit_message_text("종목을 입력하시오 (예: KRW-BTC, KRW-ETH)")
        return Q3_SYMBOL
    else:
        await q.edit_message_text("종료합니다.")
        return ConversationHandler.END

# ---- 기능 1: 변동성
async def on_q1_period(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_allowed(update.effective_user.id):
        return ConversationHandler.END
    try:
        period = int(update.message.text.strip())
        if period > MAX_CANDLE_COUNT:
            period = MAX_CANDLE_COUNT
            await update.message.reply_text("Upbit API 한계로 200일까지만 조회합니다.")
        await update.message.reply_text("계산 중입니다. 잠시만요…")
        df = await calc_top_volatility(period)
        if df.empty:
            await update.message.reply_text("결과가 없습니다.")
        else:
            lines = ["[변동성 Top10] (기간: {}일)".format(period)]
            for i, row in df.iterrows():
                lines.append(f"{i+1:>2}. {row['market']}: {row['mean_oc_volatility_pct']:.2f}%")
            await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"오류: {e}")
    # 끝나면 다시 메뉴
    return await bts_menu(update, context)

# ---- 기능 2: 거래액
async def on_q2_period(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_allowed(update.effective_user.id):
        return ConversationHandler.END
    try:
        period = int(update.message.text.strip())
        if period > MAX_CANDLE_COUNT:
            period = MAX_CANDLE_COUNT
            await update.message.reply_text("Upbit API 한계로 200일까지만 조회합니다.")
        await update.message.reply_text("계산 중입니다. 잠시만요…")
        df = await calc_top_value(period)
        if df.empty:
            await update.message.reply_text("결과가 없습니다.")
        else:
            lines = ["[거래액(원화) Top10] (기간: {}일)".format(period)]
            for i, row in df.iterrows():
                lines.append(f"{i+1:>2}. {row['market']}: {int(row['mean_daily_value_krw']):,} 원/일")
            await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"오류: {e}")
    return await bts_menu(update, context)

# ---- 기능 3: 백테스트
async def on_q3_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not user_allowed(update.effective_user.id):
        return ConversationHandler.END
    sym = update.message.text.strip().upper()
    if not sym.startswith("KRW-"):
        await update.message.reply_text("형식: KRW-XXX (예: KRW-BTC)")
        return Q3_SYMBOL
    context.user_data["bt_symbol"] = sym
    await update.message.reply_text("조회기간(일)을 입력하시오 (예: 200, 300)")
    return Q3_PERIOD

async def on_q3_period(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        period = int(update.message.text.strip())
    except:
        await update.message.reply_text("숫자로 입력해주세요. (예: 200)")
        return Q3_PERIOD
    context.user_data["bt_period"] = period
    bw = f"{DEFAULT_BUY_WINDOW[0]}~{DEFAULT_BUY_WINDOW[1]}"
    await update.message.reply_text(
        f"매입 시간대를 입력하시오 (기본: {bw})\n예) 19:00~01:00",
    )
    return Q3_BUY_WINDOW

async def on_q3_buy_window(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    if "~" not in txt:
        await update.message.reply_text("형식: HH:MM~HH:MM (예: 19:00~01:00)")
        return Q3_BUY_WINDOW
    s, e = [t.strip() for t in txt.split("~", 1)]
    context.user_data["bt_buy_window"] = (s, e)
    sw = f"{DEFAULT_SELL_WINDOW[0]}~{DEFAULT_SELL_WINDOW[1]}"
    await update.message.reply_text(
        f"매각 시간대를 입력하시오 (기본: {sw})\n예) 08:00~12:00",
    )
    return Q3_SELL_WINDOW

async def on_q3_sell_window(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    if "~" not in txt:
        await update.message.reply_text("형식: HH:MM~HH:MM (예: 08:00~12:00)")
        return Q3_SELL_WINDOW
    s, e = [t.strip() for t in txt.split("~", 1)]
    context.user_data["bt_sell_window"] = (s, e)
    await update.message.reply_text(
        f"분봉(1/3/5/10/15/30/60/240)을 입력하시오 (기본: {DEFAULT_INTERVAL_MIN})"
    )
    return Q3_INTERVAL

async def on_q3_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        interval = int(update.message.text.strip())
        if interval not in [1,3,5,10,15,30,60,240]:
            raise ValueError
    except:
        await update.message.reply_text("분봉은 1/3/5/10/15/30/60/240 중 하나여야 합니다.")
        return Q3_INTERVAL

    sym = context.user_data["bt_symbol"]
    period = context.user_data["bt_period"]
    buyw = context.user_data.get("bt_buy_window", DEFAULT_BUY_WINDOW)
    sellw = context.user_data.get("bt_sell_window", DEFAULT_SELL_WINDOW)

    await update.message.reply_text("백테스트 계산 중입니다. 다소 시간이 걸릴 수 있어요…")
    try:
        res = await backtest_intraday(sym, period, buyw, sellw, interval, DEFAULT_BUDGET)
        if "error" in res:
            await update.message.reply_text(f"오류: {res['error']}")
        else:
            msg = (
                f"[백테스트 결과]\n"
                f"종목: {res['market']}\n"
                f"기간: {res['period_days']}일, 분봉: {res['interval_min']}분\n"
                f"선택된 매입 시각(평균 최저): {res['buy_time_str']}\n"
                f"선택된 매도 시각(평균 최고): {res['sell_time_str']}\n"
                f"체결 가능 일수: {res['n_trades']}일\n"
                f"① 고정예산 합산 결과: {res['fixed_total_krw']:,} 원\n"
                f"② 매일 재투자(복리) 결과: {res['reinvest_final_krw']:,} 원\n"
                f"\n최근 5일 샘플:\n{res['sample']}"
            )
            await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"오류: {e}")

    return await bts_menu(update, context)

# ========== 앱 구동 ==========
import logging
logging.basicConfig(level=logging.INFO)

def build_app():
    app = ApplicationBuilder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("bts", bts_menu)],
        states={
            MENU: [CallbackQueryHandler(on_menu)],
            Q1_PERIOD: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_q1_period)],
            Q2_PERIOD: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_q2_period)],
            Q3_SYMBOL: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_q3_symbol)],
            Q3_PERIOD: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_q3_period)],
            Q3_BUY_WINDOW: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_q3_buy_window)],
            Q3_SELL_WINDOW: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_q3_sell_window)],
            Q3_INTERVAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_q3_interval)],
        },
        fallbacks=[CommandHandler("bts", bts_menu)],
        name="bts-conv",
        persistent=False,
        # ⛔️ per_message=True 는 사용하지 않습니다 (MessageHandler가 섞여있음)
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)
    return app


def main():
    app = build_app()
    # PTB v21 권장: 한 줄로 시작/대기까지 처리
    # 다른 인스턴스가 남아있을 때 발생하는 409 오류를 방지하기 위해
    # 기존 웹훅/업데이트를 정리(drop_pending_updates=True)
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
