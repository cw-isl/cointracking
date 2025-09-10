#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Upbit KRW-마켓 다중 코인 자동 분석 & Gmail 자동 전송 스크립트
- .env 경로는 고정: /root/solana_upbit/.env
- Gmail 계정/앱 비밀번호/수신자 주소는 .env에서 불러옴
- 인터랙티브 모드 지원: (아무 인자 없이 실행) 또는 --interactive
"""

import os
import sys
import time
import smtplib
import argparse
import datetime as dt
from typing import Optional, List, Tuple
from email.message import EmailMessage
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# ==== .env 경로 고정 ====
ENV_PATH = Path("/root/solana_upbit/.env")
if not ENV_PATH.exists():
    print(f"[ERROR] {ENV_PATH} 파일을 찾을 수 없습니다.")
    sys.exit(1)

load_dotenv(dotenv_path=ENV_PATH)

# ==== Gmail 설정 ====
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
GMAIL_USER = os.getenv("GMAIL_USER", "").strip()
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").strip()
DEFAULT_RECIPIENTS = os.getenv("REPORT_RECIPIENTS", "").strip()

KST = dt.timezone(dt.timedelta(hours=9))
UPBIT_MIN1_URL = "https://api.upbit.com/v1/candles/minutes/1"


# ===== Helper Functions =====
def kst_to_utc_str(kst_dt: dt.datetime) -> str:
    if kst_dt.tzinfo is None:
        kst_dt = kst_dt.replace(tzinfo=KST)
    return kst_dt.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fetch_minute_close(market: str, when_kst: dt.datetime, max_retry: int = 3, pause: float = 0.15) -> Optional[float]:
    params = {"market": market, "to": kst_to_utc_str(when_kst), "count": 1}
    for attempt in range(1, max_retry + 1):
        try:
            r = requests.get(UPBIT_MIN1_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data:
                time.sleep(pause)
                continue
            return float(data[0]["trade_price"])
        except Exception:
            if attempt == max_retry:
                print(f"[WARN] fetch failed at {when_kst} for {market}", file=sys.stderr)
            time.sleep(pause)
    return None


def daterange(start_date: dt.date, end_date: dt.date):
    d = start_date
    while d <= end_date:
        yield d
        d += dt.timedelta(days=1)


def pct_change(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a * 100.0


def build_daily_df(market: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    rows = []
    for d in daterange(start_date, end_date):
        # 비교 시점(00:00 / 08:50)은 KST 타임존 사용
        t0000 = dt.datetime.combine(d, dt.time(0, 0, tzinfo=KST))
        t0850 = dt.datetime.combine(d, dt.time(8, 50, tzinfo=KST))

        p0 = fetch_minute_close(market, t0000); time.sleep(0.12)
        p1 = fetch_minute_close(market, t0850)

        change_abs = None if (p0 is None or p1 is None) else (p1 - p0)
        change_pct = pct_change(p0, p1)

        weekday_kr = ["월","화","수","목","금","토","일"][d.weekday()]

        # ⬇️ 핵심 수정: date를 datetime64로 저장해 .dt 접근자 에러 제거
        rows.append({
            "date": pd.Timestamp(d),     # << 여기서 datetime64[ns]로 저장
            "weekday": weekday_kr,
            "price_kst_00_00": p0,
            "price_kst_08_50": p1,
            "change_abs": change_abs,
            "change_pct": None if change_pct is None else round(change_pct, 4),
        })
    return pd.DataFrame(rows)


def monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()

    # ⬇️ 방어적 변환 (혹시라도 object로 들어온 경우를 대비)
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"])

    tmp["ym"] = tmp["date"].dt.to_period("M").astype(str)

    def agg(g: pd.DataFrame) -> pd.Series:
        days = len(g)
        up_days = (g["change_abs"] > 0).sum()
        up_prob = round((up_days / days * 100), 2) if days else None
        avg_abs = round(g["change_abs"].mean(), 2) if days else None
        avg_pct = round(g["change_pct"].mean(), 2) if days else None

        # 최대/최소 상승폭 안전 처리
        valid = g.dropna(subset=["change_abs"])
        if not valid.empty:
            idx_up = valid["change_abs"].idxmax()
            idx_dn = valid["change_abs"].idxmin()
            max_up_date = valid.loc[idx_up, "date"].date().isoformat()
            max_up_abs  = int(valid.loc[idx_up, "change_abs"])
            max_dn_date = valid.loc[idx_dn, "date"].date().isoformat()
            max_dn_abs  = int(valid.loc[idx_dn, "change_abs"])
        else:
            max_up_date = None
            max_up_abs  = None
            max_dn_date = None
            max_dn_abs  = None

        return pd.Series({
            "days": days, "up_days": up_days, "up_prob_%": up_prob,
            "avg_change_abs": avg_abs, "avg_change_pct": avg_pct,
            "max_up_date": max_up_date, "max_up_abs": max_up_abs,
            "max_down_date": max_dn_date, "max_down_abs": max_dn_abs
        })

    return tmp.groupby("ym", as_index=False).apply(agg)


def top10_moves(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["date","weekday","price_kst_00_00","price_kst_08_50","change_abs","change_pct"]
    valid = df.dropna(subset=["change_abs"]).copy()
    top_up = valid.sort_values("change_abs", ascending=False).head(10)[cols].copy()
    top_dn = valid.sort_values("change_abs", ascending=True).head(10)[cols].copy()
    return top_up, top_dn


def save_excel(df: pd.DataFrame, market: str, start_date: dt.date, end_date: dt.date, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{market}_kst_0000_0850_ANALYSIS_{start_date}_{end_date}.xlsx"
    mon = monthly_stats(df)
    up, dn = top10_moves(df)
    with pd.ExcelWriter(fname, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Raw_Daily")
        mon.to_excel(w, index=False, sheet_name="Monthly_Stats")
        up.to_excel(w, index=False, sheet_name="Top10_Ups")
        dn.to_excel(w, index=False, sheet_name="Top10_Downs")
    return fname


def send_email_gmail(to_list: List[str], subject: str, body: str, attachments: List[Path]):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = ", ".join(to_list)
    msg.set_content(body)

    for p in attachments:
        data = p.read_bytes()
        msg.add_attachment(
            data,
            maintype="application",
            subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=p.name
        )

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as s:
        s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        s.send_message(msg)


# ====== Interactive inputs ======
def prompt_markets() -> List[str]:
    while True:
        s = input("코인 티커(쉼표로 구분, 예: KRW-SOL,KRW-ETC): ").strip().upper()
        parts = [x.strip() for x in s.split(",") if x.strip()]
        parts = [p if p.startswith("KRW-") else f"KRW-{p}" for p in parts]
        if parts:
            return parts
        print("값이 비었습니다. 다시 입력해주세요.")

def prompt_period() -> Tuple[dt.date, dt.date]:
    while True:
        mode = input("기간 입력 방식 선택: 1) 최근 N일  2) 시작~종료 날짜  (1/2): ").strip()
        today = dt.datetime.now(KST).date()
        if mode == "1":
            while True:
                n = input("최근 N일 (예: 30): ").strip()
                if n.isdigit() and int(n) > 0:
                    days = int(n)
                    end = today
                    start = end - dt.timedelta(days=days-1)
                    return start, end
                print("정수로 다시 입력해주세요.")
        elif mode == "2":
            s = input("시작 날짜 (YYYY-MM-DD): ").strip()
            e = input("종료 날짜 (YYYY-MM-DD): ").strip()
            try:
                start = dt.date.fromisoformat(s)
                end = dt.date.fromisoformat(e)
                if start > end:
                    print("시작 날짜가 종료 날짜보다 늦습니다. 다시 입력해주세요.")
                    continue
                return start, end
            except Exception:
                print("날짜 형식을 확인해주세요 (예: 2025-09-10).")
        else:
            print("1 또는 2를 선택해주세요.")


# ==== Main ====
def parse_args():
    ap = argparse.ArgumentParser(description="Upbit 다중코인 자동 분석 + Gmail 자동 발송")
    ap.add_argument("--markets", required=False, help="쉼표로 구분된 마켓 리스트 (예: KRW-SOL,KRW-ETC,KRW-MANA)")
    ap.add_argument("--days", type=int, default=None, help="최근 N일 (start/end 미지정 시에만 사용)")
    ap.add_argument("--start", type=str, default=None, help="시작 날짜 YYYY-MM-DD (KST)")
    ap.add_argument("--end", type=str, default=None, help="종료 날짜 YYYY-MM-DD (KST, 기본=오늘)")
    ap.add_argument("--to", type=str, default=DEFAULT_RECIPIENTS, help="수신자 이메일(쉼표로 여러 명)")
    ap.add_argument("--outdir", type=str, default="upbit_outputs", help="엑셀 저장 디렉토리")
    ap.add_argument("--interactive", action="store_true", help="프롬프트로 티커/기간을 입력받아 실행")
    return ap.parse_args()


def main():
    args = parse_args()

    # === 인터랙티브 모드 판단 ===
    interactive = args.interactive or (len(sys.argv) == 1)

    # === 티커 ===
    if interactive or not args.markets:
        markets = prompt_markets()
    else:
        markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
        markets = [m if m.startswith("KRW-") else f"KRW-{m}" for m in markets]
    if not markets:
        print("마켓이 비어 있습니다. 예: KRW-SOL,KRW-ETC")
        sys.exit(1)

    # === 기간 ===
    if interactive or (not args.days and not (args.start and args.end)):
        start_date, end_date = prompt_period()
    else:
        today = dt.datetime.now(KST).date()
        if args.start and args.end:
            start_date = dt.date.fromisoformat(args.start)
            end_date   = dt.date.fromisoformat(args.end)
        else:
            days = args.days if args.days is not None else 30
            end_date = today
            start_date = end_date - dt.timedelta(days=days-1)

    # === 수신자/출력 디렉토리 ===
    to_list = [x.strip() for x in (args.to or DEFAULT_RECIPIENTS).split(",") if x.strip()]
    out_dir = Path(args.outdir)

    print(f"[INFO] 대상: {markets}")
    print(f"[INFO] 기간: {start_date} ~ {end_date} (KST)")
    print(f"[INFO] 결과 디렉토리: {out_dir.resolve()}")
    attachments: List[Path] = []

    for m in markets:
        if not m.startswith("KRW-"):
            print(f"[WARN] {m} 는 KRW-마켓이 아님. 건너뜁니다.")
            continue
        print(f"[RUN] 수집/분석: {m}")
        df = build_daily_df(m, start_date, end_date)
        xlsx_path = save_excel(df, m, start_date, end_date, out_dir)
        print(f"[OK] 저장: {xlsx_path}")
        attachments.append(xlsx_path)

    # === 메일 자동 전송 ===
    subject = f"[Upbit] {start_date}~{end_date} KST 00:00 vs 08:50 분석 ({len(attachments)}개 종목)"
    body = "첨부된 엑셀 파일들을 확인하세요."
    print(f"[MAIL] Gmail 전송 → {to_list}")
    send_email_gmail(to_list, subject, body, attachments)
    print("[OK] 메일 전송 완료")


if __name__ == "__main__":
    main()