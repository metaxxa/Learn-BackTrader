"""
Advanced Dynamic Grid Bot v2.0
===============================
Фичи:
- GARCH(1,1) для прогноза волатильности
- Трейлинг сетки (Trailing Grid)
- Риск-менеджмент (Max DD, position limits)
- Telegram-уведомления
- Бэктест на истории
- SQLite логирование сделок

pip install ccxt pandas pandas_ta arch numpy sqlite3 python-telegram-bot httpx
"""

import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import sqlite3
import time
import logging
import asyncio
import httpx
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# ============ КОНФИГУРАЦИЯ ============
@dataclass
class Config:
    # Биржа
    exchange: str = "binance"
    api_key: str = ""
    api_secret: str = ""

    # Торговля
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    initial_capital: float = 1000.0
    order_size_pct: float = 0.02  # 2% от капитала на уровень

    # Сетка
    grid_levels: int = 7
    vol_method: str = "garch"  # "atr", "bollinger", "garch", "hybrid"
    vol_lookback: int = 300
    k_multiplier: float = 1.5

    # Риск-менеджмент
    max_drawdown_pct: float = 0.15  # Стоп при просадке 15%
    max_position_pct: float = 0.5   # Макс 50% капитала в позиции
    recalc_threshold: float = 0.10  # Пересчёт при изменении σ на 10%

    # Трейлинг
    trailing_enabled: bool = True
    trailing_activation: float = 0.02  # Активация после 2% движения

    # Режим
    dry_run: bool = True
    backtest: bool = False
    csv_path: str = ""
    commission: float = 0.001  # 0.1% по умолчанию
    telegram_token: str = ""
    telegram_chat_id: str = ""


# ============ МОДЕЛЬ РЫНКА ============
class VolatilityModel:
    """Модуль оценки и прогноза волатильности."""

    def __init__(self, method: str = "garch"):
        self.method = method

    def calculate(self, df: pd.DataFrame) -> float:
        """Возвращает прогнозируемую волатильность (в % цены)."""
        if self.method == "atr":
            return self._atr(df)
        elif self.method == "bollinger":
            return self._bollinger_width(df)
        elif self.method == "garch":
            return self._garch_forecast(df)
        elif self.method == "hybrid":
            return self._hybrid(df)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _atr(self, df: pd.DataFrame) -> float:
        atr = ta.atr(df["high"], df["low"], df["close"], length=14)
        return float(atr.iloc[-1])

    def _bollinger_width(self, df: pd.DataFrame) -> float:
        bb = ta.bbands(df["close"], length=20, std=2)
        upper = bb["BBU_20_2.0"].iloc[-1]
        lower = bb["BBL_20_2.0"].iloc[-1]
        return (upper - lower) / 4  # Примерный шаг

    def _garch_forecast(self, df: pd.DataFrame) -> float:
        """GARCH(1,1) прогноз на 1 период вперёд."""
        try:
            returns = df["close"].pct_change().dropna() * 100
            if len(returns) < 100:
                return self._atr(df)  # Fallback

            # Use rescale=True for better convergence
            model = arch_model(returns, p=1, q=1, dist='studentst', rescale=True)
            res = model.fit(disp='off', show_warning=False)
            forecast = res.forecast(horizon=1)
            var_forecast = forecast.variance.iloc[-1].values[0]
            vol_pct = np.sqrt(var_forecast)

            # Конвертируем % в абсолютное значение
            current_price = df["close"].iloc[-1]
            return vol_pct * current_price / 100
        except Exception as e:
            logging.warning(f"GARCH failed: {e}, fallback to ATR")
            return self._atr(df)

    def _hybrid(self, df: pd.DataFrame) -> float:
        """Комбинация ATR + GARCH с весами."""
        atr_val = self._atr(df)
        garch_val = self._garch_forecast(df)
        return 0.4 * atr_val + 0.6 * garch_val


# ============ УПРАВЛЕНИЕ РИСКАМИ ============
class RiskManager:
    def __init__(self, config: Config):
        self.config = config
        self.peak_equity = config.initial_capital
        self.current_equity = config.initial_capital

    def update_equity(self, equity: float):
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    @property
    def drawdown(self) -> float:
        if self.peak_equity == 0:
            return 0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def can_open_position(self, order_value: float, current_pos_value: float) -> bool:
        """Проверка лимитов перед открытием."""
        if self.drawdown > self.config.max_drawdown_pct:
            logging.error(f"🛑 Max DD breached: {self.drawdown:.2%}")
            return False

        total_val = abs(current_pos_value) + order_value
        if total_val / self.current_equity > self.config.max_position_pct:
            logging.warning(f"🛑 Position limit reached: {total_val/self.current_equity:.2%}")
            return False

        return True

    def should_stop(self) -> bool:
        return self.drawdown > self.config.max_drawdown_pct


# ============ ОСНОВНОЙ БОТ ============
@dataclass
class GridLevel:
    price: float
    side: str
    order_id: Optional[str] = None
    filled: bool = False
    timestamp: Optional[datetime] = None


class Backtester:
    def __init__(self, config: Config):
        self.config = config
        self.bot = AdvancedDynamicGridBot(config)

    def load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.csv_path)
        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df.rename(columns={'date': 'ts'}, inplace=True)
        df['ts'] = pd.to_datetime(df['ts'])
        return df

    async def run(self):
        df_full = self.load_csv()
        logging.info(f"📊 Backtest started | {len(df_full)} bars")

        # Initial lookback
        lookback = self.config.vol_lookback

        for i in range(lookback, len(df_full)):
            df_slice = df_full.iloc[i-lookback:i]
            current_bar = df_full.iloc[i]

            # For backtest, we check if high/low touched the grid levels
            high = current_bar.get('high', current_bar['close'])
            low = current_bar.get('low', current_bar['close'])
            price = current_bar['close']

            # Mock fetch_data for the bot
            self.bot.fetch_data = lambda: df_slice

            # Execution logic (same as bot.run but step-by-step)
            vol = self.bot.vol_model.calculate(df_slice)
            step = vol * self.config.k_multiplier

            await self.bot.check_fills(price, high=high, low=low)

            current_equity = self.bot.balance + (self.bot.position * price)
            self.bot.risk.update_equity(current_equity)

            if self.bot.risk.should_stop():
                logging.critical(f"🛑 Backtest STOP: Max drawdown @ bar {i}")
                break

            if self.bot.check_trailing(price, step):
                self.bot.grid_levels = self.bot.calculate_grid(self.bot.center_price, step)
                await self.bot.deploy_grid()
                continue

            if self.bot.should_recalc(vol):
                if self.bot.center_price == 0:
                    self.bot.center_price = price
                self.bot.grid_levels = self.bot.calculate_grid(self.bot.center_price, step)
                await self.bot.deploy_grid()
                self.bot.last_vol = vol

        final_equity = self.bot.balance + (self.bot.position * df_full.iloc[-1]['close'])
        logging.info(f"🏁 Backtest finished | Final Equity: ${final_equity:.2f} | PnL: {((final_equity/self.config.initial_capital)-1):.2%}")


class AdvancedDynamicGridBot:
    def __init__(self, config: Config):
        self.config = config
        self.vol_model = VolatilityModel(config.vol_method)
        self.risk = RiskManager(config)
        self.exchange = self._init_exchange()
        self.db = self._init_db()

        self.grid_levels: List[GridLevel] = []
        self.center_price: float = 0
        self.last_vol: Optional[float] = None
        self.trailing_high: float = 0
        self.trailing_low: float = float('inf')

        self.balance = config.initial_capital
        self.position = 0.0

        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s | %(levelname)s | %(message)s')

    def _init_exchange(self):
        params = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
        if not self.config.dry_run:
            params["apiKey"] = self.config.api_key
            params["secret"] = self.config.api_secret
        exchange_class = getattr(ccxt, self.config.exchange)
        return exchange_class(params)

    def _init_db(self):
        conn = sqlite3.connect("grid_bot.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, symbol TEXT, side TEXT,
                price REAL, amount REAL, pnl REAL
            )
        """)
        conn.commit()
        return conn

    async def fetch_data(self) -> pd.DataFrame:
        raw = await self.exchange.fetch_ohlcv(
            self.config.symbol, self.config.timeframe,
            limit=self.config.vol_lookback
        )
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df

    def calculate_grid(self, center: float, step: float) -> List[GridLevel]:
        """Построение сетки вокруг центра."""
        levels = []
        for i in range(1, self.config.grid_levels + 1):
            levels.append(GridLevel(
                price=self.format_price(center - i * step),
                side="buy", timestamp=datetime.now()
            ))
            levels.append(GridLevel(
                price=self.format_price(center + i * step),
                side="sell", timestamp=datetime.now()
            ))
        return sorted(levels, key=lambda l: l.price)

    def log_trade(self, side: str, price: float, amount: float, pnl: float = 0):
        try:
            self.db.execute(
                "INSERT INTO trades (timestamp, symbol, side, price, amount, pnl) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), self.config.symbol, side, price, amount, pnl)
            )
            self.db.commit()
        except Exception as e:
            logging.error(f"DB error: {e}")

    async def place_order(self, level: GridLevel) -> Optional[str]:
        amount = self.format_amount((self.config.initial_capital * self.config.order_size_pct) / level.price)

        current_pos_value = self.position * level.price # Approximate
        if not self.risk.can_open_position(amount * level.price, current_pos_value):
            return None

        if self.config.dry_run:
            logging.info(f"[DRY] {level.side.upper()} {amount:.6f} @ {level.price}")
            return f"sim_{int(time.time()*1000)}"

        try:
            order = await self.exchange.create_limit_order(
                self.config.symbol, level.side, amount, level.price
            )
            return order["id"]
        except Exception as e:
            logging.error(f"Order error: {e}")
            return None

    async def cancel_all(self):
        if self.config.dry_run:
            return
        try:
            orders = await self.exchange.fetch_open_orders(self.config.symbol)
            for o in orders:
                await self.exchange.cancel_order(o["id"], self.config.symbol)
        except Exception as e:
            logging.error(f"Cancel error: {e}")

    async def load_markets(self):
        if not self.config.dry_run and not self.config.backtest:
            await self.exchange.load_markets()

    def format_price(self, price: float) -> float:
        if self.config.dry_run or self.config.backtest:
            return round(price, 2)
        return float(self.exchange.price_to_precision(self.config.symbol, price))

    def format_amount(self, amount: float) -> float:
        if self.config.dry_run or self.config.backtest:
            return round(amount, 6)
        return float(self.exchange.amount_to_precision(self.config.symbol, amount))

    async def deploy_grid(self):
        for lvl in self.grid_levels:
            lvl.order_id = await self.place_order(lvl)
            if not self.config.backtest:
                await asyncio.sleep(0.2)

    async def notify(self, message: str):
        """Отправка уведомления в Telegram."""
        logging.info(f"📢 Notification: {message}")
        if not self.config.telegram_token or not self.config.telegram_chat_id:
            return

        url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        try:
            async with httpx.AsyncClient() as client:
                await client.post(url, json=payload, timeout=10.0)
        except Exception as e:
            logging.error(f"Telegram error: {e}")

    async def check_fills(self, current_price: float, high: Optional[float] = None, low: Optional[float] = None):
        if not self.config.dry_run and not self.config.backtest:
            # For live trading, fetch all open orders once to reduce API calls
            try:
                open_orders = await self.exchange.fetch_open_orders(self.config.symbol)
                open_ids = [o['id'] for o in open_orders]
            except Exception as e:
                logging.error(f"Error fetching open orders: {e}")
                return

        for lvl in self.grid_levels:
            if lvl.filled or not lvl.order_id:
                continue

            filled = False
            fill_price = lvl.price

            if self.config.dry_run or self.config.backtest:
                # In backtest we can use high/low for better accuracy
                if self.config.backtest and high is not None and low is not None:
                    if lvl.side == "buy" and low <= lvl.price:
                        filled = True
                    elif lvl.side == "sell" and high >= lvl.price:
                        filled = True
                else:
                    if (lvl.side == "buy" and current_price <= lvl.price) or \
                       (lvl.side == "sell" and current_price >= lvl.price):
                        filled = True
            else:
                if lvl.order_id not in open_ids:
                    # Order is no longer open, check if it was closed (filled)
                    try:
                        order = await self.exchange.fetch_order(lvl.order_id, self.config.symbol)
                        if order['status'] == 'closed':
                            filled = True
                            fill_price = order['average'] or order['price']
                    except Exception as e:
                        logging.error(f"Error fetching order {lvl.order_id}: {e}")

            if filled:
                lvl.filled = True
                amount = self.format_amount((self.config.initial_capital * self.config.order_size_pct) / fill_price)

                fee = amount * fill_price * self.config.commission if self.config.backtest or self.config.dry_run else 0

                if lvl.side == "buy":
                    self.balance -= (amount * fill_price + fee)
                    self.position += amount
                else:
                    self.balance += (amount * fill_price - fee)
                    self.position -= amount

                # Update Risk Manager
                current_equity = self.balance + (self.position * current_price)
                self.risk.update_equity(current_equity)

                self.log_trade(lvl.side, fill_price, amount)
                await self.notify(f"✅ {lvl.side.upper()} FILLED: {amount:.6f} @ {fill_price} | Equity: ${current_equity:.2f}")

    def check_trailing(self, current_price: float, step: float) -> bool:
        """Трейлинг сетки: сдвиг при пробое границ."""
        if not self.config.trailing_enabled:
            return False

        self.trailing_high = max(self.trailing_high, current_price)
        self.trailing_low = min(self.trailing_low, current_price)

        upper_boundary = self.center_price + self.config.grid_levels * step
        lower_boundary = self.center_price - self.config.grid_levels * step

        activation_move = self.center_price * self.config.trailing_activation

        # Цена ушла вверх — сдвигаем сетку вверх
        if current_price > upper_boundary and (current_price - self.center_price) > activation_move:
            logging.info(f"⬆️ TRAILING UP: {self.center_price:.2f} → {current_price:.2f}")
            self.center_price = current_price
            self.trailing_high = current_price
            return True

        # Цена ушла вниз — сдвигаем сетку вниз
        if current_price < lower_boundary and (self.center_price - current_price) > activation_move:
            logging.info(f"⬇️ TRAILING DOWN: {self.center_price:.2f} → {current_price:.2f}")
            self.center_price = current_price
            self.trailing_low = current_price
            return True

        return False

    def should_recalc(self, new_vol: float) -> bool:
        if self.last_vol is None:
            return True
        change = abs(new_vol - self.last_vol) / self.last_vol
        return change >= self.config.recalc_threshold

    async def run(self):
        logging.info(f"🚀 Bot started | {self.config.symbol} | {self.config.vol_method}")
        logging.info(f"   Capital: ${self.config.initial_capital} | Levels: {self.config.grid_levels}")

        await self.load_markets()
        self.risk.update_equity(self.config.initial_capital)

        while True:
            try:
                df = await self.fetch_data()
                price = float(df["close"].iloc[-1])
                vol = self.vol_model.calculate(df)
                step = vol * self.config.k_multiplier

                # Check for filled orders
                await self.check_fills(price)

                # Update Risk Manager
                current_equity = self.balance + (self.position * price)
                self.risk.update_equity(current_equity)

                # Проверка риск-менеджера
                if self.risk.should_stop():
                    await self.notify("🛑 STOP: Max drawdown reached. Closing positions...")
                    await self.cancel_all()
                    if abs(self.position) > 0:
                        side = "sell" if self.position > 0 else "buy"
                        await self.place_order(GridLevel(price=price, side=side))
                    break

                # Трейлинг
                if self.check_trailing(price, step):
                    await self.cancel_all()
                    self.grid_levels = self.calculate_grid(self.center_price, step)
                    await self.deploy_grid()
                    await self.notify(f"🔄 Grid trailed to {self.center_price:.2f}")
                    continue

                # Пересчёт при смене волатильности
                if self.should_recalc(vol):
                    logging.info(f"🔄 Vol changed: {self.last_vol} → {vol:.2f}")
                    if self.center_price == 0:
                        self.center_price = price
                    await self.cancel_all()
                    self.grid_levels = self.calculate_grid(self.center_price, step)
                    await self.deploy_grid()
                    self.last_vol = vol
                    await self.notify(f"🔄 Grid redeployed. Vol: {vol:.2f}, Step: {step:.2f}")

                await asyncio.sleep(60)

            except KeyboardInterrupt:
                logging.info("⛔ Manual stop")
                await self.cancel_all()
                await self.exchange.close()
                break
            except Exception as e:
                logging.error(f"Loop error: {e}")
                await asyncio.sleep(30)


if __name__ == "__main__":
    import sys

    mode = "live"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == "backtest":
        cfg = Config(
            initial_capital=100000,
            vol_method="garch",
            grid_levels=5,
            k_multiplier=2.0,
            backtest=True,
            csv_path="GAZP_D1.csv",
            dry_run=True
        )
        tester = Backtester(cfg)
        asyncio.run(tester.run())
    else:
        cfg = Config(
            symbol="BTC/USDT",
            timeframe="1h",
            initial_capital=1000,
            vol_method="hybrid",  # GARCH + ATR
            grid_levels=7,
            k_multiplier=1.5,
            trailing_enabled=True,
            dry_run=True
        )
        bot = AdvancedDynamicGridBot(cfg)
        asyncio.run(bot.run())
