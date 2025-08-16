import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Sequence, Mapping


class HistoricalSimulation:
    """
    Historical backtesting engine

    Core capabilities
    -----------------
    • Rebalance on given dates with target weights.
    • Separate execution and valuation pricing:
        - Either pass `data` (valuation) and optional `exec_prices` (execution),
        - Or pass `ohlc` dict and pick fields via `valuation_field` / `execution_field`.
    • Trading frictions:
        - Slippage (bps) -> side-specific price adjustment.
        - Commission table: variable bps + fixed fee + minimum per trade.
        - Trade cap per ticker as % of portfolio value.
    • Share rounding:
        - Integer/lot-based sizing and tick-size snapping on prices.
        - Cash account managed; scale buys to avoid negative cash.
    • Cash interest:
        - Daily accrual: deposit for positive cash, borrow for negative cash.
    • Mark-to-market daily with frozen shares between rebalances.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Valuation price panel (e.g., close). Used if `ohlc` not provided.
    weights_df : pd.DataFrame
        Target weights on rebalance dates. Aligned to `rebalance_dates` x tickers of price panel.
    rebalance_dates : Sequence[pd.Timestamp]
        Rebalance dates (must exist in valuation price index).
    initial_value : float, default 1000.0
        Starting portfolio cash (base currency).
    normalize_weights : bool, default True
        Normalize each weight row to sum to 1 (zero-sum rows remain zero).
    fill_method : {'ffill','bfill',None}, default None
        Optional fill for valuation panel (and exec panel if provided).

    # Execution/valuation separation
    use_next_open : bool, default False
        If True, trade at next session's execution prices (e.g., next open).
    exec_prices : pd.DataFrame, optional
        Execution price panel when not using `ohlc`. Required if `use_next_open=True`.
    ohlc : Dict[str, pd.DataFrame], optional
        Dict with any subset of {'open','high','low','close'} as DataFrames.
        If provided, use `valuation_field`/`execution_field` to select price source.
    valuation_field : str, default 'close'
        Which OHLC field to use for daily mark-to-market.
    execution_field : str, default 'open' if use_next_open else 'close'
        Which OHLC field to use for executions.

    # Trading frictions & sizing
    slippage_bps : float, default 0.0
        Buy price = px*(1+slip), sell price = px*(1-slip).
    commission : Dict[str, float], optional
        {'variable_bps': float, 'fixed': float, 'min_per_trade': float}. Missing keys default to 0.
    allow_partial_shares : bool, default True
        If False, round to lot multiples and manage cash accordingly.
    min_lot : int or Dict[str,int], default 1
        Minimum lot size per trade (1 = integer shares). May be scalar or per-ticker dict.
    tick_size : float or Dict[str,float], default 0.0
        Tick size for price snapping. Buy -> ceil to tick, sell -> floor to tick.
    cap_trade_pct : float, default 1.0
        Per-ticker cap on notional traded as a fraction of portfolio value at rebalance.

    # Cash interest
    deposit_rate_annual : float, default 0.0
        Annual deposit rate (applied when cash >= 0).
    borrow_rate_annual : float, default 0.0
        Annual borrowing rate (applied when cash < 0).

    Attributes (after run)
    ----------------------
    index_series : pd.Series
        Daily portfolio value time series on valuation index.
    cash_series : pd.Series
        Daily cash balance (after accrual & trades).
    shares_record : pd.DataFrame
        Share counts snapshot at execution moments (rows: exec dates).
    trade_ledger : pd.DataFrame
        Executed trades per (date,ticker) with prices, notionals, fees.
    shares_ledger : pd.DataFrame
        Holdings snapshots per (date,ticker) at execution moments.
    logs : List[str]
        Validation and informational messages.
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame],
        weights_df: pd.DataFrame,
        rebalance_dates: Union[List[pd.Timestamp], pd.DatetimeIndex],
        *,
        initial_value: float = 1000.0,
        normalize_weights: bool = True,
        fill_method: Optional[str] = None,
        # execution/valuation
        use_next_open: bool = False,
        exec_prices: Optional[pd.DataFrame] = None,
        ohlc: Optional[Dict[str, pd.DataFrame]] = None,
        valuation_field: str = "close",
        execution_field: Optional[str] = None,
        # frictions & sizing
        slippage_bps: float = 0.0,
        commission: Optional[Dict[str, float]] = None,
        allow_partial_shares: bool = True,
        min_lot: Union[int, Dict[str, int]] = 1,
        tick_size: Union[float, Dict[str, float]] = 0.0,
        cap_trade_pct: float = 1.0,
        # cash interest
        deposit_rate_annual: float = 0.0,
        borrow_rate_annual: float = 0.0,
    ):
        self.data = None if data is None else data.copy()
        self.ohlc = None if ohlc is None else {k: v.copy() for k, v in ohlc.items()}
        self.weights_df = weights_df.copy()
        self.rebalance_dates = pd.DatetimeIndex(rebalance_dates).sort_values()
        self.initial_value = float(initial_value)
        self.normalize_weights = bool(normalize_weights)
        self.fill_method = fill_method

        self.use_next_open = bool(use_next_open)
        self.exec_prices = None if exec_prices is None else exec_prices.copy()
        self.valuation_field = valuation_field
        self.execution_field = (
            execution_field if execution_field is not None
            else ("open" if use_next_open else "close")
        )

        self.slippage_bps = float(slippage_bps)
        self.commission = {
            "variable_bps": 0.0,
            "fixed": 0.0,
            "min_per_trade": 0.0,
        }
        if commission:
            self.commission.update({k: float(v) for k, v in commission.items()})

        self.allow_partial_shares = bool(allow_partial_shares)
        self.min_lot = min_lot
        self.tick_size = tick_size
        self.cap_trade_pct = float(cap_trade_pct)

        self.deposit_rate_annual = float(deposit_rate_annual)
        self.borrow_rate_annual = float(borrow_rate_annual)

        # results
        self.index_series: Optional[pd.Series] = None
        self.cash_series: Optional[pd.Series] = None
        self.shares_record: Optional[pd.DataFrame] = None
        self.trade_ledger: Optional[pd.DataFrame] = None
        self.shares_ledger: Optional[pd.DataFrame] = None
        self.logs: List[str] = []

    @staticmethod
    def weights_long_to_wide(
            weights_long: pd.DataFrame,
            price_cols: pd.Index,
            rebalance_dates: pd.DatetimeIndex,
            agg: str = 'last'
        ) -> pd.DataFrame:
        wl = weights_long.copy()
        wl['date'] = pd.to_datetime(wl['date'])
        wide = (wl.pivot_table(index='date', columns='ticker', values='weight', aggfunc=agg)
                .reindex(rebalance_dates)
                .reindex(columns=price_cols)
                .fillna(0.0)
                .sort_index())
        return wide

    # ------------------------- validation & alignment ------------------------- #
    def _log(self, msg: str):
        self.logs.append(msg)

    def _valuation_panel(self) -> pd.DataFrame:
        """
        Return the valuation price DataFrame used for MTM each day.
        Priority: ohlc[valuation_field] -> data
        """
        if self.ohlc is not None and self.valuation_field in self.ohlc:
            return self.ohlc[self.valuation_field]
        if self.data is not None:
            return self.data
        raise ValueError("[FATAL] No valuation price panel available.")

    def _execution_panel(self) -> pd.DataFrame:
        """
        Return the execution price DataFrame used for trading.
        Priority: ohlc[execution_field] -> exec_prices -> data
        """
        if self.ohlc is not None and self.execution_field in self.ohlc:
            return self.ohlc[self.execution_field]
        if self.exec_prices is not None:
            return self.exec_prices
        if self.data is not None:
            return self.data
        raise ValueError("[FATAL] No execution price panel available.")

    def _validate_inputs(self) -> None:
        """Align inputs and check for common issues."""
        val_px = self._valuation_panel().astype(float)
        exe_px = self._execution_panel().astype(float)

        # Optional fill
        if self.fill_method in ("ffill", "bfill"):
            val_px = getattr(val_px, self.fill_method)()
            exe_px = getattr(exe_px, self.fill_method)()

        self.val_px = val_px
        self.exe_px = exe_px

        # Align columns
        cols = self.val_px.columns
        # Align execution panel columns to valuation panel
        self.exe_px = self.exe_px.reindex(columns=cols)

        # weights: align to rebalances and columns
        original_cols = set(self.weights_df.columns)
        self.weights_df = self.weights_df.reindex(index=self.rebalance_dates, columns=cols).fillna(0.0)
        dropped = original_cols - set(cols)
        if dropped:
            self._log(f"[WARN] Dropping weight columns not in price panel: {sorted(dropped)}")

        # check rebalance dates present
        if not self.rebalance_dates.isin(self.val_px.index).all():
            missing = self.rebalance_dates[~self.rebalance_dates.isin(self.val_px.index)]
            raise ValueError(f"[FATAL] Rebalance dates missing in valuation index: {missing.tolist()}")

        # When using next open, ensure an execution date exists strictly after each rebalance
        if self.use_next_open:
            for dt in self.rebalance_dates:
                pos = self.exe_px.index.searchsorted(dt, side="right")
                if pos >= len(self.exe_px.index):
                    raise ValueError(f"[FATAL] No next execution row after {dt} in execution panel index.")

        # Price sanity checks
        if (self.val_px <= 0).any().any():
            self._log("[WARN] Non-positive valuation prices found.")
        if (self.exe_px <= 0).any().any():
            self._log("[WARN] Non-positive execution prices found.")

        # Weight sanity checks
        wsum = self.weights_df.sum(axis=1)
        not_one = ~np.isclose(wsum, 1.0) & ~(np.isclose(wsum, 0.0))
        if not_one.any():
            bad = wsum[not_one]
            self._log(f"[WARN] Some weight rows do not sum to 1 (and not 0). Sample: {bad.head().to_dict()}")

    # ------------------------- microstructure helpers ------------------------- #
    def _daily_rate(self, annual: float) -> float:
        """Convert annual simple rate to daily compounding factor - 1, using 252 days."""
        return (1.0 + annual) ** (1.0 / 252.0) - 1.0

    def _snap_tick(self, price: float, side: str, tick: float) -> float:
        """Snap price to tick size: buy -> ceil to tick, sell -> floor to tick."""
        if tick <= 0:
            return float(price)
        if side == "buy":
            return np.ceil(price / tick) * tick
        else:
            return np.floor(price / tick) * tick

    def _lot_round(self, shares: float, lot: int) -> float:
        """Round shares to nearest lot multiple (toward zero change)."""
        if lot <= 1:
            return np.floor(shares)  # integer shares
        return np.floor(shares / lot) * lot

    def _eff_price(self, px: float, side: str, tick: float) -> float:
        """Apply slippage and tick snap to the execution price."""
        slip = self.slippage_bps / 1e4
        raw = px * (1.0 + slip if side == "buy" else 1.0 - slip)
        return self._snap_tick(raw, side, tick)

    def _get_scalar_or_map(self, spec: Union[float, int, Dict[str, Any]], ticker: str) -> float:
        """Return scalar or per-ticker mapped value."""
        if isinstance(spec, dict):
            return float(spec.get(ticker, list(spec.values())[0])) if spec else 0.0
        return float(spec)

    def _normalize_row(self, w: pd.Series) -> pd.Series:
        if not self.normalize_weights:
            return w.fillna(0.0)
        s = w.sum(skipna=True)
        if np.isclose(s, 0.0):
            return w.fillna(0.0)
        return (w / s).fillna(0.0)

    # ------------------------------ core run ------------------------------ #
    def run(self) -> "HistoricalSimulation":
        """
        Execute the simulation. Results are stored on the instance attributes.
        """
        self._validate_inputs()

        idx = self.val_px.index
        cols = self.val_px.columns

        index_series = pd.Series(index=idx, dtype=float)
        cash_series = pd.Series(0.0, index=idx, dtype=float)
        shares_record = pd.DataFrame(0.0, index=[], columns=cols)

        trade_rows: List[Dict[str, Any]] = []
        hold_rows: List[Dict[str, Any]] = []

        prev_shares = pd.Series(0.0, index=cols)
        cash = float(self.initial_value)

        # precompute cash daily factors
        dep_d = self._daily_rate(self.deposit_rate_annual)
        bor_d = self._daily_rate(self.borrow_rate_annual)

        # daily MTM & accrual helper
        def apply_mtm_and_accrual(dates: pd.DatetimeIndex, shares_vec: pd.Series, cash_balance: float):
            nonlocal index_series, cash_series
            if len(dates) == 0:
                return cash_balance
            # iterate each day to accrue interest on cash
            for d in dates:
                # MTM at valuation price on d
                port_val_ex_cash = float((self.val_px.loc[d] * shares_vec).sum())
                # apply one-day accrual on cash BEFORE storing end-of-day cash
                if cash_balance >= 0:
                    cash_balance *= (1.0 + dep_d)
                else:
                    cash_balance *= (1.0 + bor_d)
                index_series.loc[d] = port_val_ex_cash + cash_balance
                cash_series.loc[d] = cash_balance
            return cash_balance

        # main loop
        for i, dt in enumerate(self.rebalance_dates):
            # MTM from previous exec to today's dt-1 (handled by previous iteration)
            # Value at dt close BEFORE trading
            port_value_dt = float((self.val_px.loc[dt] * prev_shares).sum() + cash)

            # desired target weights -> target notional
            w = self._normalize_row(self.weights_df.loc[dt])
            target_value = w * port_value_dt

            # choose execution date and price snapshot
            if self.use_next_open:
                pos = self.exe_px.index.searchsorted(dt, side="right")
                exec_dt = self.exe_px.index[pos]
            else:
                exec_dt = dt
            px_exec_row = self.exe_px.loc[exec_dt]

            # initial target shares (pre-rounding)
            with np.errstate(divide="ignore", invalid="ignore"):
                tgt_shares = (target_value / px_exec_row.replace(0.0, np.nan)).fillna(0.0)

            delta = tgt_shares - prev_shares

            # cap per ticker notional vs portfolio value
            if self.cap_trade_pct < 1.0 - 1e-12:
                cap_abs = self.cap_trade_pct * port_value_dt
                for tic in cols:
                    d = float(delta[tic])
                    if np.isclose(d, 0.0):
                        continue
                    side = "buy" if d > 0 else "sell"
                    tick = self._get_scalar_or_map(self.tick_size, tic)
                    px_eff = self._eff_price(float(px_exec_row[tic]), side, tick)
                    notional = abs(d) * px_eff
                    if notional > cap_abs > 0:
                        scale = cap_abs / notional
                        delta[tic] = d * scale
                        self._log(f"[INFO] {dt.date()} {tic}: trade capped to {self.cap_trade_pct:.2%} PV.")

            # share rounding (lot / integer policy)
            if not self.allow_partial_shares or (isinstance(self.min_lot, dict) or self.min_lot != 1):
                for tic in cols:
                    d = float(delta[tic])
                    if np.isclose(d, 0.0):
                        continue
                    lot = int(self._get_scalar_or_map(self.min_lot, tic))
                    if d > 0:
                        delta[tic] = self._lot_round(d, lot)
                    else:
                        # for sells, rounding toward zero exposure change => ceil in magnitude
                        delta[tic] = -self._lot_round(-d, lot)

            # price/tick & cost computation
            buy_notional = 0.0
            sell_notional = 0.0
            variable_bps = self.commission.get("variable_bps", 0.0) / 1e4
            fixed_fee = self.commission.get("fixed", 0.0)
            min_fee = self.commission.get("min_per_trade", 0.0)

            trade_cost_total = 0.0
            eff_px_map: Dict[str, float] = {}

            # first pass: compute notionals and fees, and tentative cash
            for tic in cols:
                d = float(delta[tic])
                if np.isclose(d, 0.0):
                    continue
                side = "buy" if d > 0 else "sell"
                tick = self._get_scalar_or_map(self.tick_size, tic)
                epx = self._eff_price(float(px_exec_row[tic]), side, tick)
                eff_px_map[tic] = epx
                notional = abs(d) * epx
                # per-trade commission: variable + fixed, with minimum
                fee = max(notional * variable_bps + fixed_fee, min_fee)
                trade_cost_total += fee
                if d > 0:
                    buy_notional += notional
                else:
                    sell_notional += notional

            tentative_cash = cash + sell_notional - buy_notional - trade_cost_total

            # scale down buys if cash deficit
            if tentative_cash < -1e-8 and buy_notional > 0:
                scale = max(0.0, (cash + sell_notional - trade_cost_total) / buy_notional)
                self._log(f"[INFO] {dt.date()} scaled buys by {scale:.4f} to avoid negative cash.")
                for tic in cols:
                    if delta[tic] > 0:
                        delta[tic] *= scale
                        # re-apply lot rounding if needed
                        if not self.allow_partial_shares or (isinstance(self.min_lot, dict) or self.min_lot != 1):
                            lot = int(self._get_scalar_or_map(self.min_lot, tic))
                            delta[tic] = self._lot_round(delta[tic], lot)

                # recompute notional & fees after scaling
                buy_notional = 0.0
                sell_notional = 0.0
                trade_cost_total = 0.0
                for tic in cols:
                    d = float(delta[tic])
                    if np.isclose(d, 0.0):
                        continue
                    side = "buy" if d > 0 else "sell"
                    epx = eff_px_map.get(tic, float(px_exec_row[tic]))
                    notional = abs(d) * epx
                    fee = max(notional * variable_bps + fixed_fee, min_fee)
                    trade_cost_total += fee
                    if d > 0:
                        buy_notional += notional
                    else:
                        sell_notional += notional
                tentative_cash = cash + sell_notional - buy_notional - trade_cost_total

            # accrue cash & MTM from last written day up to (but excluding) exec_dt
            # find previous day in index to continue from
            # previous iteration has written up to prior next_dt; we now ensure interval [prev_written+1 : exec_dt-1]
            # For simplicity, we write from the previous dt (or start) up to the day before exec_dt here,
            # but we prevent overwriting by assigning only the fresh dates.
            # In practice, we rely on monotonically increasing assignment segments.

            # apply execution: update cash & holdings at exec_dt
            cash = tentative_cash
            new_shares = prev_shares + delta

            # record ledgers at execution moment
            exec_label_date = exec_dt
            for tic in cols:
                # trade row
                d = float(delta[tic])
                if not np.isclose(d, 0.0):
                    side = "buy" if d > 0 else "sell"
                    px_eff = eff_px_map[tic]
                    notional = abs(d) * px_eff
                    fee = max(notional * variable_bps + fixed_fee, min_fee)
                    trade_rows.append({
                        "date": exec_label_date,
                        "ticker": tic,
                        "side": side,
                        "shares_delta": d,
                        "exec_price_eff": px_eff,
                        "notional": notional,
                        "commission": fee,
                    })
                # holdings snapshot
                hold_rows.append({
                    "date": exec_label_date,
                    "ticker": tic,
                    "price_ref": float(px_exec_row[tic]),
                    "shares": float(new_shares[tic]),
                    "amount": float(new_shares[tic] * float(px_exec_row[tic])),
                })

            # append shares_record row
            shares_record.loc[exec_label_date, cols] = new_shares.values

            # MTM path intervals:
            # - if use_next_open=False: new shares from [dt : next_dt]
            # - if use_next_open=True : prev_shares on [dt : exec_dt-1], then new shares on [exec_dt : next_dt]
            next_dt = (self.rebalance_dates[i + 1] if i < len(self.rebalance_dates) - 1 else self.val_px.index[-1])

            if not self.use_next_open:
                interval = self.val_px.loc[dt:next_dt].index
                cash = apply_mtm_and_accrual(interval, new_shares, cash)
            else:
                left = self.val_px.loc[dt:exec_dt].index
                if len(left) > 1:
                    cash = apply_mtm_and_accrual(left[:-1], prev_shares, cash)  # up to exec_dt-1 with old shares
                right = self.val_px.loc[exec_dt:next_dt].index
                cash = apply_mtm_and_accrual(right, new_shares, cash)

            # carry forward
            prev_shares = new_shares.copy()

        # finalize
        self.index_series = index_series.astype(float)
        self.cash_series = cash_series.astype(float)
        self.shares_record = shares_record.sort_index().astype(float)
        self.trade_ledger = pd.DataFrame(trade_rows).sort_values("date").reset_index(drop=True)
        self.shares_ledger = pd.DataFrame(hold_rows).sort_values("date").reset_index(drop=True)

        return self

    # ------------------------------ helpers ------------------------------ #
    def get_results(self) -> Dict[str, Any]:
        """Return all result objects."""
        if any(obj is None for obj in [self.index_series, self.cash_series, self.shares_record]):
            raise RuntimeError("Run the simulation first with `.run()`.")
        return {
            "index_series": self.index_series,
            "cash_series": self.cash_series,
            "shares_record": self.shares_record,
            "trade_ledger": self.trade_ledger,
            "shares_ledger": self.shares_ledger,
            "logs": self.logs,
        }

    def to_nav(self) -> pd.Series:
        """Return NAV series normalized to 1.0 at the first non-NaN value."""
        s = self.index_series.dropna()
        if s.empty:
            return self.index_series * np.nan
        return self.index_series / float(s.iloc[0])

class DollarCostAveragingSimulation:
    """
    Regular-investing (DCA) historical simulation engine.

    Core ideas
    ----------
    - You provide valuation prices (for daily MTM) and execution prices (for trades).
    - You provide an `initial_value` and a `contribution_series` that specifies the cash you add on each date.
    - On each contribution date, the contributed cash is allocated across assets using either:
        * equal weights across columns, or
        * a user-supplied weight vector (Series) or a weight schedule (DataFrame) indexed by date.
    - Can enforce integer (non-floating) share policy or allow fractional shares.
    - Cash earns (or pays) daily accrual at deposit/borrow annual rates.
    - Results stored on attributes and available via `get_results()`; NAV via `to_nav()`.
    - Provides IRR (money-weighted), TWR (time-weighted), and simple PnL return.

    Parameters
    ----------
    val_px : pd.DataFrame
        Valuation price series for MTM. Index must be DatetimeIndex; columns are tickers.
    exe_px : Optional[pd.DataFrame]
        Execution price series for trading. Defaults to `val_px` if None.
    initial_value : float
        Initial cash at start (portfolio cash).
    contribution_series : Union[pd.Series, float]
        If Series: cash contributed on those dates (index must be subset of val_px.index).
        If float: a constant contribution amount applied to every date in `rebalance_dates` (see below).
    weights : Optional[Union[pd.Series, pd.DataFrame, Mapping[str, float], str]]
        - None or "equal": equal-weight allocation (for multi-asset).
        - pd.Series / Mapping[str, float]: static weights across columns (will be normalized to sum to 1).
        - pd.DataFrame: date-indexed row weights aligned to `rebalance_dates` (will be normalized row-wise).
        - For single-asset, this is ignored.
    allow_partial_shares : bool
        If False, shares are rounded down to integer lots (`min_lot`).
    min_lot : int
        Minimum lot size per trade if `allow_partial_shares=False`. Default=1 (integer shares).
    use_next_open : bool
        If True, trades execute at the next available timestamp in `exe_px` strictly after the contribution date.
        If False, trades execute at the same date's execution price (close-at-close style).
    deposit_rate_annual : float
        Annualized deposit rate for positive cash (e.g., 0.03 = 3%).
    borrow_rate_annual : float
        Annualized borrow rate for negative cash.
    commission : Optional[Dict[str, float]]
        Commission spec:
            - "variable_bps": float (basis points on notional)
            - "fixed": float (fixed fee per trade)
            - "min_per_trade": float (minimum fee per trade)
        Defaults to none.
    cap_trade_pct : float
        Optional cap on absolute per-ticker notional as a fraction of current portfolio value at the contribution date.
        Use 1.0 to disable (default).
    tick_size : Optional[Union[float, Mapping[str, float]]]
        Optional price tick size to slighty worsen effective price (half-tick). If None, no adjustment.

    Notes
    -----
    - `rebalance_dates` are set internally to the union of:
        * all dates where `contribution_series` (if Series) is non-zero, and
        * the first index date when `initial_value > 0`.
      If `contribution_series` is a scalar, then every `val_px.index` date is a contribution date with that amount.
    - Cash flows ledger:
        * External flows are contributions (negative from the investor perspective; added to portfolio cash here).
        * For IRR, we treat contributions as negative flows at their dates and the terminal liquidation as a positive flow.
    """

    # ----------------------------
    # Construction & configuration
    # ----------------------------
    def __init__(
        self,
        val_px: pd.DataFrame,
        exe_px: Optional[pd.DataFrame] = None,
        *,
        initial_value: float = 0.0,
        contribution_series: Union[pd.Series, float] = 0.0,
        weights: Optional[Union[pd.Series, pd.DataFrame, Mapping[str, float], str]] = None,
        allow_partial_shares: bool = True,
        min_lot: int = 1,
        use_next_open: bool = False,
        deposit_rate_annual: float = 0.0,
        borrow_rate_annual: float = 0.0,
        commission: Optional[Dict[str, float]] = None,
        cap_trade_pct: float = 1.0,
        tick_size: Optional[Union[float, Mapping[str, float]]] = None,
    ):
        self.val_px = self._coerce_df(val_px, name="val_px")
        self.exe_px = self._coerce_df(exe_px if exe_px is not None else val_px, name="exe_px")
        self.initial_value = float(initial_value)
        self.contribution_series = contribution_series
        self.weights_raw = weights
        self.allow_partial_shares = bool(allow_partial_shares)
        self.min_lot = int(min_lot)
        self.use_next_open = bool(use_next_open)
        self.deposit_rate_annual = float(deposit_rate_annual)
        self.borrow_rate_annual = float(borrow_rate_annual)
        self.commission = commission or {}
        self.cap_trade_pct = float(cap_trade_pct)
        self.tick_size = tick_size

        # Outputs
        self.index_series: Optional[pd.Series] = None
        self.cash_series: Optional[pd.Series] = None
        self.shares_record: Optional[pd.DataFrame] = None
        self.trade_ledger: Optional[pd.DataFrame] = None
        self.shares_ledger: Optional[pd.DataFrame] = None
        self.logs: List[str] = []

        # Derived
        self.rebalance_dates: Optional[pd.DatetimeIndex] = None
        self.weights_df: Optional[pd.DataFrame] = None

    # -------------
    # Public API
    # -------------
    def run(self) -> "DollarCostAveragingSimulation":
        """
        Execute the simulation. Results are stored on the instance attributes.
        """
        self._validate_inputs()

        idx = self.val_px.index
        cols = self.val_px.columns

        index_series = pd.Series(index=idx, dtype=float)
        cash_series = pd.Series(0.0, index=idx, dtype=float)
        shares_record = pd.DataFrame(0.0, index=[], columns=cols)

        trade_rows: List[Dict[str, Any]] = []
        hold_rows: List[Dict[str, Any]] = []

        # state
        prev_shares = pd.Series(0.0, index=cols)
        cash = float(self.initial_value)

        # daily accrual factors
        dep_d = self._daily_rate(self.deposit_rate_annual)
        bor_d = self._daily_rate(self.borrow_rate_annual)

        # pre-build rebalance dates & weights frame
        self._build_rebalance_and_weights()

        # helper: MTM & cash accrual over date slice with a fixed share vector
        def apply_mtm_and_accrual(dates: pd.DatetimeIndex, shares_vec: pd.Series, cash_balance: float):
            nonlocal index_series, cash_series
            if len(dates) == 0:
                return cash_balance
            for d in dates:
                port_val_ex_cash = float((self.val_px.loc[d] * shares_vec).sum())
                # daily cash accrual
                if cash_balance >= 0:
                    cash_balance *= (1.0 + dep_d)
                else:
                    cash_balance *= (1.0 + bor_d)
                # write end-of-day values
                index_series.loc[d] = port_val_ex_cash + cash_balance
                cash_series.loc[d] = cash_balance
            return cash_balance

        # main loop over contribution/rebalance dates
        last_written_date = None
        for i, dt in enumerate(self.rebalance_dates):
            # accrual & MTM between last written day and day before trade execution
            exec_dt = self._exec_date(dt)
            left = self.val_px.loc[(last_written_date or idx[0]):exec_dt].index
            if last_written_date is None:
                # from series start up to exec_dt-1 (no previous writes yet)
                if len(left) > 1:
                    cash = apply_mtm_and_accrual(left[:-1], prev_shares, cash)
            else:
                # we already wrote last_written_date; continue after it
                if len(left) > 1:
                    cash = apply_mtm_and_accrual(left[1:-1], prev_shares, cash)

            # 1) bring in external contribution (if any) at dt
            contrib_amt = self._contribution_at(dt)
            if abs(contrib_amt) > 0:
                cash += float(contrib_amt)
                self._log(f"[FLOW] {dt.date()} external contribution +{contrib_amt:,.2f}")

            # 2) compute target buy notional by current weights * contribution_at_dt (only fresh cash)
            #    If negative cash (unlikely for pure DCA), we still respect cap/rounding logic.
            px_exec_row = self.exe_px.loc[exec_dt]
            w_row = self._normalize_row(self.weights_df.loc[dt]) if len(cols) > 1 else pd.Series({cols[0]: 1.0})
            buy_notional_target = max(0.0, float(contrib_amt))  # we only invest positive contributions
            target_notionals = w_row * buy_notional_target

            # cap per-ticker notional relative to portfolio value, if requested
            if self.cap_trade_pct < 1.0 - 1e-12:
                pv_before = float((self.val_px.loc[dt] * prev_shares).sum() + cash)
                cap_abs = max(0.0, self.cap_trade_pct * pv_before)
                if cap_abs > 0:
                    target_notionals = target_notionals.clip(upper=cap_abs)

            # translate notionals to raw share deltas
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_delta = (target_notionals / px_exec_row.replace(0.0, np.nan)).fillna(0.0)

            # rounding policy
            delta = raw_delta.copy()
            if not self.allow_partial_shares:
                for tic in cols:
                    q = float(delta[tic])
                    delta[tic] = self._lot_round(q, self.min_lot)

            # compute effective prices and commissions; ensure cash sufficiency (scale down if needed)
            variable_bps = self.commission.get("variable_bps", 0.0) / 1e4
            fixed_fee = self.commission.get("fixed", 0.0)
            min_fee = self.commission.get("min_per_trade", 0.0)

            eff_px_map: Dict[str, float] = {}
            buy_notional = 0.0
            trade_cost_total = 0.0
            for tic in cols:
                q = float(delta[tic])
                if q <= 1e-12:
                    eff = float(px_exec_row[tic])
                else:
                    eff = self._eff_price(float(px_exec_row[tic]), "buy", self._get_tick(tic))
                eff_px_map[tic] = eff
                notional = max(0.0, q) * eff
                fee = 0.0 if q <= 1e-12 else max(notional * variable_bps + fixed_fee, min_fee)
                buy_notional += notional
                trade_cost_total += fee

            tentative_cash = cash - buy_notional - trade_cost_total

            # scale down if cash deficit
            if tentative_cash < -1e-8 and buy_notional > 0:
                scale = max(0.0, (cash - trade_cost_total) / buy_notional)
                self._log(f"[INFO] {dt.date()} scaled buys by {scale:.4f} (cash constraint).")
                for tic in cols:
                    if delta[tic] > 0:
                        delta[tic] *= scale
                        if not self.allow_partial_shares:
                            delta[tic] = self._lot_round(delta[tic], self.min_lot)

                # recompute totals after scaling
                buy_notional = 0.0
                trade_cost_total = 0.0
                for tic in cols:
                    q = float(delta[tic])
                    if q <= 1e-12:
                        continue
                    eff = eff_px_map[tic]
                    notional = q * eff
                    fee = max(notional * variable_bps + fixed_fee, min_fee)
                    buy_notional += notional
                    trade_cost_total += fee
                tentative_cash = cash - buy_notional - trade_cost_total

            # apply execution at exec_dt
            cash = tentative_cash
            new_shares = prev_shares + delta

            # ledgers
            for tic in cols:
                q = float(delta[tic])
                if q > 1e-12:
                    notional = q * eff_px_map[tic]
                    fee = max(notional * variable_bps + fixed_fee, min_fee)
                    trade_rows.append({
                        "date": exec_dt,
                        "ticker": tic,
                        "side": "buy",
                        "shares_delta": q,
                        "exec_price_eff": eff_px_map[tic],
                        "notional": notional,
                        "commission": fee,
                    })
                # snapshot
                hold_rows.append({
                    "date": exec_dt,
                    "ticker": tic,
                    "price_ref": float(px_exec_row[tic]),
                    "shares": float(new_shares[tic]),
                    "amount": float(new_shares[tic] * float(px_exec_row[tic])),
                })

            # append holdings row
            shares_record.loc[exec_dt, cols] = new_shares.values

            # MTM from exec_dt through the next key date (or end)
            next_dt = (self.rebalance_dates[i + 1] if i < len(self.rebalance_dates) - 1 else idx[-1])
            right = self.val_px.loc[exec_dt:next_dt].index
            cash = apply_mtm_and_accrual(right, new_shares, cash)

            prev_shares = new_shares.copy()
            last_written_date = right[-1] if len(right) else exec_dt

        # finalize
        self.index_series = index_series.astype(float)
        self.cash_series = cash_series.astype(float)
        self.shares_record = shares_record.sort_index().astype(float)
        self.trade_ledger = pd.DataFrame(trade_rows).sort_values("date").reset_index(drop=True)
        self.shares_ledger = pd.DataFrame(hold_rows).sort_values("date").reset_index(drop=True)

        return self

    def get_results(self) -> Dict[str, Any]:
        """Return all result objects."""
        if any(obj is None for obj in [self.index_series, self.cash_series, self.shares_record]):
            raise RuntimeError("Run the simulation first with `.run()`.")
        return {
            "index_series": self.index_series,
            "cash_series": self.cash_series,
            "shares_record": self.shares_record,
            "trade_ledger": self.trade_ledger,
            "shares_ledger": self.shares_ledger,
            "logs": self.logs,
        }

    def to_nav(self) -> pd.Series:
        """Return NAV series normalized to 1.0 at the first non-NaN value."""
        s = self.index_series.dropna()
        if s.empty:
            return self.index_series * np.nan
        return self.index_series / float(s.iloc[0])

    # -----------------------
    # Performance statistics
    # -----------------------
    def money_weighted_irr(self) -> float:
        """
        Money-weighted return (IRR) using irregular cash flows (XIRR).
        Contributions are negative flows (investor outflow), terminal value is a positive flow.
        """
        if self.index_series is None:
            raise RuntimeError("Run the simulation first.")
        flows = []
        dates = []

        # initial value as negative flow at the first available date (if provided)
        first_date = self.index_series.first_valid_index()
        if first_date is None:
            return np.nan
        if self.initial_value != 0:
            flows.append(-float(self.initial_value))
            dates.append(pd.Timestamp(first_date))

        # contribution flows
        if isinstance(self.contribution_series, pd.Series):
            cs = self.contribution_series[self.contribution_series != 0]
            for d, v in cs.items():
                flows.append(-float(v))
                dates.append(pd.Timestamp(d))
        elif isinstance(self.contribution_series, (int, float)) and abs(self.contribution_series) > 0:
            for d in self.rebalance_dates:
                flows.append(-float(self.contribution_series))
                dates.append(pd.Timestamp(d))

        # terminal liquidation at the last index date
        last_date = self.index_series.last_valid_index()
        terminal_value = float(self.index_series.loc[last_date])
        flows.append(terminal_value)
        dates.append(pd.Timestamp(last_date))

        return self._xirr(np.array(flows, dtype=float), pd.to_datetime(dates))

    def time_weighted_return(self) -> float:
        """
        Time-weighted return (TWR), chain-linking subperiod returns between external flows.
        """
        if self.index_series is None:
            raise RuntimeError("Run the simulation first.")
        nav = self.index_series.astype(float)

        # define flow dates
        flow_dates = []
        if self.initial_value != 0:
            flow_dates.append(nav.first_valid_index())
        if isinstance(self.contribution_series, pd.Series):
            flow_dates.extend(list(self.contribution_series[self.contribution_series != 0].index))
        elif isinstance(self.contribution_series, (int, float)) and abs(self.contribution_series) > 0:
            flow_dates.extend(list(self.rebalance_dates))

        flow_dates = sorted(set(pd.to_datetime(flow_dates)))  # unique

        # build subperiods as (start_exclusive, end_inclusive)
        dates = nav.dropna().index
        if len(dates) == 0:
            return np.nan
        ends = list(sorted(set(flow_dates + [dates[-1]])))
        starts = [dates[0]] + ends[:-1]

        rets = []
        for s, e in zip(starts, ends):
            if s == e:
                continue
            # subperiod return: (V_e - sum(flows in (s, e]) - V_s) / V_s
            V_s = float(nav.loc[s])
            V_e = float(nav.loc[e])
            cf = 0.0
            for d in flow_dates:
                if (d > s) and (d <= e):
                    if d == dates[0] and self.initial_value != 0:
                        cf += float(self.initial_value)
                    else:
                        if isinstance(self.contribution_series, pd.Series):
                            if d in self.contribution_series.index:
                                cf += float(self.contribution_series.loc[d])
                        else:
                            # scalar contribution
                            if abs(self.contribution_series) > 0 and d in self.rebalance_dates:
                                cf += float(self.contribution_series)
            # From the portfolio's perspective, contributions increase V_e by cf;
            # to get the pure return, remove cf from V_e.
            R = (V_e - cf - V_s) / max(1e-12, V_s)
            rets.append(1.0 + R)

        if len(rets) == 0:
            return 0.0
        return float(np.prod(rets) - 1.0)

    def simple_return_on_invested(self) -> float:
        """
        Simple return on invested capital: (FinalValue - NetInvested) / NetInvested,
        where NetInvested = initial_value + sum(contributions).
        """
        if self.index_series is None:
            raise RuntimeError("Run the simulation first.")
        V_T = float(self.index_series.dropna().iloc[-1])
        net_invested = float(self.initial_value)
        if isinstance(self.contribution_series, pd.Series):
            net_invested += float(self.contribution_series.sum())
        else:
            if abs(self.contribution_series) > 0:
                net_invested += float(self.contribution_series) * int(len(self.rebalance_dates))
        if net_invested == 0:
            return np.nan
        return (V_T - net_invested) / net_invested

    # -----------------------------
    # Internal helpers / utilities
    # -----------------------------
    @staticmethod
    def _coerce_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{name}.index must be a DatetimeIndex.")
        if df.isnull().all().all():
            raise ValueError(f"{name} has all-NaN values.")
        return df.sort_index()

    def _validate_inputs(self):
        if not self.val_px.index.equals(self.val_px.index.unique()):
            raise ValueError("val_px index contains duplicates.")
        if not self.exe_px.index.equals(self.exe_px.index.unique()):
            raise ValueError("exe_px index contains duplicates.")
        if not self.val_px.index.is_monotonic_increasing:
            raise ValueError("val_px index must be increasing.")
        if not self.exe_px.index.is_monotonic_increasing:
            raise ValueError("exe_px index must be increasing.")
        if not set(self.val_px.columns) <= set(self.exe_px.columns):
            missing = set(self.val_px.columns) - set(self.exe_px.columns)
            raise ValueError(f"exe_px missing columns: {missing}")
        if not (self.min_lot >= 1):
            raise ValueError("min_lot must be >= 1")

    def _build_rebalance_and_weights(self):
        idx = self.val_px.index
        cols = self.val_px.columns

        # contribution dates
        if isinstance(self.contribution_series, pd.Series):
            cs = self.contribution_series.copy()
            if not isinstance(cs.index, pd.DatetimeIndex):
                raise TypeError("contribution_series index must be DatetimeIndex.")
            cs = cs.reindex(idx).fillna(0.0)
            dates = list(cs.index[cs != 0.0])
        else:
            # scalar: contribute every day
            if abs(self.contribution_series) > 0:
                dates = list(idx)
            else:
                dates = []

        # ensure we include start if initial_value > 0
        if self.initial_value != 0 and len(idx) > 0:
            if len(dates) == 0 or idx[0] < pd.to_datetime(dates[0]):
                dates = sorted(set([idx[0]] + dates))

        self.rebalance_dates = pd.DatetimeIndex(sorted(set(dates)))

        # weights frame
        if len(cols) == 1:
            self.weights_df = pd.DataFrame({cols[0]: 1.0}, index=self.rebalance_dates)
            return

        if self.weights_raw is None or (isinstance(self.weights_raw, str) and self.weights_raw.lower() == "equal"):
            w = pd.Series(1.0 / len(cols), index=cols)
            self.weights_df = pd.DataFrame([w.values] * len(self.rebalance_dates), index=self.rebalance_dates, columns=cols)
        elif isinstance(self.weights_raw, (pd.Series, dict, Mapping)):
            w = pd.Series(self.weights_raw, index=cols).fillna(0.0)
            self.weights_df = pd.DataFrame([w.values] * len(self.rebalance_dates), index=self.rebalance_dates, columns=cols)
        elif isinstance(self.weights_raw, pd.DataFrame):
            dfw = self.weights_raw.reindex(self.rebalance_dates).reindex(columns=cols).fillna(0.0)
            self.weights_df = dfw
        else:
            raise TypeError("Invalid `weights` type. Use None/'equal', Series/Mapping, or DataFrame.")

        # normalize rows (avoid all-zero)
        self.weights_df = self.weights_df.apply(self._normalize_row, axis=1)

    def _normalize_row(self, row: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        s = pd.Series(row).fillna(0.0)
        tot = float(s.sum())
        if tot <= 0:
            # default to equal if user gave all zeros
            return pd.Series(1.0 / len(s), index=s.index)
        return s / tot

    def _exec_date(self, dt: pd.Timestamp) -> pd.Timestamp:
        if not self.use_next_open:
            return dt
        pos = self.exe_px.index.searchsorted(dt, side="right")
        if pos >= len(self.exe_px.index):
            return self.exe_px.index[-1]
        return self.exe_px.index[pos]

    @staticmethod
    def _daily_rate(annual_rate: float) -> float:
        if annual_rate == 0:
            return 0.0
        return (1.0 + annual_rate) ** (1.0 / 252.0) - 1.0

    @staticmethod
    def _lot_round(q: float, lot: int) -> float:
        """Round down to nearest multiple of lot (buys only in this engine)."""
        if q <= 0:
            return 0.0
        return float((int(q) // lot) * lot)

    def _log(self, msg: str):
        self.logs.append(msg)

    def _get_tick(self, ticker: str) -> Optional[float]:
        if self.tick_size is None:
            return None
        if isinstance(self.tick_size, Mapping):
            return float(self.tick_size.get(ticker, 0.0))
        return float(self.tick_size)

    @staticmethod
    def _eff_price(px: float, side: str, tick: Optional[float]) -> float:
        """
        Apply a half-tick adverse selection to execution price (optional).
        """
        if (tick is None) or (tick <= 0):
            return float(px)
        half = 0.5 * float(tick)
        if side == "buy":
            return float(px) + half
        else:
            return float(px) - half

    def _contribution_at(self, dt: pd.Timestamp) -> float:
        if isinstance(self.contribution_series, pd.Series):
            v = float(self.contribution_series.reindex([dt]).fillna(0.0).iloc[0])
            return v
        else:
            return float(self.contribution_series) if abs(self.contribution_series) > 0 else 0.0

    # -------------------
    # IRR (XIRR) utility
    # -------------------
    @staticmethod
    def _xnpv(rate: float, cashflows: np.ndarray, dates: pd.DatetimeIndex) -> float:
        """Present value at `rate` for irregular dated cashflows."""
        if rate <= -1.0:
            return np.inf
        t0 = dates[0]
        years = np.array([(d - t0).days / 365.2425 for d in dates], dtype=float)
        disc = (1.0 + rate) ** years
        return float(np.sum(cashflows / disc))

    def _xirr(self, cashflows: np.ndarray, dates: pd.DatetimeIndex) -> float:
        """Solve for r such that XNPV(r)=0 via bracketed Newton/Bisect hybrid."""
        # quick exit
        if len(cashflows) < 2:
            return np.nan
        # bracket search
        low, high = -0.9999, 10.0
        f_low = self._xnpv(low, cashflows, dates)
        f_high = self._xnpv(high, cashflows, dates)
        # expand high if same sign
        tries = 0
        while np.sign(f_low) == np.sign(f_high) and tries < 20:
            high *= 2.0
            f_high = self._xnpv(high, cashflows, dates)
            tries += 1
        if np.sign(f_low) == np.sign(f_high):
            return np.nan  # cannot bracket

        # bisection
        for _ in range(100):
            mid = 0.5 * (low + high)
            f_mid = self._xnpv(mid, cashflows, dates)
            if abs(f_mid) < 1e-10:
                return float(mid)
            if np.sign(f_mid) == np.sign(f_low):
                low, f_low = mid, f_mid
            else:
                high, f_high = mid, f_mid
        return float(0.5 * (low + high))