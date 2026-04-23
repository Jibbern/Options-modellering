"""Read-only IBKR socket session wrapper for delayed-only ingestion."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

from .models import ConnectionSettings, market_data_mode_code


class IbkrConnectionError(RuntimeError):
    """Raised when a local TWS / IB Gateway session is unavailable or unusable."""


def _ibapi_classes():
    try:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper
    except ModuleNotFoundError as exc:
        raise IbkrConnectionError(
            "ibapi is not installed. Install the official IBKR Python API to use fetch-ibkr commands."
        ) from exc
    return EClient, EWrapper


class DelayedOnlyIbkrSession:
    """Small read-only wrapper around the official IBKR socket API."""

    def __init__(self, settings: ConnectionSettings, *, timeout: float = 10.0):
        self.settings = settings
        self.timeout = timeout
        self._next_valid_id_event = threading.Event()
        self._disconnect_event = threading.Event()
        self._lock = threading.Lock()
        self._next_request_id = 1
        self._errors: list[dict[str, Any]] = []
        self._errors_by_req: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._contract_details: dict[int, list[Any]] = defaultdict(list)
        self._contract_done: dict[int, threading.Event] = defaultdict(threading.Event)
        self._secdef_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._secdef_done: dict[int, threading.Event] = defaultdict(threading.Event)
        self._market_data_type: dict[int, int] = {}
        self._market_ticks: dict[int, dict[str, Any]] = defaultdict(dict)
        self._market_tick_events: dict[int, threading.Event] = defaultdict(threading.Event)

        EClient, EWrapper = _ibapi_classes()
        outer = self

        class _App(EWrapper, EClient):
            def __init__(self):
                EClient.__init__(self, self)

            def nextValidId(self, orderId: int):
                outer._on_next_valid_id(orderId)

            def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
                outer._on_error(reqId, errorCode, errorString, advancedOrderRejectJson)

            def connectionClosed(self):
                outer._disconnect_event.set()

            def contractDetails(self, reqId: int, contractDetails):
                outer._contract_details[reqId].append(contractDetails)

            def contractDetailsEnd(self, reqId: int):
                outer._contract_done[reqId].set()

            def securityDefinitionOptionParameter(
                self,
                reqId: int,
                exchange: str,
                underlyingConId: int,
                tradingClass: str,
                multiplier: str,
                expirations,
                strikes,
            ):
                outer._secdef_rows[reqId].append(
                    {
                        "exchange": exchange,
                        "underlying_conid": underlyingConId,
                        "trading_class": tradingClass,
                        "multiplier": multiplier,
                        "expirations": sorted(expirations or []),
                        "strikes": sorted(float(value) for value in strikes or []),
                    }
                )

            def securityDefinitionOptionParameterEnd(self, reqId: int):
                outer._secdef_done[reqId].set()

            def marketDataType(self, reqId: int, marketDataType: int):
                outer._market_data_type[reqId] = int(marketDataType)
                outer._market_tick_events[reqId].set()

            def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
                mapping = {
                    1: "bid",
                    2: "ask",
                    4: "last",
                    9: "close",
                }
                key = mapping.get(int(tickType))
                if key is not None:
                    outer._market_ticks[reqId][key] = float(price) if price is not None and price >= 0 else None
                    outer._market_tick_events[reqId].set()

            def tickSize(self, reqId: int, tickType: int, size: int):
                mapping = {
                    0: "bid_size",
                    3: "ask_size",
                    5: "last_size",
                    8: "volume",
                }
                key = mapping.get(int(tickType))
                if key is not None:
                    outer._market_ticks[reqId][key] = float(size) if size is not None and size >= 0 else None
                    outer._market_tick_events[reqId].set()

            def tickGeneric(self, reqId: int, tickType: int, value: float):
                mapping = {
                    100: "option_volume",
                    101: "open_interest",
                    104: "historical_volatility",
                    106: "implied_volatility",
                    221: "mark",
                }
                key = mapping.get(int(tickType))
                if key is not None:
                    outer._market_ticks[reqId][key] = float(value) if value is not None and value >= 0 else None
                    outer._market_tick_events[reqId].set()

            def tickString(self, reqId: int, tickType: int, value: str):
                if int(tickType) == 45:
                    outer._market_ticks[reqId]["last_timestamp"] = value
                    outer._market_tick_events[reqId].set()

            def tickOptionComputation(
                self,
                reqId: int,
                tickType: int,
                tickAttrib,
                impliedVol: float,
                delta: float,
                optPrice: float,
                pvDividend: float,
                gamma: float,
                vega: float,
                theta: float,
                undPrice: float,
            ):
                prefix = {
                    10: "bid",
                    11: "ask",
                    12: "last",
                    13: "model",
                }.get(int(tickType), "model")
                target = outer._market_ticks[reqId]
                if impliedVol is not None and impliedVol >= 0:
                    target.setdefault("implied_volatility", float(impliedVol))
                if delta is not None and delta > -2:
                    target[f"{prefix}_delta"] = float(delta)
                    target.setdefault("delta", float(delta))
                if optPrice is not None and optPrice >= 0:
                    target[f"{prefix}_option_price"] = float(optPrice)
                    target.setdefault("option_price", float(optPrice))
                if pvDividend is not None and pvDividend > -1e12:
                    target["pv_dividend"] = float(pvDividend)
                if gamma is not None and gamma > -1e12:
                    target["gamma"] = float(gamma)
                if vega is not None and vega > -1e12:
                    target["vega"] = float(vega)
                if theta is not None and theta > -1e12:
                    target["theta"] = float(theta)
                if undPrice is not None and undPrice >= 0:
                    target["under_price"] = float(undPrice)
                outer._market_tick_events[reqId].set()

        self._app = _App()
        self._thread: threading.Thread | None = None

    def _on_next_valid_id(self, order_id: int) -> None:
        with self._lock:
            self._next_request_id = max(self._next_request_id, int(order_id))
        self._next_valid_id_event.set()

    def _on_error(self, req_id: int, error_code: int, error_string: str, advanced_json: str = "") -> None:
        payload = {
            "req_id": int(req_id),
            "error_code": int(error_code),
            "error_string": error_string,
        }
        if advanced_json:
            payload["advanced_order_reject_json"] = advanced_json
        self._errors.append(payload)
        self._errors_by_req[int(req_id)].append(payload)
        if int(req_id) in self._contract_done:
            self._contract_done[int(req_id)].set()
        if int(req_id) in self._secdef_done:
            self._secdef_done[int(req_id)].set()
        if int(req_id) in self._market_tick_events:
            self._market_tick_events[int(req_id)].set()

    def connect(self) -> None:
        connected = self._app.connect(self.settings.host, self.settings.port, self.settings.client_id)
        if connected is False:
            raise IbkrConnectionError(
                f"Could not connect to IBKR on {self.settings.host}:{self.settings.port}. "
                "Check that TWS or IB Gateway is running and API access is enabled."
            )
        self._thread = threading.Thread(target=self._app.run, daemon=True)
        self._thread.start()
        if not self._next_valid_id_event.wait(self.timeout):
            error_text = "; ".join(error["error_string"] for error in self._errors[-3:]) or "no API response received"
            self.disconnect()
            raise IbkrConnectionError(
                f"Connected to {self.settings.host}:{self.settings.port} but did not receive nextValidId within {self.timeout:.1f}s: {error_text}"
            )

    def disconnect(self) -> None:
        try:
            self._app.disconnect()
        finally:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()
        return False

    def next_request_id(self) -> int:
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
        return request_id

    def errors_for_request(self, req_id: int) -> list[dict[str, Any]]:
        return list(self._errors_by_req.get(int(req_id), []))

    def request_market_data_mode(self, mode: str) -> int:
        code = market_data_mode_code(mode)
        self._app.reqMarketDataType(code)
        return code

    def request_contract_details(self, contract, *, timeout: float | None = None) -> list[Any]:
        req_id = self.next_request_id()
        self._contract_details[req_id] = []
        self._contract_done[req_id] = threading.Event()
        self._app.reqContractDetails(req_id, contract)
        if not self._contract_done[req_id].wait(timeout or self.timeout):
            raise TimeoutError(f"Timed out waiting for IBKR contract details for reqId={req_id}.")
        return list(self._contract_details.get(req_id, []))

    def request_option_parameters(
        self,
        *,
        symbol: str,
        underlying_sec_type: str,
        underlying_conid: int,
        fut_fop_exchange: str = "",
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        req_id = self.next_request_id()
        self._secdef_rows[req_id] = []
        self._secdef_done[req_id] = threading.Event()
        self._app.reqSecDefOptParams(req_id, symbol, fut_fop_exchange, underlying_sec_type, underlying_conid)
        if not self._secdef_done[req_id].wait(timeout or self.timeout):
            raise TimeoutError(f"Timed out waiting for IBKR option parameters for reqId={req_id}.")
        return list(self._secdef_rows.get(req_id, []))

    def collect_market_snapshot(
        self,
        contract,
        *,
        market_data_mode: str,
        generic_tick_list: str = "",
        wait_seconds: float = 4.0,
    ) -> dict[str, Any]:
        req_id = self.next_request_id()
        self.request_market_data_mode(market_data_mode)
        self._market_tick_events[req_id] = threading.Event()
        self._market_ticks[req_id] = {}
        self._app.reqMktData(req_id, contract, generic_tick_list, False, False, [])

        deadline = time.time() + max(wait_seconds, 0.5)
        first_data_at: float | None = None
        while time.time() < deadline:
            if self._market_tick_events[req_id].wait(0.25):
                self._market_tick_events[req_id].clear()
                if self._market_ticks.get(req_id) and first_data_at is None:
                    first_data_at = time.time()
                if first_data_at is not None and (time.time() - first_data_at) >= 0.75:
                    break
        self._app.cancelMktData(req_id)
        return {
            "req_id": req_id,
            "market_data_type_code": self._market_data_type.get(req_id),
            "ticks": dict(self._market_ticks.get(req_id, {})),
            "errors": self.errors_for_request(req_id),
        }

    def collect_market_snapshots(
        self,
        contracts,
        *,
        market_data_mode: str,
        generic_tick_list: str = "",
        wait_seconds: float = 4.0,
        settle_seconds: float = 1.0,
    ) -> list[dict[str, Any]]:
        request_pairs: list[tuple[int, Any]] = []
        self.request_market_data_mode(market_data_mode)
        for contract in contracts:
            req_id = self.next_request_id()
            self._market_tick_events[req_id] = threading.Event()
            self._market_ticks[req_id] = {}
            request_pairs.append((req_id, contract))
            self._app.reqMktData(req_id, contract, generic_tick_list, False, False, [])

        deadline = time.time() + max(wait_seconds, 0.5)
        last_update_at: float | None = None
        real_data_seen = False
        try:
            while time.time() < deadline:
                updated = False
                for req_id, _contract in request_pairs:
                    event = self._market_tick_events[req_id]
                    if event.is_set():
                        event.clear()
                        if self._market_ticks.get(req_id):
                            real_data_seen = True
                        updated = True
                if updated:
                    last_update_at = time.time()
                    continue
                if (
                    real_data_seen
                    and last_update_at is not None
                    and (time.time() - last_update_at) >= max(settle_seconds, 0.1)
                ):
                    break
                if (
                    last_update_at is not None
                    and not real_data_seen
                    and (time.time() - last_update_at) >= max(wait_seconds, 0.5)
                ):
                    break
                time.sleep(0.1)
        finally:
            for req_id, _contract in request_pairs:
                self._app.cancelMktData(req_id)

        return [
            {
                "req_id": req_id,
                "market_data_type_code": self._market_data_type.get(req_id),
                "ticks": dict(self._market_ticks.get(req_id, {})),
                "errors": self.errors_for_request(req_id),
            }
            for req_id, _contract in request_pairs
        ]
