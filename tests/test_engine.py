import queue
import pytest
from unittest.mock import MagicMock
from engine import Backtest
from event import MarketEvent, SignalEvent, OrderEvent, FillEvent
from datetime import datetime

def test_engine_initialization():
    dh = MagicMock()
    st = MagicMock()
    pf = MagicMock()
    eh = MagicMock()
    q = queue.Queue()
    
    backtest = Backtest(data_handler=dh, strategy=st, portfolio=pf, execution_handler=eh, event_queue=q)
    assert backtest.data_handler == dh
    assert backtest.strategy == st
    assert backtest.portfolio == pf
    assert backtest.execution_handler == eh
    assert backtest.queue == q

def test_engine_run():
    dh = MagicMock()
    st = MagicMock()
    pf = MagicMock()
    eh = MagicMock()
    q = queue.Queue()
    
    # Configure dh.continue_backtest to run twice then stop
    dh.continue_backtest = True
    
    def side_effect_update_bars():
        # Stop backtest on second call to prevent infinite loop
        if dh.update_bars.call_count == 2:
            dh.continue_backtest = False
        
        # Add events to queue on first call
        if dh.update_bars.call_count == 1:
            q.put(MarketEvent(symbol="AAPL", timestamp=datetime.now(), close=100.0, high=101.0, low=99.0, volume=1000.0))
            q.put(SignalEvent(symbol="AAPL", timestamp=datetime.now(), direction="LONG"))
            q.put(OrderEvent(symbol="AAPL", timestamp=datetime.now(), quantity=100, direction="LONG", orderType="MARKET"))
            q.put(FillEvent(symbol="AAPL", timestamp=datetime.now(), quantity=100, direction="LONG", fillPrice=100.0, commission=1.0, slippage=0.0))

    dh.update_bars.side_effect = side_effect_update_bars
    
    backtest = Backtest(data_handler=dh, strategy=st, portfolio=pf, execution_handler=eh, event_queue=q)
    backtest.run()
    
    # Assert dispatch methods called
    st.calculate_signals.assert_called_once()
    pf.update_timeindex.assert_called_once()
    pf.update_signal.assert_called_once()
    eh.execute_order.assert_called_once()
    pf.update_fill.assert_called_once()
