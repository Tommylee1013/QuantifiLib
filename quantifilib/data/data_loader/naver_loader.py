import pandas_datareader.data as web
import pandas as pd
from typing import List
from .loader_base import BaseDataLoader

class NaverFinanceLoader(BaseDataLoader):
    """
    Naver Finance data loader using pandas-datareader.
    Symbol should be in KRX format (e.g., '005930.KQ', '000660.KQ').
    """

    def load(
            self, symbols: List[str],
            start: str,
            end: str = None,
            **kwargs
        ) -> pd.DataFrame :
        """
        Load Naver finance data.
        :param symbols: Ticker Symbols
        :param start: start date in 'yyyy-mm-dd'
        :param end: end date in 'yyyy-mm-dd'
        :param kwargs:
        :return:
        """
        frames = []

        for symbol in symbols:
            try:
                df = web.DataReader(symbol, data_source='naver', start=start, end=end, **kwargs)
                df['Symbol'] = symbol
                frames.append(df)
            except Exception as e:
                print(f"[NaverFinanceLoader] Failed to load {symbol}: {e}")

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames)
        result = result.reset_index().set_index(['Symbol', 'Date']).sort_index()
        return result