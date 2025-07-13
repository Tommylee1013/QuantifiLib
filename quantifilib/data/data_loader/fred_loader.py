import pandas_datareader as pdr
import pandas as pd
from typing import List
from .loader_base import BaseDataLoader

class FREDLoader(BaseDataLoader):
    """
    Loader for FRED (Federal Reserve Economic Data) using pandas-datareader.
    Each symbol corresponds to a FRED series code (e.g., CPIAUCSL, FEDFUNDS).
    """

    def load(
            self, symbols : List[str],
            start : str,
            end : str = None,
            **kwargs
        ) -> pd.DataFrame :
        """
        Loads FRED series data for given symbols.
        :param symbols: Ticker Symbols
        :param start: start date in 'yyyy-mm-dd'
        :param end: end date in 'yyyy-mm-dd'
        :param kwargs:
        :return:
        """
        frames = []
        for sym in symbols:
            try:
                df = pdr.DataReader(sym, 'fred', start, end)
                df.columns = ['Value']
                df['Symbol'] = sym
                frames.append(df)
            except Exception as e:
                print(f"[FREDLoader] Failed to load {sym}: {e}")

        result = pd.concat(frames)
        result = result.reset_index().set_index(['Symbol', 'DATE']).sort_index()
        result.index.names = ['Symbol', 'Date']
        return result