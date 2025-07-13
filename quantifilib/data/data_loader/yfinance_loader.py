import yfinance as yf
import pandas as pd
from typing import List
from .loader_base import BaseDataLoader

class YahooFinanceLoader(BaseDataLoader):
    """
    Yahoo finance based data loader using yfinance library
    """

    def load(
            self, symbols : List[str],
            start : str,
            end : str = None,
            **kwargs
        ) -> pd.DataFrame :
        """
        Loads data from Yahoo Finance API
        :param symbols: Ticker Symbols
        :param start: start date in 'yyyy-mm-dd'
        :param end: end date in 'yyyy-mm-dd'
        :param kwargs:
        :return:
        """

        data = yf.download(
            tickers = symbols,
            start = start,
            end = end,
            **kwargs
        )

        # tidy multiindex format
        if isinstance(data.columns, pd.MultiIndex):
            stacked = []
            for symbol in symbols:
                if symbol in data.columns.levels[0]:
                    sym_df = data[symbol].copy()
                    sym_df['Symbol'] = symbol
                    stacked.append(sym_df.reset_index())
            result = pd.concat(stacked).set_index(['Symbol', 'Date']).sort_index()
        else:
            data['Symbol'] = symbols[0]
            result = data.reset_index().set_index(['Symbol', 'Date'])

        return result