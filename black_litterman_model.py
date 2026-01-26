import numpy as np

import yfinance as yf



def black_litterman_adjustment(market_returns, investor_views, view_confidences, historical_data, tau=0.025):
    """
    Adjust market returns based on the investor's views and confidences using historical data
    :param dict market_returns: expected market returns for each asset
    :param dict investor_views: investor's specific views on the expected returns of assets
    :param dict view_confidences: investor's confidence in each view
    :param pandas Dataframe historical_data: historical market data
    :param float tau: The uncertainty of the market equilibrium
    :return: numpy array, the adjusted returns for each asset after considering the investor's views
    """
    num_assets = len(market_returns)
    P = np.eye(num_assets)  # Proportion matrix
    Q = np.array(list(investor_views.values())).reshape(-1, 1)

    cov_matrix = historical_data['Adj Close'].pct_change().dropna().cov()

    # Ensure investor views and confidences are also arrays
    omega = np.diag([tau / confidence for confidence in view_confidences.values()])

    # Black-Litterman formula
    inv_omega = np.linalg.inv(omega)
    adjusted_returns = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(inv_omega, P)))
    adjusted_returns = np.dot(adjusted_returns, np.dot(np.linalg.inv(tau * cov_matrix), np.array(list(market_returns.values())).reshape(-1, 1)) + np.dot(P.T, np.dot(inv_omega, Q)))

    return adjusted_returns.flatten()


def get_market_caps(tickers):
    """
    Obtiene la capitalización de mercado actual desde Yahoo Finance
    usando yfinance.

    :param tickers: list[str] – lista de tickers (ej: ['AAPL','MSFT','GOOGL'])
    :return: dict {ticker: market_cap}
    """
    market_caps = {}

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info

        market_cap = info.get("marketCap")
        if market_cap is None:
            raise ValueError(f"No se pudo obtener marketCap para {ticker}")

        market_caps[ticker] = market_cap

    return market_caps


def get_market_returns(market_caps, index_return):
    """
    Calculate market returns based on market capitalizations and index return.
    :param dict market_caps: Market capitalizations of the stocks.
    :param float index_return: Return of the market index.
    :return: A dictionary with tickers and their market returns.
    """
    total_market_cap = sum(market_caps.values())
    return {ticker: (cap / total_market_cap) * index_return for ticker, cap in market_caps.items()}
