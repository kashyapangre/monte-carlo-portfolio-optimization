"""
Monte Carlo portfolio optimisation on the Dow Jones 30.

Same idea as the HKU Excel project: randomise the portfolio weights, work out
the risk and return of each portfolio, and plot the efficient frontier.
Python lets us also solve for the true frontier with an optimiser and compare.

Setup:  pip install yfinance numpy pandas matplotlib scipy
Run:    python portfolio_monte_carlo.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Current Dow constituents at the time of writing. Check the index hasn't changed.
TICKERS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]
START = "2010-01-01"
N_PORTFOLIOS = 500_000   # Python is fast, so we can go well past 1,000
RISK_FREE = 0.04        # annual, a rough assumption. Change it to the current rate
PERIODS = 12            # monthly data, so multiply by 12 to annualise


def load_returns():
    import yfinance as yf
    prices = yf.download(TICKERS, start=START, interval="1mo", auto_adjust=True)["Close"]
    return prices.dropna().pct_change().dropna()


def monte_carlo(mu, cov, n, seed=42):
    """Random long-only portfolios: draw random numbers, then scale so weights sum to 1."""
    rng = np.random.default_rng(seed)
    w = rng.random((n, len(mu)))
    w /= w.sum(axis=1, keepdims=True)
    rets = w @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    return w, rets, vols, (rets - RISK_FREE) / vols


def optimise(mu, cov, target=None, maximise_sharpe=False):
    """Long-only optimiser: min variance (optionally at a target return), or max Sharpe."""
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target is not None:
        cons.append({"type": "eq", "fun": lambda w, t=target: w @ mu - t})
    if maximise_sharpe:
        obj = lambda w: -(w @ mu - RISK_FREE) / np.sqrt(w @ cov @ w)
    else:
        obj = lambda w: w @ cov @ w
    return minimize(obj, np.ones(n) / n, method="SLSQP",
                    bounds=[(0, 0.1)] * n, constraints=cons)


def efficient_frontier(mu, cov, points=40):
    min_var = optimise(mu, cov)
    start = min_var.x @ mu
    vols, rets = [], []
    for t in np.linspace(start, mu.max() * 0.98, points):
        res = optimise(mu, cov, target=t)
        if res.success:
            rets.append(t)
            vols.append(np.sqrt(res.x @ cov @ res.x))
    return np.array(vols), np.array(rets)


def show_top_weights(label, w, names, k=5):
    top = pd.Series(w, index=names).sort_values(ascending=False).head(k)
    print(f"\n{label}")
    for name, weight in top.items():
        print(f"  {name:<6}{weight:6.1%}")


def main():
    returns = load_returns()
    names = returns.columns
    split = 150
    train, test = returns.iloc[:split], returns.iloc[split:]
    
    mu_train = train.mean().values * PERIODS
    cov_train = train.cov().values * PERIODS
    opt_w = optimise(mu_train, cov_train, maximise_sharpe=True).x
    minvar_w = optimise(mu_train, cov_train).x
    equal_w = np.ones(len(names)) / len(names)
    
    def score(w, data):
        monthly = data.values @ w            # portfolio return each month
        ret = monthly.mean() * PERIODS
        vol = monthly.std(ddof=1) * np.sqrt(PERIODS)
        return ret, vol, (ret - RISK_FREE) / vol
    
    for label, w, data in [("Optimiser, in-sample", opt_w, train),
                           ("Optimiser, out-of-sample", opt_w, test),
                           ("Min variance, in-sample", minvar_w, train),
                           ("Min variance, out-of-sample", minvar_w, test),
                           ("Equal weight, out-of-sample", equal_w, test)]:
        r, v, s = score(w, data)
        print(f"{label:<28} return {r:.1%}  vol {v:.1%}  Sharpe {s:.2f}")
    show_top_weights("Top weights (min variance)", minvar_w, names)
    mu = returns.mean().values * PERIODS
    cov = returns.cov().values * PERIODS
    print(f"{returns.shape[1]} stocks, {returns.shape[0]} months of data")

    w, rets, vols, sharpe = monte_carlo(mu, cov, N_PORTFOLIOS)
    best = sharpe.argmax()
    print(f"\nBest random portfolio ({N_PORTFOLIOS:,} tries): "
          f"return {rets[best]:.1%}, vol {vols[best]:.1%}, Sharpe {sharpe[best]:.2f}")
    show_top_weights("Top weights (random best)", w[best], names)

    opt = optimise(mu, cov, maximise_sharpe=True)
    o_ret, o_vol = opt.x @ mu, np.sqrt(opt.x @ cov @ opt.x)
    print(f"\nOptimiser max-Sharpe portfolio: return {o_ret:.1%}, vol {o_vol:.1%}, "
          f"Sharpe {(o_ret - RISK_FREE) / o_vol:.2f}")
    show_top_weights("Top weights (optimiser)", opt.x, names)

    f_vols, f_rets = efficient_frontier(mu, cov)

    plt.figure(figsize=(9, 6))
    sc = plt.scatter(vols, rets, c=sharpe, cmap="viridis", s=6, alpha=0.7)
    plt.colorbar(sc, label="Sharpe ratio")
    plt.plot(f_vols, f_rets, "r-", lw=2, label="Efficient frontier (optimiser)")
    plt.scatter(vols[best], rets[best], c="orange", s=80, marker="*", label="Best random portfolio")
    plt.scatter(o_vol, o_ret, c="red", s=80, marker="o", label="Max Sharpe (optimiser)")
    plt.xlabel("Annual volatility")
    plt.ylabel("Annual return")
    plt.title("Dow Jones 30: Monte Carlo portfolios vs efficient frontier")
    plt.legend()
    plt.tight_layout()
    plt.savefig("efficient_frontier.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
