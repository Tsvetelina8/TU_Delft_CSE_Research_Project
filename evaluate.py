import numpy as np
from env import PairsTradingEnv

ZSCORE_WINDOW    = 20
TRANSACTION_COST = 0.001
INITIAL_CAPITAL  = 10_000.0

def rollout(model, spreads, regimes, use_regimes, *, beta,
            deterministic=True, seed=42,
            mu_r=None, sigma_r=None, kappa=None, calib_alpha=0.05):

    rng = np.random.default_rng(seed)
    N, T = spreads.shape

    ep_returns      = []
    ep_sharpes      = []
    ep_max_dds      = []
    ep_n_trades     = []
    all_capital     = []

    env = PairsTradingEnv(
        spreads, regimes,
        use_regimes      = use_regimes,
        transaction_cost = TRANSACTION_COST,
        zscore_window    = ZSCORE_WINDOW,
        mu_r             = mu_r,
        sigma_r          = sigma_r,
        kappa            = kappa,
        calib_alpha      = calib_alpha,
    )

    for ep_idx in range(N):
        env.reset()
        env.ep = ep_idx
        env.t  = ZSCORE_WINDOW
        env.position = 0.0
        obs    = env._obs()

        capital  = INITIAL_CAPITAL
        cap_hist = [capital]
        ret_hist = []
        n_trades = 0
        prev_pos = 0.0

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            econ_ret = info.get("pnl_net", reward) if isinstance(info, dict) else reward
            dollar_ret = capital * econ_ret / (1.0 + beta)
            capital   += dollar_ret
            capital    = max(capital, 1.0)

            w_now = float(env.position)
            if abs(w_now - prev_pos) > 1e-4:
                n_trades += 1
            prev_pos = w_now

            cap_hist.append(capital)
            ret_hist.append(dollar_ret / (capital - dollar_ret + 1e-8))

        ret_hist  = np.array(ret_hist)
        cap_hist  = np.array(cap_hist)

        total_ret = (cap_hist[-1] / cap_hist[0] - 1) * 100
        sr        = sharpe(ret_hist)
        mdd       = max_drawdown(cap_hist)

        ep_returns.append(total_ret)
        ep_sharpes.append(sr)
        ep_max_dds.append(mdd)
        ep_n_trades.append(n_trades)
        all_capital.append(cap_hist)

    return {
        "returns"    : np.array(ep_returns),
        "sharpes"    : np.array(ep_sharpes),
        "max_dds"    : np.array(ep_max_dds),
        "n_trades"   : np.array(ep_n_trades),
        "capital"    : np.array(all_capital),
    }

class ZScorePolicy:
    """
    Classical z-score threshold strategy.

    Enter long  when z < -threshold   (spread below mean, expect reversion up)
    Enter short when z > +threshold   (spread above mean, expect reversion down)
    Exit        when |z| < exit_threshold  (spread has reverted to near mean)
    Hold        otherwise              (within threshold band, no new signal)
    """
    def __init__(self, threshold=1.5, exit_threshold=0.5):
        self.thr  = threshold
        self.exit = exit_threshold

    def predict(self, obs, deterministic=True):
        z   = float(obs[0])   # z-score (first element of observation)
        pos = float(obs[1])   # current position (second element)

        if z < -self.thr:
            w = 1.0           # long spread — below lower band
        elif z > self.thr:
            w = -1.0          # short spread — above upper band
        elif abs(z) < self.exit:
            w = 0.0           # exit — spread has reverted
        else:
            w = pos           # hold — inside band, no new signal

        return np.array([w], dtype=np.float32), None


def sharpe(daily_returns, periods_per_year=252):
    """Annualised Sharpe ratio from daily return array."""
    r = np.array(daily_returns)
    if r.std() < 1e-10:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def max_drawdown(capital_curve):
    """Maximum drawdown as a percentage."""
    c    = np.array(capital_curve)
    peak = np.maximum.accumulate(c)
    dd   = (c - peak) / (peak + 1e-8) * 100
    return float(dd.min())
