import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PairsTradingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        spreads,
        regimes,
        use_regimes      = False,
        transaction_cost = 0.001,
        zscore_window    = 20,
        mu_r             = None,
        sigma_r          = None,
        kappa            = None,
        n_regimes        = 3,
        calib_alpha      = 0.05,
    ):
        super().__init__()

        self.mu_r     = mu_r
        self.sigma_r  = sigma_r
        self.kappa    = kappa
        if mu_r is not None and sigma_r is not None and kappa is not None:
            self.stat_std_r = {r: float(sigma_r[r] / np.sqrt(2.0 * kappa))
                               for r in range(n_regimes)}
        else:
            self.stat_std_r = None
        
        self.calib_alpha  = float(calib_alpha)
        self.mu_r_live    = None

        self.spreads     = spreads.astype(np.float32)
        self.regimes     = regimes.astype(np.int8)
        self.use_regimes = use_regimes
        self.tc          = transaction_cost
        self.zscore_window = zscore_window
        self.n_regimes   = n_regimes
        self.n_episodes  = spreads.shape[0]
        self.horizon     = spreads.shape[1]

        obs_dim = 3

        self.observation_space = spaces.Box(
            low  = -np.ones(obs_dim, dtype=np.float32) * 10,
            high =  np.ones(obs_dim, dtype=np.float32) * 10,
            dtype = np.float32,
        )

        self.action_space = spaces.Box(
            low  = np.array([-1.0], dtype=np.float32),
            high = np.array([ 1.0], dtype=np.float32),
            dtype = np.float32,
        )

        self.ep       = 0
        self.t        = 0
        self.position = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ep = int(self.np_random.integers(0, self.n_episodes))
        self.t = self.zscore_window
        self.position = 0.0
        if self.mu_r is not None:
            self.mu_r_live = {r: float(self.mu_r[r]) for r in range(self.n_regimes)}
        return self._obs(), {}

    def step(self, action):
        w_new  = float(np.clip(action[0], -1.0, 1.0))
        s = self.spreads[self.ep]
        t = self.t
        dz = float(s[t] - s[t - 1]) if t > 0 else 0.0

        pnl = self.position * dz
        cost = self.tc * abs(w_new - self.position)
        pnl_net = pnl - cost
        reward = pnl_net

        self.position = w_new
        self.t += 1
        terminated = (self.t >= self.horizon)
        if terminated:
            self.position = 0.0

        return self._obs(), float(reward), terminated, False, {"pnl_net": pnl_net}

    def _obs(self):
        s = self.spreads[self.ep]
        t = min(self.t, self.horizon - 1)

        # Z-score
        lo = max(0, t - self.zscore_window)
        window = s[lo: t] if t > lo else s[:1]
        zscore = float(np.clip(
            (s[t] - window.mean()) / (window.std() + 1e-8), -5.0, 5.0
        ))

        # Volatility (over the z-score window)
        vlo   = max(0, t - self.zscore_window)
        diffs = np.diff(s[vlo: t + 1]) if t > vlo else np.array([0.0])
        vol   = float(np.clip(diffs.std() * np.sqrt(252), 0.0, 3.0))

        if not self.use_regimes:
            return np.array([zscore, self.position, vol], dtype=np.float32)

        r = int(self.regimes[self.ep, t])

        # Regime-adjusted z-score
        if self.stat_std_r is not None:
            if self.mu_r_live is None:
                self.mu_r_live = {rr: float(self.mu_r[rr])
                                  for rr in range(self.n_regimes)}
            self.mu_r_live[r] = (self.calib_alpha * float(s[t])
                                 + (1.0 - self.calib_alpha) * self.mu_r_live[r])
            z_adj = float(np.clip(
                (s[t] - self.mu_r_live[r]) / (self.stat_std_r[r] + 1e-8), -5.0, 5.0))
        else:
            z_adj = zscore

        return np.array([z_adj, self.position, vol], dtype=np.float32)
