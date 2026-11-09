import numpy as np

# HAC covariance matrix
HAC_matrix = np.array([
    [8.507526398719898e-06, -2.481778447058299e-06, 7.966160692733066e-06,
     -7.867716334907634e-06, 1.209793606002429e-07, -6.058187308713758e-05],
    [-2.481778447058299e-06, 0.0002222572618298202, -5.083520951131941e-05,
     1.030834129704509e-05, -0.0002267137873032802, -0.0002898724110253803],
    [7.966160692733066e-06, -5.083520951131941e-05, 0.0004538704845508114,
     -1.539527789330668e-05, 8.75283613440443e-05, 0.0002569638379923147],
    [-7.867716334907634e-06, 1.030834129704509e-05, -1.539527789330668e-05,
     1.029641781553392e-05, -7.276883740540779e-06, -1.977877749943097e-05],
    [1.209793606002429e-07, -0.0002267137873032802, 8.75283613440443e-05,
     -7.276883740540779e-06, 0.000259886179521454, 0.0003731099645247996],
    [-6.058187308713758e-05, -0.0002898724110253803, 0.0002569638379923147,
     -1.977877749943097e-05, 0.0003731099645247996, 0.003484345146219229]
])

HAC_scaled = HAC_matrix * 10000

print(r"""
\subsection*{HAC Covariance Matrix and Confidence Interval Computation}

The impulse response function confidence intervals are computed using the delta method with the HAC (Heteroskedasticity and Autocorrelation Consistent) covariance matrix from the STR estimation. Given that the effective coefficients vary with the transition index $G$:

\begin{equation}
\beta(G) = \alpha_1 + \beta_1 \cdot G, \quad \phi(G) = \alpha_2 + \beta_2 \cdot G
\end{equation}

where $\alpha_1, \beta_1$ are the linear and non-linear coefficients on $\Delta FSI_{t-1}$, and $\alpha_2, \beta_2$ are the linear and non-linear coefficients on $\Delta \log(RGDP_{t-1})$.

The variance of these effective coefficients is computed via the delta method:

\begin{equation}
\text{Var}(\beta(G)) = \mathbf{J}_\beta^\top \mathbf{\Sigma}_{HAC} \mathbf{J}_\beta, \quad
\text{Var}(\phi(G)) = \mathbf{J}_\phi^\top \mathbf{\Sigma}_{HAC} \mathbf{J}_\phi
\end{equation}

where $\mathbf{J}_\beta = [0, 1, 0, 0, G, 0]^\top$ and $\mathbf{J}_\phi = [0, 0, 1, 0, 0, G]^\top$ are Jacobian vectors with respect to the parameter vector $\boldsymbol{\theta} = [\alpha_0, \alpha_1, \alpha_2, \beta_0, \beta_1, \beta_2]^\top$, and $\mathbf{\Sigma}_{HAC}$ is the HAC covariance matrix:

\begin{equation}
\mathbf{\Sigma}_{HAC} \times 10^4 = 
\begin{array}{c|cccccc}
& \alpha_0 & \alpha_1 & \alpha_2 & \beta_0 & \beta_1 & \beta_2 \\
\hline
""")

labels = [r"\alpha_0", r"\alpha_1", r"\alpha_2", r"\beta_0", r"\beta_1", r"\beta_2"]
for i, label in enumerate(labels):
    row_str = " & ".join([f"{HAC_scaled[i,j]:.4f}" for j in range(6)])
    if i < 5:
        print(f"{label} & {row_str} \\\\")
    else:
        print(f"{label} & {row_str}")

print(r"""\end{array}
\end{equation}

The IRF variance at each horizon $h$ is then propagated through the autoregressive structure:

\begin{equation}
\begin{aligned}
\text{IRF}[1] &= \beta(G) \cdot \sigma, \quad \text{Var}(\text{IRF}[1]) = \text{Var}(\beta(G)) \cdot \sigma^2 \\
\text{IRF}[h] &= \text{IRF}[h-1] \cdot \phi(G) \quad (h \geq 2) \\
\text{Var}(\text{IRF}[h]) &= \phi(G)^2 \cdot \text{Var}(\text{IRF}[h-1]) + \text{IRF}[h-1]^2 \cdot \text{Var}(\phi(G))
\end{aligned}
\end{equation}

The 68\% confidence intervals are constructed as $\text{IRF}[h] \pm \sqrt{\text{Var}(\text{IRF}[h])}$. This approach properly accounts for:
\begin{enumerate}
\item \textbf{Cross-coefficient correlations}: The off-diagonal elements of $\mathbf{\Sigma}_{HAC}$ capture correlations between parameter estimates. For example, $\text{Cov}(\alpha_1, \beta_1) = -0.227 \times 10^{-4}$ indicates that linear and non-linear FSI coefficients are negatively correlated, which substantially reduces uncertainty in high-stress regimes.
\item \textbf{Heteroskedasticity and autocorrelation}: The HAC estimator (Newey-West with Bartlett kernel, bandwidth 4) corrects for both heteroskedasticity in residuals and serial correlation up to 4 lags.
\item \textbf{Variance accumulation}: Uncertainty compounds over horizons through the autoregressive dynamics, with wider bands in regimes exhibiting high persistence ($\phi \approx 0.56$ in HS).
\end{enumerate}

Note: $\alpha_0, \beta_0$ are the linear and non-linear intercepts (not used in IRF computation as we analyze deviations from baseline paths).
""")

