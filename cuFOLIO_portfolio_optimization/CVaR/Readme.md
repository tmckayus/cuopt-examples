# Conditional Value-at-Risk (CVaR)

**Value-at-Risk (VaR)** is a widely used risk measure that estimates the potential loss in value of a portfolio over a defined time horizon, given a specific confidence level. VaR answers the question: "*What is the maximum loss that could occur with a certain probability over a given period?*"

While useful, VaR has several drawbacks that make it less suitable for portfolio optimization computations. These include a lack of sensitivity to tail risk, violation of subadditivity (which can penalize diversification), and the fact that it often leads to non-convex optimization problems, which are more difficult to solve reliably.

**Conditional Value-at-Risk (CVaR)** addresses these issues. It is defined as the average of all losses that exceed the VaR threshold, making it particularly effective for capturing the risk of extreme losses and providing a clearer picture of potential worst-case outcomes. Additionally, CVaR is a coherent risk measure, satisfying key mathematical properties such as subadditivity, monotonicity, and positive homogeneity, making it both theoretically sound and practically effective for optimization.

![cvar_example](../images/cvar_example.png)

This image illustrates the distribution of historical daily log-returns over a given period. The **95% VaR** (vertical dashed line) indicates that, with 95% confidence, the portfolio is not expected to lose more than 4.35% in a single day. The 95% CVaR (shaded region) measures the average loss in the worst 5% of cases, estimated at 5.58% per day.

# Mean-CVaR Problem

The Mean-CVaR optimization problem can be formulated as below. 

Starting with:

$$
\begin{aligned}
\min \quad &\lambda_{risk}\left(t + \frac{1}{1-\alpha}p^\top \mathbf{u}\right) - \mu^\top w\\
\text{s.t.} \quad & \mathbf{1}^\top \mathbf{w} +c = 1\\
&\mathbf{u} \geq -R^\top \mathbf{w} - t\\
& w^{\min} \leq w \leq w^{\max}, c^{\min} \leq c \leq c^{\max},\\
& L= \Vert w \Vert_1 \leq L^{tar}
\end{aligned}
$$
we can rewrite it as a primal-dual LP program:

$$
\begin{aligned}
\min_{x\mathbb{R}^{n}}\quad &d^\top x\\
\text{s.t.} \quad & Gx \geq h\\
&Ax = b\\
&l \leq x \leq u,
\end{aligned}
$$
where the objective is $$d = \begin{bmatrix} - \mu^\top\\0\\ \lambda_{risk}\\ \frac{\lambda_{risk}}{1-\alpha}p^\top \\ \mathbf{0}^\top \end{bmatrix}, \quad x = \begin{bmatrix} w\\c\\ t\\ u\\ v\end{bmatrix},$$ where $\mu, w, v \in \mathbb{R}^N$ are the mean return, portfolio weights, and auxiliary variable for $l1$ norm constraint, $c\in \mathbb{R}$ is the cash, $t\in \mathbb{R}$, and $u, p \in \mathbb{R}^M$ with $M$ the number of scenarios. Then, 
$$
\begin{aligned}
G = \begin{bmatrix}
R^\top & 0 & 1 & I & 0 \\
-I & 0 & 0& 0 & I\\
I & 0 & 0 & 0 & I\\ 
\end{bmatrix}, \quad 
h = \begin{bmatrix}
0 \\
0\\
0\\ 
\end{bmatrix}
\end{aligned}
$$
and the equality constraint is given as: 
$$
\begin{aligned}
A = \begin{bmatrix}
\mathbf{1}^\top & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & \mathbf{1}^\top \\
\end{bmatrix}, \quad b = \begin{bmatrix} 1\\ L^{tar}\end{bmatrix}
\end{aligned}
$$
Lastly, the lower bound is $l = \begin{bmatrix} w^{min}\\ c^{min}  \\ -\infty \\0 \\ 0  \end{bmatrix}$ and the upper bound is $$u = \begin{bmatrix} w^{max}\\ c^{max} \\ \infty \\ \infty \\ w^{max}  \end{bmatrix}$$

To model the $l1$ constraint, we use the following trick of introducing an auxiliary variable $v_i = |w_i|$ and remodel the constraint $\Vert w \Vert_1 \leq L^{tar}$ as: 
$$
\begin{aligned}
\begin{cases}
v_i \geq w_i\\
v_i \geq -w_i\\
\sum_i {v_i} = L^{tar}
\end{cases}
\end{aligned}
$$


# Usage
Notebooks in this repo provide step-by-step guide on running a portfolio optimization using cuFOLIO.
1. [Optimization with cuFOLIO](01_optimization_with_cufolio.ipynb):
This notebook demonstrates leveraging cuFOLIO's GPU acceleration for optimization and the performance gain over CPU-based methods.
2. [Backtesting](02_backtesting.ipynb):
This notebook provides a framework for backtesting strategies, allowing users to analyze historical performance.
3. [Advanced topics](03_advanced_topics.ipynb):
This notebook contains advanced topics including portfolio rebalancing strategies to maintain optimal asset allocation.

# Configure Host Networking
Edit your `/etc/hosts` file to map the `cuopt` hostname to your host machine's IP address. This allows the PyTorch container to communicate with the cuOpt service.

   ```bash
   sudo nano /etc/hosts
   ```

   Add a line like this (replace the IP with your actual host IP):

   ```
   192.168.1.123    cuopt
   ```
