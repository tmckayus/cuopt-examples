# cuFOLIO: GPU-Accelerated Portfolio Optimization

cuFOLIO is an accelerated computing library designed to accelerate investment portfolio construction and management. cuFOLIO leverages NVIDIA's cutting-edge technologies including cuOpt, RAPIDS, and the HPC SDK to deliver substantial performance gains in portfolio optimization.

cuFOLIO is built to leverage and extend key components of the NVIDIA accelerated computing stack for financial optimization workflows. The components we integrated are illustrated below:

![cuFOLIO](./images/cufolio.png)

## Portfolio Management and Portfolio Optimization

Effective Portfolio Management is fundamental for investment-focused institutions, including both buy-side and sell-side operations across various asset classes. It involves the strategic creation, allocation, and management of financial asset portfolios, aiming to balance risk and return to achieve specific investment objectives. As financial markets become increasingly complex and diverse, the necessity for advanced portfolio optimization techniques and tools has grown correspondingly.

A central aspect of portfolio management is **Portfolio Optimization**, which focuses on constructing portfolios that maximize expected returns for a given level of risk. This process entails solving high-dimensional, non-linear numerical optimization problems that are computationally intensive.

GPU-accelerated solutions allow financial institutions to navigate the complexities of modern markets more effectively, optimizing portfolio performance while managing risk with greater precision.

## Portfolio Optimization Workflow
![Portfolio Optimization workflow](./images/PO-workflow.png)

## cuOpt for GPU-Accelerated Optimization

cuFOLIO utilizes NVIDIA cuOpt, a GPU-accelerated combinatorial and linear optimization engine for solving complex route optimization problems such as Vehicle Routing and large Linear Programming problems. cuOpt is designed to tackle large-scale problems with millions of variables and constraints, enabling near-real-time optimization and driving significant cost savings.

## Optimization Methods

The current implementation of cuFOLIO supports Conditional Value-at-Risk (CVaR), with support for mean-variance optimization under active development.

### Conditional Value-at-Risk (CVaR)

Conditional Value-at-Risk (CVaR), also known as Expected Shortfall, provides a more comprehensive view of tail risk than traditional Value-at-Risk (VaR). While VaR estimates the maximum expected loss at a given confidence level (e.g., 95%), it doesn't take the magnitude of losses beyond that threshold into account. In contrast, CVaR captures the average of those extreme losses that exceed the VaR cutoff, offering a more robust and informative measure of downside risk. This makes CVaR particularly valuable for understanding true risk exposure, as it addresses the limitations of VaR by focusing on extreme loss scenarios.

### Mean-Variance Optimization (Future Support Planned)

Mean-variance optimization, pioneered by Harry Markowitz, evaluates portfolios by balancing expected return against variance, providing a trade-off between risk and return. Mean-variance optimization is a quadratic problem and not supported in cuFOLIO at this time.
