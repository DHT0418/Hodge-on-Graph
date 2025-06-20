以下是按照文章脉络顺序罗列的主要公式：
1. **HodgeRank 基础公式**
   - 最小二乘法推广的 Borda 计数问题：
   \[min_{x}\left\| y - D_{0}x\right\| _{2}^{2}\]
   其中$x\in X:=\mathbb{R}^{|V|}$是全局评分分数，$D_{0}:X\to Y$是有限差分（上边缘）算子，$(D_{0}x)(\alpha,i,j)=x_{i}-x_{j}$。
   - 图拉普拉斯方程：\[D_{0}^{T}D_{0}x = D_{0}^{T}y\]
   其中\[L = D_{0}^{T}D_{0}\]是加权图拉普拉斯，$L(i,j)=m_{ij}(m_{ij}=|A_{ij}|)$（$i\neq j$），$L(i,i)=\sum_{j:(i,j)\in E}m_{ij}$
   $L^{\dagger}$是$L$的 Moore - Penrose 逆。最小范数最小二乘估计为\[\hat{x}=L^{\dagger}D_{0}^{T}y\]
2. **新的 Hodge 分解公式**
   - 定理 1（Hodge 分解定理）：
   \[y = b + u + D_{0}x + D_{1}^{T}z + w\]
   其中$b$是$y$的对称部分，$b_{ij}^{\alpha}=b_{ji}^{\alpha}=(y_{ij}^{\alpha}+y_{ji}^{\alpha})/2$；$u$是通用核，$\sum_{\alpha}u_{ij}^{\alpha}=0$；$x$是全局评分分数；$z$捕获平均三角循环；$w\in ker(D_{0}^{T})\cap ker(D_{1})$是调和排名。
   - 三角卷曲（迹）算子$D_{1}$：$(D_{1}y)(i,j,k)=\frac{1}{m_{ij}}\sum_{\alpha}y_{ij}^{\alpha}+\frac{1}{m_{jk}}\sum_{\alpha}y_{jk}^{\alpha}+\frac{1}{m_{ki}}\sum_{\alpha}y_{ki}^{\alpha}$。
3. **统计模型相关公式**
   - 广义线性模型假设：$\pi_{ij}=Prob\{i\succ j\}=\Phi(x_{i}^{*}-x_{j}^{*})$，其中$\Phi:\mathbb{R}\to[0,1]$是对称累积分布函数，通过$\hat{y}_{ij}=\Phi^{-1}(\hat{\pi}_{ij})$可将经验偏好概率$\hat{\pi}_{ij}$映射为成对比较数据。
4. **Fisher 信息最大化（无监督采样）公式**
   - 最大似然问题：$max_{x}\frac{(2\pi)^{-m/2}}{det(\sum_{\epsilon})}exp\left(-\frac{1}{2}(y - D_{0}x)^{T}\sum_{\epsilon}^{-1}(y - D_{0}x)\right)$，在假设噪声独立且方差为$\sigma_{\epsilon}^{2}$（$\sum_{\epsilon}=\sigma_{\epsilon}I_{m}$）时，等价于求解 Fisher 最大似然问题。
   - Fisher 信息：$I=-E\frac{\partial^{2}l}{\partial x^{2}}=D_{0}^{T}\sum_{\epsilon}^{-1}D_{0}=L/\sigma_{\epsilon}^{2}$。
   - 无监督采样优化目标：$max_{(\alpha_{t},i_{t},j_{t})}f(L_{t})$，其中$f$是关于边权重的凹函数且具有置换不变性，一种常见选择是$f(L_{t})=\lambda_{2}(L_{t})$，通过贪婪算法近似求解为$max\lambda_{2}(L_{t})\approx max[\lambda_{2}(L_{t - 1})+\left\|d_{t}v_{2}(L_{t - 1})\right\|^{2}]=\lambda_{2}(L_{t - 1})+max(v_{2}(i_{t}) - v_{2}(j_{t}))^{2}$，$v_{2}$是$L_{t - 1}$的第二非零特征向量（Fiedler 向量）。
5. **Bayesian 信息最大化（有监督采样）公式**
   - 正则化的 HodgeRank 问题：$min_{x}\left\|y - D_{0}x\right\|_{2}^{2}+\gamma\left\|x\right\|_{2}^{2}$，等价于$max_{x}exp\left(-\frac{\left\|y - D_{0}x\right\|_{2}^{2}}{2\sigma_{\epsilon}^{2}}-\frac{\left\|x\right\|_{2}^{2}}{2\sigma_{x}^{2}}\right)$（当$\sigma_{\epsilon}^{2}/\sigma_{x}^{2}=\gamma$）。
   - 预期信息增益（EIG）最大化：$(i^{*},j^{*}) = arg max_{(i,j)}EIG_{(i,j)}$，其中$EIG_{(i,j)}:=E_{y_{ij}^{t + 1}|y^{t}}KL(P^{t + 1}|P^{t})$，$KL(P^{t + 1}|P^{t}):=\int P^{t + 1}(x|y^{t + 1})ln\frac{P^{t + 1}(x|y^{t + 1})}{P^{t}(x|y^{t})}dx$。
   - 在$\ell_{2}$正则化的 HodgeRank 设置下，当似然和先验都是高斯分布时：
     - 后验分布$x|y^{t}\sim N(\mu^{t},\sigma_{\epsilon}^{2}\sum^{t})$，其中$\mu^{t}=(L_{t}+\gamma I)^{-1}(D_{0}^{t})^{T}y^{t}$，$\sum^{t}=(L_{t}+\gamma I)^{-1}$。
     - $2KL(P^{t + 1}|P^{t})=\frac{1}{\sigma_{\epsilon}^{2}}(\mu^{t}-\mu^{t + 1})^{T}(L_{t}+\gamma I)(\mu^{t}-\mu^{t + 1})-n+tr((L_{t}+\gamma I)(L_{t + 1}+\gamma I)^{-1})+ln\frac{det(L_{t + 1}+\gamma I)}{det(L_{t}+\gamma I)}$。
     - 利用 Sherman - Morrison - Woodbury 公式简化后的 KL - divergence：$KL(P^{t + 1}|P^{t})=\frac{1}{2}[\frac{1}{\sigma_{\epsilon}^{2}}(\frac{y_{ij}^{t + 1}-d_{t + 1}\mu^{t}}{1 + C})^{2}C + ln(1 - C)-\frac{C}{1 + C}]$，其中$C = d_{t + 1}L_{t,\gamma}^{-1}d_{t + 1}^{T}$，$\mu^{t + 1}=\mu^{t}+\frac{y_{ij}^{t + 1}-d_{t + 1}\mu^{t}}{1 + C}L_{t,\gamma}^{-1}d_{t + 1}^{T}$。

6. **在线跟踪拓扑演化相关公式**
   - 持久同调用于监测团复形$\chi_{G}$的零阶 Betti 数$\beta_{0}$（连通分量数）和一阶 Betti 数$\beta_{1}$（独立环数），以判断全局排名和调和排名情况。

7. **实验评估相关公式**
   - Kendall 秩相关系数（$\tau$）用于衡量真实排名与 HodgeRank 估计器之间的秩相关性。
   - 在计算计算复杂度和运行成本时，涉及到不同算法在不同数据集规模$n$下的运行时间统计，如模拟数据中对比了离线有监督、在线有监督和无监督算法在不同$n$值下的平均运行时间（秒）。

这些公式在文章中相互关联，从理论基础到采样策略的推导，再到实验评估，共同构成了整个研究的数学框架，用于解决众包成对排名聚合中的问题，并验证所提出方法的有效性。 