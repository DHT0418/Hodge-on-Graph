# Hodge + GNN

## 背景
多数图神经网络模型遵循消息传递框架，聚合邻域信息更新节点状态。但面对大规模图数据时，存在内存需求大、计算效率低、受异质性影响等问题，图采样算法成为解决关键。
现有采样算法存在结构信息提取不完全、未充分利用特征和结构信息、不能适用于不同模型等问题。


## 1.图结构信息
以霍奇分数作为节点结构重要性评判方法，定义霍奇节点重要性，考虑节点及邻域节点重要性，通过节点间霍奇重要性差定义节点相互影响，霍奇重要性可捕捉全局结构信息。
### Hodge数学公式推导与计算过程
霍奇分解是将图上的边流分解为梯度流、旋度流和调和流。以下是基于上述代码示例的数学公式推导及计算过程： 
##### 1).有向图表示与基础定义

设有向图 $G = ( V , E ) $, 其中 $V$ 是节点集合， $E$ 是 边集合。对于每条边 $e = ( u , v ) \in E $, 有一个对应的权重 $w ( e ) $。 
·在示例中，创建的有向图 $G$ 有 $V = \{ 0 , 1 , 2 , 3 \}$ 和 $E = \{ ( 0 , 1 ) , ( 1 , 2 ) , ( 2 , 3 ) , ( 3 , 0 ) \} $, 且每条边权重为1。 

##### 2).梯度矩阵计算
梯度矩阵 $G$ 的定义为：对于边 $e = ( u , v ) , G _ { e , u } = 1 , G _ { e , v } = - 1 $, 其他位置为0。 
数学公式表示为： $G _ { i j } = \left\{ \begin{array} { l l } 1 , & \text{if} e _ { j } = \left( v _ { i } , v _ { k } \right) \in E \\ - 1 , & \text{if} e _ { j } = \left( v _ { k } , v _ { i } \right) \in E \\ 0 , & \text{otherwise} \end{array} \right. $ 
在示例中，计算得到的梯度矩阵为： 
\[G = \left[ \begin{array} { c c c c } 1 & - 1 & 0 & 0 \\ 0 & 1 & - 1 & 0 \\ 0 & 0 & 1 & - 1 \\ - 1 & 0 & 0 & 1 \end{array} \right]\] 
##### 3).旋度矩阵计算
旋度矩阵 $C$ 的定义与图中的圈结构相关。对于一个简单的圈 $O = \left( v _ { 1 } , v _ { 2 } , \cdots , v _ { n } , v _ { 1 } \right) ,$ 如果边 $e = \left( v _ { i } , v _ { i + 1 } \right)$ 在圈中正向， $C _ { O , e } = 1 $; 如果边 $e = \left( v _ { i + 1 } , v _ { i } \right)$ 在圈中反向， $C _ { O , e } = - 1 $; 其他边对应的位置为0。 ·在示例中，只有一个圈 $( 0 , 1 , 2 , 3 , 0 ) $, 计算得到的旋度矩阵为： 

\[C = \left[ \begin{array} { l l l l } 1 & - 1 & 1 & - 1 \end{array} \right] ^ { T }\] 

##### 4).有向图上的霍奇分解定理
对于任意的有向图 $\mathcal { D } \mathcal { G } \in \Omega ( V )$ 以及有向图上的任意边流 $\varphi \in C _ { 1 } \left( E _ { \mathcal { D G } } \right) ,$ 存在唯一的点函数 $\phi ^ { \varphi } \in C _ { 0 } \left( V _ { \mathcal { D G } } \right) ,$ 其满足 $\sum _ { v _ { i } \in V _ { D G } } \phi ^ { \varphi } \left( v _ { i } \right) = 0 ;$ 唯一的边函数 $h ^ { \varphi } \in C _ { 1 } \left( E _ { D G } \right)$ 和唯一的圈函数 $\psi ^ { \varphi } \in C _ { 2 } \left( \mathcal { O } _ { \mathcal { D G } } \right) ,$ 使得 \[\vec { \varphi } \left( E _ { \mathcal { D G } } \right) = G ( \mathcal { D } \mathcal { G } ) \vec { \phi } ^ { \varphi } \left( V _ { \mathcal { D G } } \right) + C ( \mathcal { D } \mathcal { G } ) ^ { T } \vec { \psi } ^ { \varphi } \left( \mathcal { O } _ { \mathcal { D G } } \right) + \vec { h } ^ { \varphi } \left( E _ { \mathcal { D G } } \right)\]

其中，边函数满足
 \[G ^ { T } ( \mathcal { D } \mathcal { G } ) \overrightarrow { h ^ { \phi } } \left( E _ { \mathcal { D G } } \right) = \vec { 0 }\] 以及
 \[C ( \mathcal { D } \mathcal { G } ) \overrightarrow { h ^ { \phi } } \left( E _ { \mathcal { D G } } \right) = \vec { 0 }\]

##### 5).霍奇分数 与 霍奇排序
如果点函数 $\phi ^ { \varphi }$ 是边函数 $\varphi$ 经过霍奇分解得到的，则点函数 $\phi ^ { \varphi }$ 满足： 
\[G ^ { T } ( \mathcal { D } \mathcal { G } ) G ( \mathcal { D } \mathcal { G } ) \vec { \phi } ^ { \varphi } \left( V _ { D G } \right) = G ^ { T } ( \mathcal { D } \mathcal { G } ) \vec { \varphi } \left( E _ { D G } \right)\] 
它就是霍奇分数向量，有霍奇分数向量，就可以得到霍奇排序

## 2.图节点特征相似度
用图注意力网络计算节点间特征相似度，通过权重矩阵变换和注意力函数映射，经softmax归一化得到。
给定图 $G$ 和节点特征的节点集 $X = \{ \vec{x}_1, \vec{x}_2, \ldots, \vec{x}_m \}, \vec{x}_i \in \mathbb{R}^F$，其中 $m$ 是节点数量，$F$ 是节点特征的维度。首先，对每个节点应用权重矩阵 $W \in \mathbb{R}^{F^{'} \times F}$。
然后，注意力函数 $a: \mathbb{R}^{F \times F} \rightarrow \mathbb{R}$ 拼接后的高维特征映射到一个实数上来计算特征相似度：

$$
f_{ij} = a(W \vec{x}_i, W \vec{x}_j)
\tag{5.1}
$$

这表示节点 $v_i$ 和节点 $v_j$ 之间的特征相似度。然后，使用 $softmax$ 函数对 $v_i$ 的所有特征相似度进行归一化：

$$
\alpha_{ij}^*  = \frac{\exp(f_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(f_{ik})}
\tag{5.2}
$$

$v_i$ 和 $v_j$ 之间的特征相似度是：
$$
\alpha_{ij}^* = \frac{\exp( LeakyReLU( \vec{a}^T [ W \vec{x}_i \| W \vec{x}_j ] ))}{\sum_{k \in \mathcal{N}_i} \exp( LeakyReLU( \vec{a}^T [ W \vec{x}_i \| W \vec{x}_k ] ))}
\tag{5.3}
$$

## 3.基于特征信息和结构信息的通用采样策略
综合特征相似度和霍奇重要性得到综合邻近度，通过softmax操作和相乘计算，FSS根据综合邻近度为节点采样top k个邻居形成采样子图，该策略即插即用，可用于下游模型训练。

## 4.实验设计与结果分析
    - **对比实验**：FSS应用于不同模型在多个数据集上与多种基线模型对比，在多个数据集上性能优异，综合考虑特征和结构信息，减少异质边影响，提升模型性能。
    - **采样策略有效性实验**：FSS采样子图在多数数据集上运行时间短，能加速训练；可提高图同质性，消除异质边影响，提升模型性能。
    - **消融实验**：FSS集成到多种模型中提升分类准确率；特征相似度和霍奇重要性对模型性能有正面影响，霍奇重要性影响更显著；霍奇重要性比PageRank重要性更能准确衡量节点重要性，图注意力网络特征相似度优于余弦相似度；参数k影响FSS性能，选择合适k值很重要；FSS计算复杂度主要源于注意力分数和霍奇重要性计算，运行时间短，可方便嵌入其他模型；可视化结果表明FSS有助于提升节点表示效果和模型分类能力。

### （二）未来研究方向
1. 在有向图数据集上进行基于特征信息和结构信息的采样策略实验，进一步探索霍奇分解在有向图数据中的应用效果，拓展研究成果的适用范围。
2. 改进节点之间的特征相似度计算方法，引入异质图处理技术，尝试在频域内更精确计算特征相似度，以提升模型性能，使模型在处理复杂图数据时更加准确和有效。
3. 探索霍奇评分排序方法及融合特征信息和结构信息的通用采样算法在社交网络、推荐系统等领域的应用，充分发挥其在实际场景中的价值，如优化社交网络中的节点重要性评估、提高推荐系统的准确性等。
