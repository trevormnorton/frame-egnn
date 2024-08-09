# Frame EGNN

This repository is an experimental extension of the model detailed in [E(n)-Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844v1). The original paper treats the nodes as point masses. This can be extended to a model where the nodes are rigid bodies (similar to the frames used in AlphaFold2). The graph neural network has node embeddings $\mathbf{h}^l$, coordinate embeddings $\mathbf{x}^l$, and orientation embeddings $\mathbf{R}^l$ for each layer $l$. Each $x^l_i$ is an element of $\mathbb R^3$ and each element $R^l_i$ is in $\mathrm{SO}(3)$ (3x3 orthogonal matrices with determinants 1). Updates are given by 

```math
\begin{align*}
m_{ij} &= \phi_e\left(h_i^l, h_j^l, \| x_i - x_j\|^2, (R_i^l)^{-1}R_j^l, (R_i^l)^{-1}\frac{x_j^l - x_i^l}{\|x_j^l - x_i^l\|}, a_{ij}\right) \\
x_i^{l+1} &= x_i^l + \sum_{j\neq i}(x_i^l - x_j^l)\phi_x(m_{ij}) \\
R_i^{l+1} &= R_i^l \exp\left( \sum_{j\neq i}\sum_{k=1,2,3} \phi_R^k(m_{ij}) \hat{R}_k \right) \\
m_i &= \sum_{j\in\mathcal N(i)} m_{ij} \\
h_i^{l+1} &= \phi_h(h_i^l, m_i)
\end{align*}
```

The $\hat R_k$'s are a basis on $\mathfrak{so}(3)$. The implementation of the orientations is done using quaternions and the library [RoMa](https://github.com/.naver/roma)
