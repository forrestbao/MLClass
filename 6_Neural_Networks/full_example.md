---
header-includes:
  \hypersetup{colorlinks=true,
            allbordercolors={.5 .5 1},
            pdfborderstyle={/S/U/W 1}}
  \usepackage{amssymb,mathtools,blkarray,bm}
  \usepackage[vmargin={.35in,.35in},hmargin={.35in,.35in}]{geometry}
---

# Solution for HW6

Copyright 2020-2021 Forrest Sheng Bao

9. $\dim(\mathbb{W}^{(0)}) = 3\times 3$, $\dim(\mathbb{W}^{(1)}) = 4\times 2$, and $\dim(\mathbb{W}^{(2)}) = 3\times 2$

10. $\mathbf{x}^{(1)}_{[1..]} = \phi(\mathbb{W}^{(0)T}\mathbf{x}^{(0)}) = 
    \phi \left (
    \begin{pmatrix}
    0.1 & 0.1 & 0.1 \\
    0.1 & 0.1 & 0.1 \\
    0.1 & 0.1 & 0.1
    \end{pmatrix}^T
    \begin{pmatrix}
    1\\
    1\\
    1
    \end{pmatrix}
    \right )
    =
    \phi 
    \begin{pmatrix}
    0.3\\
    0.3\\
    0.3
    \end{pmatrix}
    =
    \begin{pmatrix}
    0.574\\
    0.574\\
    0.574
    \end{pmatrix}$

    $\mathbf{x}^{(2)}_{[1..]} = \phi(\mathbb{W}^{(1)T}\mathbf{x}^{(1)}) = 
    \phi \left (
    \begin{pmatrix}
    2 & 2 \\
    2 & 2 \\
    2 & 2 \\
    2 & 2
    \end{pmatrix}^T
    \begin{pmatrix}
    1\\
    0.574\\
    0.574\\
    0.574
    \end{pmatrix}
    \right )
    =
    \phi 
    \begin{pmatrix}
    5.444\\
    5.444\\
    \end{pmatrix}
    =
    \begin{pmatrix}
    0.99569644\\
    0.99569644
    \end{pmatrix}
    \approx
    \begin{pmatrix}
    0.996\\
    0.996
    \end{pmatrix}$

    $\mathbf{x}^{(3)} = \phi(\mathbb{W}^{(2)T}\mathbf{x}^{(2)}) = 
    \phi \left (
    \begin{pmatrix}
    1 & 1 \\
    1 & 1 \\
    1 & 1
    \end{pmatrix}^T
    \begin{pmatrix}
    1\\
    0.996\\
    0.996
    \end{pmatrix}
    \right )
    =
    \phi 
    \begin{pmatrix}
    2.992\\
    2.992\\
    \end{pmatrix}
    =
    \begin{pmatrix}
    0.952\\
    0.952
    \end{pmatrix}
    =\hat{\mathbf{y}}$

11. Because $\phi(x) = {1 \over 1+\exp(-x)}$, $\psi(x) = \phi'(\phi^{-1}(x)) = \phi(x)(1-\phi(x))$

    $\bm{\delta}^{(3)} = \hat{\mathbf{y}} - [1,0]^T = [-0.048,  0.952]^T$

    $\bm{\delta}^{(2)} = \psi(\mathbf{x}^{(2)}) \circ \left ( \mathbb{W}^{(2)} \bm{\delta}^{(3)} \right ) = \mathbb{W}^{(2)} \bm{\delta}^{(3)} \circ \psi(\mathbf{x}^{(2)}) 
    = 
    \begin{pmatrix}
        1 & 1 \\
        1 & 1 \\
        1 & 1
    \end{pmatrix}
    \begin{pmatrix}
        -0.048\\
        0.952
    \end{pmatrix}
    \circ
    \begin{pmatrix}
        1\\
        0.996\\
        0.996
    \end{pmatrix}
    \circ
    \begin{pmatrix}
        1-1\\
        1-0.996\\
        1-0.996
    \end{pmatrix}
    =
    \begin{pmatrix}
    0\\
    0.00360154\\
    0.00360154
    \end{pmatrix}
    \approx 
    \begin{pmatrix}
    0\\
    0.004\\
    0.004
    \end{pmatrix}$

    $\bm{\delta}^{(1)} = \psi(\mathbf{x}^{(1)}) \circ \left ( \mathbb{W}^{(1)} \bm{\delta}^{(2)}_{[1..]} \right ) = \mathbb{W}^{(1)} \bm{\delta}^{(2)}_{[1..]} \circ \psi(\mathbf{x}^{(1)}) 
    = 
    \begin{pmatrix}
        2 & 2 \\
        2 & 2 \\
        2 & 2 \\
        2 & 2 
    \end{pmatrix}
    \begin{pmatrix}
        0.004\\
        0.004
    \end{pmatrix}
    \circ
    \begin{pmatrix}
        1\\
        0.574\\
        0.574\\
        0.574
    \end{pmatrix}
    \circ
    \begin{pmatrix}
        1-1\\
        1-0.574\\
        1-0.574\\
        1-0.574\\
    \end{pmatrix}
    =
    \begin{pmatrix}
    0\\
    0.00391238\\
    0.00391238\\
    0.00391238
    \end{pmatrix}
    \approx 
    \begin{pmatrix}
    0\\
    0.004\\
    0.004\\
    0.004
    \end{pmatrix}$

12. $\nabla^{(2)} = 
  \mathbf{x}^{(2)}  \left ( \bm{\delta}^{(3)} \right )^T
  = 
  \begin{pmatrix}
   1\\
   0.996\\
   0.996
   \end{pmatrix}
   [-0.048, 0.952]=
   \begin{pmatrix}
   -0.048  &  0.952\\
   -0.047808 & 0.948192\\
   -0.047808 &  0.948192
   \end{pmatrix}
   \approx
   \begin{pmatrix}
   -0.048  &  0.952\\
   -0.048 & 0.948\\
   -0.048 &  0.948
   \end{pmatrix}$

    $\nabla^{(1)} = 
    \mathbf{x}^{(1)}  \left ( \bm{\delta}^{(2)}_{[1..]} \right )^T
    = 
    \begin{pmatrix}
    1\\
    0.574\\
    0.574\\
    0.574
    \end{pmatrix}
    [0.004, 0.004]=
    \begin{pmatrix}
    0.004  & 0.004   \\
    0.002296 & 0.002296\\
    0.002296 & 0.002296\\
    0.002296 & 0.002296
    \end{pmatrix}
    \approx
    \begin{pmatrix}
    0.004  & 0.004   \\
    0.002 & 0.002\\
    0.002 & 0.002\\
    0.002 & 0.002
    \end{pmatrix}$

    $\nabla^{(0)} = 
    \mathbf{x}^{(0)}  \left ( \bm{\delta}^{(1)}_{[1..]} \right )^T
    = 
    \begin{pmatrix}
    1\\
    1\\
    1
    \end{pmatrix}
    [0.004, 0.004, 0.004]=
    \begin{pmatrix}
    0.004  & 0.004 & 0.004  \\
    0.004  & 0.004 & 0.004  \\
    0.004  & 0.004 & 0.004 
    \end{pmatrix}$    

13. $\mathbb{W}^{(2)} \leftarrow \mathbb{W}^{(2)}  - \rho 
  \nabla^{(2)}= 
    \begin{pmatrix}
    1 & 1 \\
    1 & 1  \\
    1 & 1 
    \end{pmatrix}
    -
    1\cdot 
    \begin{pmatrix}
    -0.048  &  0.952\\
    -0.048 & 0.948\\
    -0.048 &  0.948
    \end{pmatrix}
    = 
    \begin{pmatrix}
    1.048 & 0.048\\
    1.048 & 0.052\\
    1.048 & 0.052
    \end{pmatrix}$

    $\mathbb{W}^{(1)} \leftarrow \mathbb{W}^{(1)}  - \rho 
    \nabla^{(1)}= 
    \begin{pmatrix}
    2 & 2 \\
    2 & 2  \\
    2 & 2 \\
    2 & 2 
    \end{pmatrix}
    -
    1\cdot 
    \begin{pmatrix}
    0.004  & 0.004   \\
    0.002 & 0.002\\
    0.002 & 0.002\\
    0.002 & 0.002
    \end{pmatrix}
    = 
    \begin{pmatrix}
    1.996 & 1.996 \\
    1.998 & 1.998 \\
    1.998 & 1.998 \\
    1.998 & 1.998
    \end{pmatrix}$

    $\mathbb{W}^{(0)} \leftarrow \mathbb{W}^{(0)}  - \rho 
    \nabla^{(0)}= 
    \begin{pmatrix}
    0.1 & 0.1 & 0.1 \\
    0.1 & 0.1 & 0.1 \\
    0.1 & 0.1 & 0.1 
    \end{pmatrix}
    -
    1\cdot 
    \begin{pmatrix}
    0.004  & 0.004 & 0.004  \\
    0.004  & 0.004 & 0.004  \\
    0.004  & 0.004 & 0.004 
    \end{pmatrix}
    = 
    \begin{pmatrix}
    0.096 & 0.096 & 0.096 \\
    0.096 & 0.096 & 0.096 \\
    0.096 & 0.096 & 0.096
    \end{pmatrix}$

