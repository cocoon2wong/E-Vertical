---
layout: pageWithLink
title: Supplemental Materials
subtitle: "Supplemental Materials for \"Another Vertical View: A Hierarchical Network for Heterogeneous Trajectory Prediction via Spectrums\""
# cover-img: /assets/img/2022-03-03/cat.jpeg
---
<!--
 * @Author: Conghao Wong
 * @Date: 2023-03-21 17:52:21
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2023-04-23 15:20:25
 * @Description: file content
 * @Github: https://cocoon2wong.github.io
 * Copyright 2023 Conghao Wong, All Rights Reserved.
-->

<link rel="stylesheet" type="text/css" href="../assets/css/user.css">

<script>
MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
        tags: 'all',
        macros: {
            bm: ["{\\boldsymbol #1}", 1],
        },
    }
};
</script>
<script id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>

Due to the limitation of the paper's length, we have omitted some of the minor analytical descriptions and experimental validations in the main paper.
These descriptions and experiments are still important, and we present them as supplementary material.

<div style="text-align: center;">
    <a class="btn btn-colorful btn-lg" href="https://github.com/cocoon2wong/E-Vertical/blob/page/assets/img/sm.pdf">⬇️ Download Supplemental Materials (PDF)</a>
</div>

## A. Transformer Details

---

We employ the Transformer [1] as the backbone to encode trajectory spectrums and the scene context in the two proposed sub-networks.
The Transformer used in the E-V$^{2}$-Net has two main parts, the Transformer Encoder and the Transformer Decoder, both of which are made up of several attention layers.

### Attention Layers

Multi-Head Attention operations are applied in each of the attention layers.
Following [1], each layer's multi-head dot product attention with $H$ heads is calculated as:

$$
    \mbox{Attention}(\bm{q}, \bm{k}, \bm{v}) = \mbox{softmax}\left(\frac{\bm{q}\bm{k}^T}{\sqrt{d}}\right)\bm{v},
$$

$$
    \begin{aligned}
        \mbox{MultiHead}&(\bm{q}, \bm{k}, \bm{v}) = \\ &\mbox{fc}\left(\mbox{concat}(\left\{ \mbox{Attention}_i(\bm{q}, \bm{k}, \bm{v}) \right\}_{i=1}^H)\right).
    \end{aligned}
$$

Here, $\mbox{fc}()$ denotes one fully connected layer that concatenates all heads' outputs.
Query matrix $\bm{q}$, key matrix $\bm{k}$, and value matrix $\bm{v}$, are the three layer inputs.
Each attention layer also contains an MLP (denoted as MLP$_a$) to extract the attention features further.
It contains two fully connected layers.
ReLU activations are applied in the first layer.
Formally, we have output feature $\bm{f}_o$ of this layer:

$$
    \bm{f}_{o} = \mbox{ATT}(\bm{q}, \bm{k}, \bm{v}) = \mbox{MLP}_a(\mbox{MultiHead}(\bm{q}, \bm{k}, \bm{v})).
$$

### Transformer Encoder

The transformer encoder comprises several encoder layers, and each encoder layer contains an attention layer and an encoder MLP (MLP$_e$).
Residual connections and normalization operations are applied to prevent the network from overfitting.
Let $\bm{h}^{(l+1)}$ denote the output of $l$-th encoder layer, and $\bm{h}^{(0)}$ denote the encoder's initial input.
For $l$-th encoder layer, the calculation of the layer output $\bm{h}^{(l+1)}$ can be written as:

$$
    \label{eq_alpha_encoder}
    \begin{aligned}
        \bm{a}^{(l)}   & = \mbox{ATT}(\bm{h}^{(l)}, \bm{h}^{(l)}, \bm{h}^{(l)}) + \bm{h}^{(l)}, \\
        \bm{a}^{(l)}_n & = \mbox{Normalization}(\bm{a}^{(l)}),                   \\
        \bm{c}^{(l)}   & = \mbox{MLP}_e(\bm{a}_n^{(l)}) + \bm{a}_n^{(l)},             \\
        \bm{h}^{(l+1)} & = \mbox{Normalization}(\bm{c}^{(l)}).
    \end{aligned}
$$

### Transformer Decoder

Similar to the Transformer encoder, the Transformer decoder comprises several decoder layers, and each is stacked with two different attention layers.
The first attention layer in the Transformer decoder focuses on the essential parts in the Transformer encoder's outputs $\bm{h}_e$ queried by the decoder's input $\bm{X}$.
The second layer is the same self-attention layer as in the encoder.
Similar to Equation 4, we have the decoder layer's output feature $\bm{h}^{(l+1)}$:

$$
    \label{eq_alpha_decoder}
    \begin{aligned}
        \bm{a}^{(l)}      & = \mbox{ATT}(\bm{h}^{(l)}, \bm{h}^{(l)}, \bm{h}^{(l)}) + \bm{h}^{(l)}, \\
        \bm{a}^{(l)}_n    & = \mbox{Normalization}(\bm{a}^{(l)}),                   \\
        \bm{a}_2^{(l)}    & = \mbox{ATT}(\bm{h}_e, \bm{h}^{(l)}, \bm{h}^{(l)}) + \bm{h}^{(l)},     \\
        \bm{a}_{2n}^{(l)} & = \mbox{Normalization}(\bm{a}_2^{(l)})                  \\
        \bm{c}^{(l)}      & = \mbox{MLP}_d(\bm{a}_{2n}^{(l)}) + \bm{a}_{2n}^{(l)},       \\
        \bm{h}^{(l+1)}    & = \mbox{Normalization}(\bm{c}^{(l)}).
    \end{aligned}
$$

### Positional Encoding

Before feeding agents representations or trajectory spectrums into the Transformer, we add the positional coding to inform the relative position of each timestep or frequency portion in the sequential inputs.
The position coding $\bm{f}_e^t$ at step $t~(1 \leq t \leq t_h)$ is obtained by:

$$
    \begin{aligned}
        \bm{f}_e^t                  & = \left({f_e^t}_0, ..., {f_e^t}_i, ..., {f_e^t}_{d-1}\right) \in \mathbb{R}^{d}, \\
        \mbox{where}~{f_e^t}_i & = \left\{\begin{aligned}
             & \sin \left(t / 10000^{d/i}\right),     & i \mbox{ is even}; \\
             & \cos \left(t / 10000^{d/(i-1)}\right), & i \mbox{ is odd}.
        \end{aligned}\right.
    \end{aligned}
$$

Then, we have the positional coding matrix $\bm{f}_e$ that describes $t_h$ steps of sequences:

$$
    \bm{f}_e = (\bm{f}_e^1, \bm{f}_e^2, ..., \bm{f}_e^{t_h})^T \in \mathbb{R}^{t_h\times d}.
$$

The final Transformer input $\bm{X}_T$ is the addition of the original sequential input $\bm{X}$ and the positional coding matrix $\bm{f}_e$.
Formally,

$$
    \bm{X}_T = \bm{X} + \bm{f}_e \in \mathbb{R}^{t_h \times d}.
$$

### Layer Configurations

We employ $L = 4$ layers of encoder-decoder structure with $H = 8$ attention heads in each Transformer-based sub-networks.
The MLP$_e$ and the MLP$_d$ have the same shape.
Both of them consist of two fully connected layers.
The first layer has 512 output units with the ReLU activation, and the second layer has 128 but does not use any activations.
The output dimensions of fully connected layers used in multi-head attention layers are set to $d$ = 128.

## B. Linear Least Squares Trajectory Prediction

---

The linear least squares trajectory prediction method aims to minimize the mean square error between the predicted and agents' groundtruth trajectories.
When predicting, we perform a separate least squares operation for each dimension of the $M$-dimensional observed trajectory $\bm{X}$.
Simply, we want to find the $\bm{x}_m = (b_m, w_m)^T \in \mathbb{R}^2~(1 \leq m \leq M)$, such that

$$
    \begin{aligned}
        \hat{\bm{Y}} &= (\hat{\bm{Y}}_1, \hat{\bm{Y}}_2, ..., \hat{\bm{Y}}_m, ..., \hat{\bm{Y}}_M) \in \mathbb{R}^{t_f \times M}, \\
        \mbox{where}~&\hat{\bm{Y}}_m = \bm{A}_f\bm{x}_m = \left(\begin{matrix}
            1 & t_h + 1 \\
            1 & t_h + 2 \\
            ... & ... \\
            1 & t_h + t_f 
        \end{matrix}\right)\left(\begin{matrix}
            b_m \\
            w_m 
        \end{matrix}\right).
    \end{aligned}
$$

For one of agents' observed $M$-dimensional trajectory $\bm{X} \in \mathbb{R}^{t_h \times M}$, we have the trajectory slice on the $m$-th dimension

$$
    \bm{X}_m = ({r_m}_1, {r_m}_2, ..., {r_m}_{t_h})^T.
$$

Suppose we have a coefficient matrix $\bm{A}_h$, where

$$
    \bm{A}_h = \left(\begin{matrix}
        1 & 1 & 1 & ... & 1 \\
        1 & 2 & 3 & ... & t_h 
    \end{matrix}\right)^T.
$$

We aim to find a $\bm{x}_m \in \mathbb{R}^{2}$, such that the mean square $\Vert \bm{A}_h \bm{x}_m - \bm{X}_m \Vert_2^2$ could reach its minimum value.
Under this condition, we have

$$
    \bm{x}_m = (\bm{A}_h^T \bm{A}_h)^{-1} \bm{A}_h^T \bm{X}_m.
$$

Then, we have the predicted $m$-th dimension trajectory

$$
    \hat{\bm{Y}}_m = \bm{A}_f \bm{x}_m.
$$

The final $M$-dimensional predicted trajectory $\hat{\bm{Y}}$ is obtained by stacking all results.
Formally,

$$
    \hat{\bm{Y}} = \bm{A}_f (\bm{x}_1, \bm{x}_2, ..., \bm{x}_M).
$$

## C. 2D DFT V.S. Bilinear Structure

---

We apply different transforms on *each dimension* of the trajectory to obtain the corresponding trajectory spectrums either in V$^{2}$-Net or the enhanced E-V$^{2}$-Net.
Moreover, considering that one of our main contributions is to establish connections between trajectories (or spectrums) of different dimensions, a more natural idea might be to apply some 2D transform directly to these trajectories.
However, it appears to be less effective from both theoretical analyses and experimental results.
In this section, we will discuss the discrepancy between the 2D transform and the proposed bilinear structure in describing the two factors, including the frequency response of the trajectory and the dimension-wise interactions, from different perspectives, taking DFT as an example.

### DFT on Different Directions in Trajectories

The 2D DFT can be decomposed into two consecutive 1D DFTs performed in different directions of the target 2D matrix.
The $M$-dimensional trajectory $\bm{X} \in \mathbb{R}^{N \times M}$ is also a 2D matrix similar to 2D grayscale images.
Although the 2D DFT and its variations have achieved impressive results in tasks related to image processing, they might be not directly applied to trajectories.
We will analyze this problem specifically by focusing on the different directions of the transforms in the trajectory.

<div style="text-align: center;">
    <img style="width: 100%;" src="../assets/img/appendix_1.png">
    Fig. 1. Matrices views of a trajectory (2D bounding box) and an image.
</div>

Fig. 1 shows an $M=4$ 2D bounding box trajectory and an image with the matrix view.
As shown in Fig. 1 (b), whether the image is sliced horizontally or vertically, the resulting vector could reflect the change in grayscale values in a particular direction.
Therefore, when performing the 2D transform, the first 1D transform will extract the frequency response in a specific direction, while the second 1D transform will fuse it with the frequency response in the vertical direction.

In contrast, different slice directions of the trajectory may lead to different meanings.
If the trajectories are sliced according to the time dimension, then four 1D time series will be obtained as shown in Fig. 1 (a).
Applying 1D transforms to these four sequences, we can obtain four trajectory spectrums that could describe agents' frequency responses and thus describe their motions from the global plannings and interaction details at different scales.
However, if the trajectory is sliced from the dimensional direction, then $N$ (6 in the figure) 4-dimensional vectors will be obtained.
These vectors contain information about agents' locations and postures at a particular moment.
In addition, the focused dimension-wise interactions are also contained in these vectors.
However, it should be noted that we are more interested in the relationships between the data in these vectors, *i.e.*, the "edges" between the different data.
If a 1D transform is applied to these vectors, the resulting spectrum may hardly have a clear physical meaning, because the temporal or spatial adjacencies of these points are not reflected in these 4-dimensional vectors.

For example, suppose we want to apply the 1D DFT on the 4-dimensional (2D bounding box) vector $\bm{x} = (x_l, y_l, x_r, y_r)^T$.
Simply, we have:

$$
    \begin{aligned}
        \mathcal{X} = \mbox{DFT}[\bm{x}] &= 
            \begin{pmatrix}
                1 & 1 & 1 & 1 \\
                1 & -j & -1 & j \\
                1 & -1 & 1 & -1 \\
                1 & j & -1 & -j \\
            \end{pmatrix}
            \begin{pmatrix}
                x_l \\ y_l \\ x_r \\ y_r \\
            \end{pmatrix} \\
        &= \begin{pmatrix}
                x_l + y_l + x_r + y_r \\
                x_l - x_r - j(y_l - y_r) \\
                x_l - y_l + x_r - y_r \\
                x_l - x_r + j(y_l - y_r) \\
            \end{pmatrix}.
    \end{aligned}
$$

Accordingly, we have its fundamental frequency portion $\mathcal{X}[0] = x_l + y_l + x_r + y_r$ and the high-frequency portion $\mathcal{X}[2] = x_l - y_l + x_r - y_r$.
However, since the four positions $\{x_l, y_l, x_r, y_r\}$ do not have specific time-dependent or space-dependent like time-sequences and images, these frequency components may hardly reflect the specific frequency response.
For example, the fundamental frequencies can represent their average value for a time series, yet the values obtained by directly summing the 4 position coordinates of the 2 points of the 2D bounding box would be uninterpretable.
In other words, each element in this 4-dimensional vector is relatively independent, and their connection relationships are more like a *graph* rather than a sequence where an order is required.

### Quantitative Analyses

<div style="text-align: center;">
    TABLE 1<br>
    Validation of 2D DFT and bilinear structures with best-of-20 on SDD (2D bounding box) the nuScenes (3D bounding box).
    <img style="width: 100%;" src="../assets/img/appendix_2.png">
</div>

To further verify our thoughts, we perform ablation experiments on SDD and nuScenes to compare the effects of 2D DFT and the bilinear structure quantitatively.
As shown in TABLE 1, the results of APP2 and APP3 (or APP5 and APP6) show that 2D DFT does not improve quantitative trajectory prediction performance as effectively as bilinear structures.
On the contrary, in the prediction of the more complex 3D bounding boxes ($M = 6$), using 2D DFT instead degrades the prediction performance compared to 1D DFT when no bilinear structures are used.
These experimental results validate our thoughts of not using 2D transforms but bilinear structures.

### Qualitative Analyses

<div style="text-align: center;">
    <img style="width: 100%;" src="../assets/img/appendix_3.png">
    Fig. 2. Visualized comparisons of 2D DFT bilinear structure.
</div>

We visualize the prediction results of different models under the effect of 2D DFT and bilinear structure qualitatively.
As shown in Fig. 2, the V$^{2}$-Net (2D DFT) performs not as well as E-V$^{2}$-Net in both the prediction of agent motions and the interactions within the bounding box.
In detail, predictions given by V$^{2}$-Net (2D DFT) capture fewer path possibilities in the top left prediction scenario.
In addition, some predicted trajectories are with less smoothness and naturalness.
For example, the prediction in color <strong style="color: #DF6091;">\#DF6091</strong> to the left of the bottom left prediction scene gives a turn with a large angle to observation, which could be not physical-acceptable in the actual scenario.
In contrast, predictions given by E-V$^{2}$-Net have not shown similar results in this scenario.
On the other hand, as shown in the traffic circle prediction scenario on the right, the shape of the bounding box is not well maintained in V$^{2}$-Net's predictions, such as the prediction in color <strong style="color: #93C3CA;">\#93C3CA</strong> to turn right to across the street.

## D. A Graph-View of Bilinear Structure

---

As mentioned above, the modeling of dimension-wise interactions focuses more on the relations of different trajectory dimensions, which could be difficult to represent by the 1D transform.
The bilinear structure used in this manuscript learns this connection relation through the outer product, pooling, and fully connected networks.
To make it easier to understand, we further explain it from a graph view.

Given an undirected graph $\bm{G}(t) = (\bm{V}(t), \bm{E}(t))$ with a time variable $t$, where $\bm{V}(t)$ is the set of vertices, which contains all the position information of one agent at time $t$, and $\bm{E}(t)$ is the set of edges, which represents the connection relationships between these vertices at time $t$.
Formally,

$$
    \begin{aligned}
        \bm{V}(t) &= \left\{f({r_1}_t), f({r_2}_t), ..., f({r_M}_t)\right\} \\
        &= \left\{\bm{f}_{1, t}, \bm{f}_{2, t}, ..., \bm{f}_{M, t}\right\}.
    \end{aligned}
$$

where 

$$
    \label{eq_appendix_embed}
    \bm{f}_{m, t} = f({r_m}_t) \in \mathbb{R}^d
$$

indicates an embedding function to map these vertices into the high-dimension feature space.
To establish and learn the connections between these vertices, we define the trainable adjacency matrix as:

$$
    \bm{A}(t) = \begin{pmatrix}
        \bm{W}_{1, 1}(t) & \cdots & \bm{W}_{1, M}(t) \\
        \vdots & \ddots & \vdots \\
        \bm{W}_{M, 1}(t) & \cdots & \bm{W}_{M, M}(t) \\
    \end{pmatrix}.
$$

Here, each matrix $$\bm{W}_{i, j} \in \mathbb{R}^{d \times d}$$ are made trainable.
They are used to describe the relation between node $i$ and node $j$ (*i.e.*, $$\bm{f}_{i, t}$$ and $$\bm{f}_{j, t}$$).

We converge the information on the node edges by graph convolution.
Formally,

$$
    \label{eq_appendix_graphConv}
    \bm{f}'_{m, t} = \sigma \left( \bm{W}'_m \mbox{Flatten} \left( \sum_{j=1}^{M} \bm{W}_{m, j}(t) \bm{f}_{m, t} \otimes \bm{f}_{j, t} \right) \right),
$$

where $\sigma$ represents a non-linear activation, and the $\bm{W}_m'$ is another trainable weight matrix.
Finally, we have the refined vertices

$$
    \bm{V}'(t) = \left\{\bm{f}'_{1, t}, \bm{f}'_{2, t}, ..., \bm{f}'_{M, t}\right\}.
$$

It is worth noting that the above Equation 19 and the bilinear structure introduced in the manuscript describe the same network structure, despite their difference in representation.
To reduce unnecessary misunderstandings, we do not describe the network inference process through this graph form in the manuscript, although the use of a graph may make it easier to understand the motivation for the use of the bilinear model.
In addition, all the operations above are performed on time series.
If we use trajectory spectrums to replace the Equation 17, and take the frequency variable $n$ to instead the time variable $t$, we have

$$
    \mathcal{V}(n) = \{\bm{f}_{m, n}\}_{m=1}^{\mathcal{M}},\quad\mbox{where}~\bm{f}_{m, n} = f(s_{n, m}).
$$

Accordingly, we have the refined vertices' spectrum representations:

$$
    \mathcal{V}'(t) = \left\{\bm{f}'_{1, n}, \bm{f}'_{2, n}, ..., \bm{f}'_{\mathcal{M}, n}\right\}.
$$

Then, the outer product matrix $\bm{R}[n, :, :]$ has become the adjacency matrix of the graph $\bm{G}(n) = (\mathcal{V}(n), \mathcal{E}(n))$ on the frequency node $n \in [1, \mathcal{N}_h]$.


## References

---

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in neural information processing systems, 2017, pp. 59986008.