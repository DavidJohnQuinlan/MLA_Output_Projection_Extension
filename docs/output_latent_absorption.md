# Output-Latent Absorption
## Does the inclusion of the output latent buy anything at inference time?
One of the claims from the DeepSeek-V2 paper was that at inference time a matrix absorption trick could be employed to avoid decompressing the $k$ and $v$ latent vectors back to their full dimensional size. Therefore, only the compressed latent representation needs to be cached, which leads to large memory savings.

If a write-side bottleneck (the output latent) were added, does a similar matrix absorption trick exist, and if so, is the combined trick cheaper than the read-side bottle neck (MLA) along?

Given the Q/K/score path is identical across all variants, I explore only the value to output path. 

The four variants:

1. **MHA**: dense value $\mathbf{v}_{t,i}=W^{V}_{(i)}\mathbf{h}_t$, dense output $\mathbf{u}_t=W^{O}[\mathbf{o}_{t,1};\ldots;\mathbf{o}_{t,n_h}]=\sum_{i}W^{O}_{(i)}\mathbf{o}_{t,i}$.
2. **MLA**: value from KV latent $\mathbf{v}_{t,i}^{C}=W^{UV}_{(i)}\mathbf{c}_t^{KV}$, dense output $\mathbf{u}_t=\sum_{i}W^{O}_{(i)}\mathbf{o}_{t,i}$.
3. **MHAE**: dense value $\mathbf{v}_{t,i}=W^{V}_{(i)}\mathbf{h}_t$, latent output $\mathbf{u}_t=W^{UO}\,\text{LayerNorm}\big(\sum_{i}W^{DO}_{(i)}\mathbf{o}_{t,i}\big)$.
4. **MLAE**: latent value $\mathbf{v}_{t,i}^{C}=W^{UV}_{(i)}\mathbf{c}_t^{KV}$, latent output $\mathbf{u}_t=W^{UO}\,\text{LayerNorm}\big(\sum_{i}W^{DO}_{(i)}\mathbf{o}_{t,i}\big)$.


## Notation

The table below lists the variable names used in the derivations.

| symbol | meaning |
|---------|------|
| $D$ | Down (compress) |
| $U$ | Up (decompress) |
| $(i)$ | Per head block of the projection matrix |
| $d$ | Model dimension |
| $d_h$ | Attention head dimension | 
| $n_h$ | Number of attention heads |
| $d_{kv}$ | Key/Value latent compression dimension |
| $d_o$ | Output latent compression dimension |
| $t$ | Current token position / sequence length |
| $W^{DKV}$| Key/Value down projection matrix | 
| $W^{UK}$ | Key up projection matrix |
| $W^{UV}$ | Value up projection matrix |
| $W^{DQ}$ | Query down projection matrix | 
| $W^{UQ}$ | Query up projection matrix | 
| $W^V$    | Value projection matrix |
| $W^O$    | Output projection matrix |
| $W^{DO}$ | Output down projection matrix | 
| $W^{UO}$ | Output up projection matrix | 
| $W^{QK}$ | Projection matrix combining the query and key up projection matrices |
| $W^{O'}_{(i)}$ | Projection matrix combining the value up projection and dense output projection matrices |
| $W^{DOUV}_{(i)}$ | Projection matrix combining the value up projection and the output down projection matrices |
| $\mathbf{c}_t^{KV}$ | Compressed Key/Value latent for token $t$, shared across heads |
| $\mathbf{o}_{t,i}$ | Per-head attention output at position $t$, before output projection |
| $\mathbf{u}_t$ | Attention-block output at position $t$ after the output projection |
| $\mathbf{p}_t^{(i)}$ | Per-head "mixed latent", the attention-weighted sum of Key/Value latents |
| $\mathbf{a}_{t,i,j}$ | Attention weight from query position $t$ to key $j$ in head $i$ (softmax output) |

## Preliminaries

Since decoupled RoPE has not yet been implemented in this repository, it is not included in the derivations below. Similarly, the LayerNorm is not mentioned in Appendix C of the DeepSeek-V2 paper, but is included here for completeness. It does not affect DeepSeek's MLA implementation since it is applied before the matrix absorption trick. It does, however, affect how much absorption is possible in the output latent compress-decompress variant. Because the LayerNorm is located between the head-sum and $W^{UO}$, the path cannot fully collapse into a single per head operator as is the case in MLA. However, a reduction relative to MLA is still achieved because $W^{UO}$ is shared and each per head operator is confined to the $d_o$ bottleneck rather than expanding to full model width.

## Breakdown of MLA Absorption Trick
Here is a reminder of how the MLA absorption trick was implemented for DeepSeek-V2:

$$
\begin{align*}
\mathbf{c}_t^Q &= \text{LayerNorm}(W^{DQ}\mathbf{h}_t) \\[4pt]
[\mathbf{q}_{t,1}^C;\mathbf{q}_{t,2}^C;\ldots;\mathbf{q}_{t,n_h}^C] = \mathbf{q}_t^C &= W^{UQ}\mathbf{c}_t^Q \\[4pt]

\mathbf{c}_t^{KV} &= \text{LayerNorm}(W^{DKV}\mathbf{h}_t) \\[4pt]
[\mathbf{k}_{t,1}^C;\mathbf{k}_{t,2}^C;\ldots;\mathbf{k}_{t,n_h}^C] = \mathbf{k}_t^C &= W^{UK}\mathbf{c}_t^{KV} \\[4pt]
[\mathbf{v}_{t,1}^C;\mathbf{v}_{t,2}^C;\ldots;\mathbf{v}_{t,n_h}^C] = \mathbf{v}_t^C &= W^{UV}\mathbf{c}_t^{KV} \\[4pt]
\mathbf{q}_{t}^T\mathbf{k}_{j} &= (W^{UQ}c_{t}^{Q})^{T}(W^{UK}c_{j}^{KV}) \\[4pt]
\mathbf{q}_{t}^T\mathbf{k}_{j} &= c_{t}^{QT}(W^{UQT}W^{UK})c_{j}^{KV} \\[4pt]
\mathbf{q}_{t}^T\mathbf{k}_{j} &= c_{t}^{QT}(W^{QK})c_{j}^{KV} \\[4pt]
\end{align*}
$$

Since matrix multiplication is associative, the combined matrix $W^{QK} = W^{UQT}W^{UK}$ can be pre-computed once, so $k_{j}^{C}$ never has to be materialised at inference time.

Similarly, for the attention output:

$$
\begin{align*}
\mathbf{o}_{t,i} &= \sum_{j=1}^{t}\mathrm{Softmax}_j\left(\frac{\mathbf{q}_{t,i}^T\mathbf{k}_{j,i}}{\sqrt{d_h}}\right)\mathbf{v}_{j,i}^C \\[4pt]
\mathbf{u}_t &= W^{O}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};\ldots;\mathbf{o}_{t,n_h}] \\[4pt]
\end{align*}
$$
Therefore, we can simplify $\mathbf{o}_{t,i}$:
$$
\begin{align*}
\mathbf{a}_{t,i,j} &= \mathrm{Softmax}_j\left(\frac{\mathbf{q}_{t,i}^T\mathbf{k}_{j,i}}{\sqrt{d_h}}\right) \\[4pt]
\mathbf{o}_{t,i} &= \sum_{j=1}^{t}\mathbf{a}_{t,i,j}\mathbf{v}_{j,i}^C  \\[4pt]
\mathbf{o}_{t,i} &= \sum_{j=1}^{t}\mathbf{a}_{t,i,j}W_{(i)}^{UV}\mathbf{c}_{j}^{KV}  \\[4pt]
\mathbf{o}_{t,i} &= W_{(i)}^{UV}\sum_{j=1}^{t}\mathbf{a}_{t,i,j}\mathbf{c}_{j}^{KV}  \\[4pt]
\end{align*}
$$

Summing over all heads:

$$
\begin{align*}
\mathbf{u}_t &= W^{O}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};\ldots;\mathbf{o}_{t,n_h}] \\[4pt]
\mathbf{u}_t &= \sum_{i=1}^{n_h}W_{(i)}^{O}\mathbf{o}_{t,i} \\[4pt]
\mathbf{u}_t &= \sum_{i=1}^{n_h}W_{(i)}^{O}W_{(i)}^{UV}\sum_{j=1}^{t}\mathbf{a}_{t,i,j}\mathbf{c}_{j}^{KV} \\[4pt]
\mathbf{u}_t &= \sum_{i=1}^{n_h}W_{(i)}^{O'}\sum_{j=1}^{t}\mathbf{a}_{t,i,j}\mathbf{c}_{j}^{KV} \\[4pt]
\end{align*}
$$

where $W_{(i)}^{O'} = W_{(i)}^{O}W_{(i)}^{UV}$.


## Breakdown of MLA + $W^{O}$ Absorption Trick

We continue from the last section:
$$
\begin{align*}
\mathbf{u}_t &= \sum_{i=1}^{n_h}W_{(i)}^{O'}\sum_{j=1}^{t}\mathbf{a}_{t,i,j}\mathbf{c}_{j}^{KV} \\[4pt]
\end{align*}
$$

Thus, we define the mixed latent:
$$
\begin{align*}
\mathbf{p}_t^{(i)} &= \sum_{j=1}^{t}\mathbf{a}_{t,i,j}\mathbf{c}_{j}^{KV} \\[4pt]
\mathbf{u}_t &= \sum_{i=1}^{n_h}W_{(i)}^{O'}\mathbf{p}_t^{(i)} \\[4pt]
\end{align*}
$$

We wish to apply the same compress/decompress strategy to $W^{O}$.

In other words:

$$
\begin{align*}
\mathbf{u}_t &= W^{UO}\text{LayerNorm}(W^{DO}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};\ldots;\mathbf{o}_{t,n_h}]) \\[4pt]
\mathbf{u}_t &= W^{UO}\text{LayerNorm}\Big(\sum_{i=1}^{n_h}W_{(i)}^{DO}\mathbf{o}_{t,i}\Big) \\[4pt]
\end{align*}
$$

We now expand the per-head term $W_{(i)}^{DO}\mathbf{o}_{t,i}$ inside the LayerNorm sum:

$$
\begin{align*}
W_{(i)}^{DO}\mathbf{o}_{t,i} &= W_{(i)}^{DO}W_{(i)}^{UV}\sum_{j=1}^{t}\mathbf{a}_{t,i,j}\mathbf{c}_{j}^{KV} \\[4pt]
W_{(i)}^{DO}\mathbf{o}_{t,i} &= W_{(i)}^{DO}W_{(i)}^{UV}\mathbf{p}_t^{(i)} \\[4pt]
W_{(i)}^{DO}\mathbf{o}_{t,i} &= W_{(i)}^{DOUV}\mathbf{p}_t^{(i)} \\[4pt]
\end{align*}
$$

where $W_{(i)}^{DOUV} = W_{(i)}^{DO}W_{(i)}^{UV}$, which can be pre-computed. 

Next, we need to sum over the heads to get the LayerNorm input:

$$
\begin{align*}
\mathbf{u}_t &= W^{UO}\text{LayerNorm}\Big(\sum_{i=1}^{n_h}W_{(i)}^{DOUV}\mathbf{p}_t^{(i)}\Big) \\[4pt]
\end{align*}
$$

Note that $W^{UO}$ and the LayerNorm remain outside the per-head sum. Unlike MLA, the path therefore does not collapse to a single fused operator, and the absorption stops at the $d_o$ bottleneck. This is exactly the partial reduction described above.

## Comparing MLA vs MLA + $W^{O}$
### Parameters

Holding the MLA input side fixed, the output projection only changes the number of trainable parameters, with a parameter reduction achieved by MHAE/MLAE relative to MHA/MLA when $d_o < d/2$.

| | output-projection params |
|---|---|
| Dense $W^O$ | $d^2$ |
| Output latent ($W^{DO}$ + $W^{UO}$) | $2 \times d \times d_o$ |

### Decode Compute (FLOPs)

| stage | MLA (absorbed) | MLA + $W^O$ (absorbed) |
|---|---|---|
| attention mix over $t$ keys | $n_h \times t \times d_{kv}$ | $n_h \times t \times d_{kv}$ |
| per-head operator apply | $n_h \times d \times d_{kv}$ | $n_h \times d_{kv} \times d_o$ |
| shared up-projection $W^{UO}$ | — | $d \times d_o$ |
| total | $n_h \times t \times d_{kv} + n_h \times d \times d_{kv}$ | $n_h \times t \times d_{kv} + n_h \times d_{kv} \times d_o + d \times d_o$ |

<br>

- The attention-mix term is identical for both MLA and MLA + $W^O$. They both depend on the KV latent, not the output latent. Therefore, the output latent does not help long-context decoding.
- The output latent only changes the fixed per-token apply term ($n_h \times d \times d_{kv} \to n_h \times d_{kv} \times d_o + d \times d_o$), giving the same reduction $R(d_o)$ (defined below) as the stored parameter count, since a matrix applied once per token costs as many MACs as it has parameters. However, this reduction applies to the apply term only. The total decode FLOPs are diluted by the unchanged mix term $n_h \times t \times d_{kv}$, which dominates at long context, so the total reduction is well below $R(d_o)$ and tends to $1$ as $t$ grows. For a frontier model, $t=4096$, apply-only ratio is $12.6\times$, but total decode ratio is only $\approx 2.4\times$. The win is therefore largest for wide models and/or models with many heads at short to moderate context.

### Absorbed-operator size (value/output path)

The absorption trick lets weight matrices be combined ahead of time, avoiding the need to compute intermediate steps at inference. DeepSeek's MLA applies this to the read side of attention, while this work applies it to the write side. We can estimate the number of parameters held in the pre-computed absorbed operators and compare MLA against the MLA + $W^{O}$ extension.

From above we can see that the MLA absorption trick is defined per head as $W^{O'}_{(i)} = W^O_{(i)}W^{UV}_{(i)}$, with $W^O_{(i)}$ of shape $d \times d_h$ and $W^{UV}_{(i)}$ of shape $d_h \times d_{kv}$. Therefore, across all $n_h$ heads the absorbed matrices occupy a total of $d \times d_{kv} \times n_h$ parameters.

Similarly, for the MLA + $W^{O}$ implementation, we saw that the absorption trick was defined as $W^{UO}\text{LayerNorm}\Big(\sum_{i=1}^{n_h}W_{(i)}^{DOUV}\mathbf{p}_t^{(i)}\Big)$. We know that $W^{UO}$ has shape $d \times d_o$, while we can calculate $W_{(i)}^{DOUV}=W^{DO}_{(i)}W^{UV}_{(i)}$ as having shape $d_o \times d_{kv}$ per attention head. So the absorbed matrices occupy a total of $d_o \times (d + d_{kv} \times n_h)$ parameters (the LayerNorm's $\sim 2d_o$ scale and bias parameters are negligible and omitted).

We can estimate the reduction of the number of parameters that absorbed matrices occupy as:

$$
\begin{align*}
\text{R}(d_o) &= \frac{\text{MLA}}{\text{MLA} + W^{O}} \\[4pt]
\text{R}(d_o) &=\frac{d \times d_{kv} \times n_h}{d \times d_o + d_o \times d_{kv} \times n_h} \\[4pt]
\text{R}(d_o) &=\frac{d \times d_{kv} \times n_h}{d_o \times (d + d_{kv} \times n_h)}
\end{align*}
$$

Note that $R(d_o)$ grows as $d_o$ shrinks, but $d_o$ cannot be reduced freely. The smallest usable $d_o$ has a lower bound defined by the quality of the model. The pretraining and GLUE experiments (see results in README) show how, for the TinyGPT2 model, quality degrades as $d_o$ decreases. Thus, the optimal $d_o$ value is chosen based on the experimental results rather than the value which most reduces $R$.

#### Comparing MLA vs MLA + $W^{O}$ for DeepSeek-V2

Considering the DeepSeek-V2 model, where $d=7168$, $d_{kv}=512$ and $n_h=128$, it is possible to estimate the reduction relative to the size of $d_o$.

$$
\begin{align*}
\text{R}(d_o) &=\frac{7168 \times 512 \times 128}{d_o \times (7168 + 512 \times 128)} \\[4pt]
\text{R}(d_o) &\approx \frac{469.8M}{d_o \times 72704} \\[4pt]
\text{R}(d_o) &\approx \frac{6461.3}{d_o} \\[4pt]
\end{align*}
$$

#### Comparing MLA vs MLA + $W^{O}$ for TinyGPT2

The TinyGPT2 model has the following parameter values:

$$
\begin{align*}
d &= 312 \\[4pt]
d_{kv} &\in \{16,\dots,128\} \\[4pt]
n_h &= 4 \\[4pt]
\end{align*}
$$

Using the above formula, it is possible to estimate the reduction in absorbed stored matrix values, considering the $d_o$ values in the range from 16 to 128 as per the sweep conducted during experimentation.

$$
\begin{align*}
\text{R}(16, 128) &=\Big[\frac{312 \times 16 \times 4}{16 \times (312 + 16 \times 4)}, \frac{312 \times 128 \times 4}{128 \times (312 + 128 \times 4)}\Big] \\[4pt]
\text{R}(16, 128) &\approx \Big[\frac{19968}{16 \times 376}, \frac{159744}{128 \times 824}\Big] \\[4pt]
\text{R}(16, 128) &\approx \Big[\frac{19968}{6016}, \frac{159744}{105472}\Big] \\[4pt]
\text{R}(16, 128) &\approx \Big[3.3191, 1.5146\Big] \\[4pt]
\end{align*}
$$

Therefore, for this model size, across the $d_o$ sweep we would expect the MLA + $W^{O}$ attention architecture to reduce the size of the absorbed operator by a factor of between $1.51$ and $3.32$.

Thus, it is worth noting that the MLA absorption trick trades KV cache for a bigger projection operator. However, the inclusion of the output latent trades some of that back, shrinking the absorbed operator by $1.51-3.32 \times$. Even greater savings could potentially be made on larger frontier models, but this would need to be explored in more detail to confirm.

The table below compares all three variants on the value to output path. MHA is the anchor (dense $W^V+W^O$, no absorption). The KV cache column shows what the read side bottleneck buys and what the output latent leaves untouched.

| variant | absorbed operator (params) | KV cache / token |
|---|---|---|
| MHA | $2d^2$ | $2d$ |
| MLA | $n_h \times d \times d_{kv}$ | $d_{kv}$ |
| MLA + $W^O$ | $n_h \times d_{kv} \times d_o + d \times d_o$ | $d_{kv}$ |

**DeepSeek-V2** ($d=7168,\ n_h=128,\ d_{kv}=512$, aligned $d_o=512$):

| | operator | cache / token |
|---|---|---|
| MHA | 102.8M | 14,336 |
| MLA | 469.8M | 512 |
| MLA + $W^O$ | 37.2M | 512 |

**TinyGPT2** ($d=312,\ n_h=4,\ d_{kv}=d_o=128$):

| | operator | cache / token |
|---|---|---|
| MHA | 194,688 | 624 |
| MLA | 159,744 | 128 |
| MLA + $W^O$ | 105,472 | 128 |

We can see that MLA shrinks the cache (~28× at frontier) but inflates the operator above MHA (469.8M vs 102.8M). The output latent pulls the operator back below MHA (37.2 M) while retaining the benefits of a small cache. Note that this inflation is a frontier effect (it needs $n_h\,d_{kv} > 2d$). At TinyGPT2 scale MLA's operator is already below MHA's, so there is no inflation to claw back there.

## Limitations
1. It is worth noting that the addition of the output latent has no bearing on the KV cache and provides no additional savings.
2. DeepSeek's original result was that the full $k$ and $v$ vectors are never materialised and the inclusion of the output latent only builds on their result.

## Next steps
The above analysis has shown that the fused operator is smaller in both parameters and per token FLOPs. However, we have yet to measure the wall clock latency gain. Implementing the absorbed inference for MLA, MHAE and MLAE and measuring it is left as future work.
