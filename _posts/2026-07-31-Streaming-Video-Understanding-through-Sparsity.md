---
title: 'Streaming Video Understanding through Sparsity'
date: 2026-07-31
permalink: /posts/streaming-video-understanding-through-sparsity/
read_time: 20
author: Muxin Fu
tags:
  - Streaming Video
  - Visual Sparsity
  - Multimodal
description: '流视频理解需要在实时交互的同时, 高效处理持续累积的冗余视觉信息。本文介绍 visual token compression 与 sparse vision encoder 两条技术路线及其代表性工作。'
---

<style>
.streaming-sparsity-post {
  --ssp-ink-soft: #5d6865;
  --ssp-border: rgba(0, 0, 0, 0.12);
  --ssp-shadow: 0 18px 46px rgba(22, 42, 38, 0.12);
}
.streaming-sparsity-post p,
.streaming-sparsity-post li {
  line-height: 1.82;
}
.streaming-sparsity-post h1 {
  margin-top: 2.4em;
  padding-bottom: 0.28em;
  border-bottom: 1px solid var(--ssp-border);
}
.streaming-sparsity-post h2 {
  margin-top: 2.0em;
}
.streaming-sparsity-post h3 {
  margin-top: 1.6em;
}
.streaming-sparsity-post .post-figure {
  margin: 2.1rem auto;
  text-align: center;
}
.streaming-sparsity-post .post-figure img {
  display: block;
  width: 80%;
  max-width: 860px;
  height: auto;
  margin: 0 auto;
  border: 1px solid var(--ssp-border);
  border-radius: 16px;
  background: #fff;
  box-shadow: var(--ssp-shadow);
}
.streaming-sparsity-post .post-figure--wide img {
  max-width: 980px;
}
.streaming-sparsity-post figcaption {
  max-width: 780px;
  margin: 0.85rem auto 0;
  font-size: 0.92em;
  line-height: 1.65;
}
.streaming-sparsity-post .references p {
  margin: 0.75rem 0;
  padding-left: 2.2rem;
  text-indent: -2.2rem;
  line-height: 1.68;
}
@media (max-width: 720px) {
  .streaming-sparsity-post .post-figure {
    margin: 1.6rem -0.25rem;
  }
  .streaming-sparsity-post .post-figure img {
    border-radius: 12px;
  }
}
</style>

* TOC
{: toc}

<div class="streaming-sparsity-post" markdown="1">

当前, 多模态大语言模型 (MLLM) 在应用于流视频理解时, 通常面临两个核心问题: (I) 如何主动响应实时交互需求; (II) 如何高效处理视频流中大量冗余的视觉信息。针对第一个问题, 现有方法通常构造包含主动交互行为的训练数据, 使模型能够在视频播放过程中判断何时需要主动发起对话或输出回答。针对第二个问题, 许多方法仍主要依赖关键帧采样, 但这种粗粒度策略容易遗漏持续时间较长的宏观事件以及短暂出现的局部细节。因此, 本文主要围绕第二个问题, 介绍流视频理解中的视觉信息压缩方向及其代表性工作。

# Visual Token Compression

使用 vision encoder 对视频帧进行编码后, 模型会得到大量 visual patch tokens。由于相邻视频帧之间通常存在显著的时间冗余, 同一帧内部也包含大量空间冗余, 因此可以在尽量保留重要视觉信息的前提下, 对冗余 token 进行筛除或压缩。从压缩发生的位置来看, 现有方法大致分为两类: (I) KV-cache compression; (II) visual embedding compression。两类方法的本质都是估计视觉 token 的重要性, 并据此减少流视频不断累积的视觉上下文。区别在于, 前者在视觉 token 已经进入 LLM 并被转化为逐层 KV-cache 后进行压缩, 而后者直接在视觉 embedding 进入 LLM 之前去除冗余。

## KV-Cache Compression

在 KV-cache 层进行压缩, 可以直接复用 LLM 已经计算得到的 key 和 value, 从而较为方便地评估不同视觉 token 与当前文本上下文之间的相关性。此外, 由于不同 Transformer 层维护各自独立的 KV-cache, 这类方法可以让每一层根据自身的表示特征, 独立选择最适合保留或检索的视觉信息。

*ReKV* [(Di et al., 2025)](https://openreview.net/forum?id=8g9fs6mdEG) 并不直接进行压缩, 而是进行检索。它首先完整保留当前视频中所有 visual tokens 对应的 KV-cache, 这一过程被称为 internal retrieval。当模型开始回答问题时, 每一层根据当前层的 hidden states $X$, 从该层保存的 visual KV-cache 中检索 Top-$K$ 个最相关的视觉 token, 并将其作为额外上下文参与注意力计算:

\[
O = \operatorname{Attn}\left(
W_QX, \,
[L_K, W_KX], \,
[L_V, W_VX]
\right).
\]

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/streaming-video-understanding-through-sparsity/rekv.png" alt="ReKV framework for in-context video KV-cache retrieval">
  <figcaption>Figure 1: ReKV retrieves query-relevant visual KV-cache for streaming video question answering. (Image source: Di et al. 2025)</figcaption>
</figure>

除了将 KV-cache 作为可检索的外部记忆, 也可以直接对其进行持续压缩。*LiveVLM* [(Ning et al., 2025)](https://arxiv.org/abs/2505.15269) 观察到 visual tokens 中存在类似 attention sink 的现象, 即大量视觉 token 会将较高的注意力分配给少数 sink visual tokens。因此, *LiveVLM* 使用 vision-to-vision attention score, 而不是 text-to-vision attention score, 计算每层 visual token 的重要性, 并在时间分桶的约束下选择需要保留的 token。该策略既维持了视频内容在时间维度上的覆盖, 又能够优先保留注意力得分较高的视觉信息。*InfiniPot-V* [(Kim et al., 2025)](https://arxiv.org/abs/2506.15745) 提出了一种受预算约束的持续 KV-cache 压缩机制。当缓存长度达到预设上限 $M$ 时, 每一层的 KV-cache 会被压缩至长度 $C$, 其中 $C \ll M$。具体而言, *InfiniPot-V* 首先沿时间维度执行稀疏化操作 TaR: 如果当前帧某一位置的 key 与历史帧相同位置的 key 具有较高相似度, 则认为该 token 包含重复信息并将其移除。随后, 方法沿空间维度执行 VaR: 优先保留 value norm 较大的 token。作者发现, value norm 较大的 token 通常具有更高的表示熵, 因此可能包含更丰富的信息。*HERMES* [(Zhang et al., 2026)](https://arxiv.org/abs/2601.14724) 则进一步研究了不同深度的 LLM 层对视觉信息的注意力模式。其分析表明, 浅层具有明显的近因偏好, 即模型更倾向于关注距离当前 token 较近的视觉 token; 随着网络深度增加, 这种近因偏好逐渐减弱。因此, *HERMES* 针对 shallow, middle 和 deep layers 分别设计了不同的 visual KV-cache 重要性评估规则, 并将 KV-cache 组织为 hierarchical memory, 以匹配不同层的注意力特性。此外, *HERMES* 提出了 M-RoPE 的位置重索引策略, 使压缩后保留下来的 token 在 M-RoPE 的三个坐标维度上重新保持连续。需要注意的是, 对于压缩后 token 的位置编码是否应当重索引, 目前不同工作采用了不同策略: 部分方法保留 token 的原始位置坐标, 另一些方法则重新构造连续的位置索引。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/streaming-video-understanding-through-sparsity/infinipot.png" alt="InfiniPot-V memory-constrained KV-cache compression framework">
  <figcaption>Figure 2: InfiniPot-V removes temporal redundancy and preserves informative values under a fixed memory budget. (Image source: Kim et al. 2025)</figcaption>
</figure>

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/streaming-video-understanding-through-sparsity/hermes.png" alt="HERMES hierarchical KV-cache memory framework">
  <figcaption>Figure 3: HERMES organizes visual KV-cache as hierarchical memory for layers with different attention patterns. (Image source: Zhang et al. 2026)</figcaption>
</figure>

## Visual Embedding Compression

Visual embedding compression 通常从时间和空间两个维度去除冗余的 visual patch tokens。相比于在 KV-cache 层压缩已经经过多层计算和语义混合的内部状态, visual embedding compression 能够在视觉信息进入 LLM 之前, 直接利用视频本身的时空结构过滤重复内容, 从源头减少 LLM prefill, 后续各层计算以及 KV-cache 增长所带来的计算与显存开销。

*TimeChat-Online* [(Yao et al., 2025)](https://arxiv.org/abs/2504.17343) 根据相邻两帧相同空间位置上的 visual embeddings 之间的余弦相似度, 判断对应 token 是否冗余。当相似度超过预设阈值时, 当前帧中的对应 token 会被删除。该工作也探索了基于 pixel-level difference 的冗余判断方式, 但由于像素差异容易受到光照变化, 编码噪声等因素影响, 其效果不如 feature-level similarity。对于保留下来的 token, *TimeChat-Online* 继续使用其原始 M-RoPE 坐标, 以避免删除 token 后重新排列位置所导致的空间错位。此外, *TimeChat-Online* 还利用被删除 token 的数量估计当前场景的变化程度, 并将该指标作为第一层门控信号, 用于判断是否需要将当前场景交给 backbone LLM, 从而触发模型的主动输出行为。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/streaming-video-understanding-through-sparsity/timechat-online.png" alt="TimeChat-Online differential token dropping framework">
  <figcaption>Figure 4: TimeChat-Online removes temporally redundant visual tokens through differential token dropping. (Image source: Yao et al. 2025)</figcaption>
</figure>

*FluxMem* [(Xie et al., 2026)](https://arxiv.org/abs/2603.02096) 进一步将 visual memory 划分为 short-term memory, mid-term memory 和 long-term memory, 并针对不同时间尺度的记忆采用不同的压缩策略。Short-term memory 不进行压缩, 以完整保留最近的视觉信息。Mid-term memory 采用与 *TimeChat-Online* 类似的时间冗余判断方式, 通过比较相邻帧对应位置上的 visual embeddings 来筛除重复 token。相比 *TimeChat-Online*, *FluxMem* 主要进行了两点改进: (1) 不再仅比较完全相同位置上的 token, 而是在 $3 \times 3$ 的局部窗口中计算相似度; (2) 使用 Otsu 自适应阈值法 (see Appendix for more details) 确定冗余判断阈值, 而不是采用人工设定的固定阈值。Long-term memory 则进一步处理单帧内部的空间冗余。*FluxMem* 计算同一帧内不同 visual embeddings 之间的相似度, 并将相似度高于阈值的 token 连接起来。在完成整帧 token 的相似关系构建后, 每个连通分量中的 visual patch embeddings 会被聚合为其均值, 并以该均值作为新的 visual embedding。通过这种方式, 多个语义相近或空间冗余的 patch tokens 可以被合并为单个 token, 从而显著减少长期记忆中的 token 数量。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/streaming-video-understanding-through-sparsity/fluxmem.png" alt="FluxMem adaptive hierarchical visual memory framework">
  <figcaption>Figure 5: FluxMem compresses streaming visual memory across short-, mid-, and long-term timescales. (Image source: Xie et al. 2026)</figcaption>
</figure>

*StreamingTOM* [(Chen et al., 2025)](https://arxiv.org/abs/2510.18269) 将 visual token compression 与 KV-cache retrieval 结合起来。首先, 方法在 visual embedding 层去除视频中的冗余视觉信息; 随后, 将压缩后的 visual embeddings 输入 LLM, 通过 prefill 转化为 KV-cache, 并以量化形式存储。当模型需要回答问题时, 再根据当前查询动态检索 Top-$K$ 个最相关的 frame-level KV-cache, 并将其作为视觉上下文提供给模型。因此, *StreamingTOM* 实际上结合了两类方法的优势: visual embedding compression 用于减少进入 LLM 的冗余 token, 而 KV-cache retrieval 则用于在回答阶段动态访问与当前问题相关的历史视觉信息。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/streaming-video-understanding-through-sparsity/streamingtom.png" alt="StreamingTOM token compression and KV-cache retrieval framework">
  <figcaption>Figure 6: StreamingTOM combines pre-LLM visual token compression with query-dependent KV-cache retrieval. (Image source: Chen et al. 2025)</figcaption>
</figure>

# Sparse Vision Encoder

Deep learning scales best when its architecture aligns with the fundamental structure of the data。除了在 vision encoder 输出 visual patch tokens 后进行压缩, 还可以直接调整 encoder 的架构, 使其利用视频数据本身的稀疏性, 从编码阶段便生成更少的 visual tokens。

*OneVision-Encoder* [(Tang et al., 2026)](https://arxiv.org/abs/2602.08683) 将 video codec 的编码结构引入稀疏视觉建模。它首先使用 HEVC (see Appendix for more details) 编码视频, 完整保留每个 I-frame 的 RGB 数据, 同时提取每个 P-frame 的 motion vector $\tau_t$ 和 residual signal $R_t$。对于 P-frame 中的每个 patch, *OneVision-Encoder* 将对应的 motion magnitude 与 residual energy 相加, 得到一个标量显著性分数。随后, 方法根据该分数对所有 patch 排序, 仅保留 Top-$K$ 个最显著的 patch 并送入 vision encoder 中, 从而避免对变化较小的区域进行冗余编码。为了预训练该 encoder, *OneVision-Encoder* 还提出 image and video clustering。该方法分别对训练集中的 image 和 video 表征进行聚类, 并在训练过程中拉近样本表征与所属聚类中心的距离, 同时推远其与其他聚类中心的距离。通过这种多类别对比学习目标, vision encoder 能够学习适合图像和视频输入的视觉表征。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/streaming-video-understanding-through-sparsity/one-vision-encoder.png" alt="OneVision-Encoder codec-aligned sparse vision encoder">
  <figcaption>Figure 7: OneVision-Encoder uses codec-derived motion and residual signals to select informative video patches. (Image source: Tang et al. 2026)</figcaption>
</figure>

*CoPE-VideoLM* [(Sarkar et al., 2026)](https://arxiv.org/abs/2602.13191) 同样借鉴了 HEVC, 但不再要求 vision encoder 为 P-frame 生成稠密的 patch embeddings。它引入额外的 $\delta$-encoder, 将 P-frame 的 motion vectors 和 residual signals 分别编码为固定数量的 motion tokens 与 residual tokens, 从而直接以紧凑表征描述帧间变化。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/streaming-video-understanding-through-sparsity/cope-videolm.png" alt="CoPE-VideoLM codec-primitive video language model">
  <figcaption>Figure 8: CoPE-VideoLM represents inter-frame changes with compact motion and residual tokens. (Image source: Sarkar et al. 2026)</figcaption>
</figure>

为了使 $\delta$-encoder 生成的 tokens 与标准 vision encoder 的 visual embedding space 对齐, *CoPE-VideoLM* 设计了一个 self-reconstruction simulation 框架。该框架在预训练阶段引入 reference Transformer 和 warped Transformer, 并在特征空间中模拟 HEVC 的帧间解码过程。首先, reference Transformer 根据参考 I-frame $I_{t-s}$ 和 motion tokens $\tau_{\mathrm{tok},t}$ 生成运动补偿后的参考特征 $\tilde{P}_{\mathrm{ref},t} = \theta_{\mathrm{ref}}(I_{t-s}, \tau_{\mathrm{tok},t})$。随后, warped Transformer 融合 residual tokens $R_{\mathrm{tok},t}$, 重建当前 P-frame 的视觉特征 $\tilde{P}_t = \theta_{\mathrm{warped}}(\tilde{P}_{\mathrm{ref},t}, R_{\mathrm{tok},t})$。最后, 方法计算重建特征 $\tilde{P}_t$ 与标准 vision encoder 输出 $\hat{P}_t$ 之间的 reconstruction loss。由于两个 Transformer 只能依赖 $\delta$-encoder 提供的压缩信息完成重建, 该目标会迫使 $\delta$-encoder 学会在有限 token 预算下准确表征 motion vectors 和 residual signals。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/streaming-video-understanding-through-sparsity/delta-encoder.png" alt="CoPE-VideoLM delta-encoder self-reconstruction simulation">
  <figcaption>Figure 9: Self-reconstruction simulation aligns compact codec tokens with the visual embedding space. (Image source: Sarkar et al. 2026)</figcaption>
</figure>

# References

<div class="references" markdown="1">

[1] Linli Yao, et al. ["TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos."](https://arxiv.org/abs/2504.17343) Proceedings of the 33rd ACM International Conference on Multimedia (2025).

[2] Shangzhe Di, et al. ["Streaming Video Question-Answering with In-Context Video KV-Cache Retrieval."](https://openreview.net/forum?id=8g9fs6mdEG) ICLR 2025.

[3] Zhenyu Ning, et al. ["LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval."](https://arxiv.org/abs/2505.15269) arXiv preprint arXiv:2505.15269 (2025).

[4] Minsoo Kim, et al. ["InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding."](https://arxiv.org/abs/2506.15745) Advances in Neural Information Processing Systems 38 (2025).

[5] Haowei Zhang, et al. ["HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding."](https://arxiv.org/abs/2601.14724) arXiv preprint arXiv:2601.14724 (2026).

[6] Yiweng Xie, et al. ["FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding."](https://arxiv.org/abs/2603.02096) arXiv preprint arXiv:2603.02096 (2026).

[7] Xueyi Chen, et al. ["StreamingTOM: Streaming Token Compression for Efficient Video Understanding."](https://arxiv.org/abs/2510.18269) arXiv preprint arXiv:2510.18269 (2025).

[8] Feilong Tang, et al. ["OneVision-Encoder: Codec-Aligned Sparsity as a Foundational Principle for Multimodal Intelligence."](https://arxiv.org/abs/2602.08683) arXiv preprint arXiv:2602.08683 (2026).

[9] Sayan Deb Sarkar, et al. ["CoPE-VideoLM: Leveraging Codec Primitives for Efficient Video Language Modeling."](https://arxiv.org/abs/2602.13191) arXiv preprint arXiv:2602.13191 (2026).

</div>

</div>

# Appendix

## Otsu 自适应阈值法

## HEVC
