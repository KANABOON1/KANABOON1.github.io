---
title: 'Blogs of Thinking Machines Lab'
date: 2026-05-14
permalink: /posts/thinking-machines-lab/
read_time: 20
author: Muxin Fu
tags:
  - Thinking Machines Lab
description: 'Thinking Machines Lab blogs 的整理。'
---

<style>
.thinking-machine-post {
  --tmp-ink-soft: #5d6865;
  --tmp-border: rgba(0, 0, 0, 0.12);
  --tmp-shadow: 0 18px 46px rgba(22, 42, 38, 0.12);
}
.thinking-machine-post p,
.thinking-machine-post li {
  line-height: 1.82;
}
.thinking-machine-post h1 {
  margin-top: 2.4em;
  padding-bottom: 0.28em;
  border-bottom: 1px solid var(--tmp-border);
}
.thinking-machine-post h2 {
  margin-top: 2.0em;
}
.thinking-machine-post h3 {
  margin-top: 1.6em;
}
.thinking-machine-post .post-figure {
  margin: 2.1rem auto;
  text-align: center;
}
.thinking-machine-post .post-figure img {
  display: block;
  width: 80%;
  max-width: 860px;
  height: auto;
  margin: 0 auto;
  border: 1px solid var(--tmp-border);
  border-radius: 16px;
  background: #fff;
  box-shadow: var(--tmp-shadow);
}
.thinking-machine-post .post-figure--wide img {
  max-width: 980px;
}
.thinking-machine-post figcaption {
  max-width: 780px;
  margin: 0.85rem auto 0;
  font-size: 0.92em;
  line-height: 1.65;
}
.thinking-machine-post .references p {
  margin: 0.75rem 0;
  padding-left: 2.2rem;
  text-indent: -2.2rem;
  line-height: 1.68;
}
@media (max-width: 720px) {
  .thinking-machine-post .post-figure {
    margin: 1.6rem -0.25rem;
  }
  .thinking-machine-post .post-figure img {
    border-radius: 12px;
  }
}
</style>

* TOC
{:toc}

<div class="thinking-machine-post" markdown="1">

***Special thanks to [Thinking Machines Lab](https://thinkingmachines.ai) for their insightful work.***

# Interaction Models: A Scalable Approach to Human-AI Collaboration [1]

## Motivation
AI labs 经常将模型的自主运行能力视为最重要的能力。然而在很多场景下, agent 仍然需要与用户协作, 而当前机制往往限制了 agent 的主动交互性:
- 当前的 agent 系统几乎都是 turn-based。agent 在本轮任务完成前, 感知通道处于关闭状态; 直到本轮生成完成后, 感知才会重新开启。这一性质严重制约了模型的主动实时交互能力。
- 现有做法往往通过 harness 来增强系统与用户的交互性。然而, 这种方式难以 scale。*The Bitter Lesson* 告诉我们: 手工打造的系统往往会被基于计算的通用系统超越。

综上, 将主动交互能力内化到模型中是十分自然的:
> We think interactivity should scale alongside intelligence; the way we work with AI should not be treated as an afterthought.

## Methodology

### *System Overview*
整个系统分为两个部分: interaction model 和 background model [1]。interaction model 始终保持与用户的交互; 当问题超出简单推理的范围时, interaction model 会将任务交给后台异步运行的 background model (与 planner-executor 机制相同), 同时继续与用户交互, 并能够将 background model 返回的结果融入后续对话中。
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/thinking-machines-lab/system.png" alt="System overview">
  <figcaption>Figure 1: System overview of the interaction model and the background model. (Image source: Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration")</figcaption>
</figure>

### *The interaction model*

***Time-aligned micro-turns.*** 模型具备交互性的核心特征在于能够: perceiving and responding at the same time. 基于这个 insight, interaction model 的核心机制是 **Time-aligned micro-turns**。也就是说, 系统每 200ms 都会将这一阶段的用户内容输入给模型, 使模型的输入与输出交错在同一个序列中。这样一来, 在用户说话时, 模型也能持续感知, 并生成回应、插话、沉默等行为。本质上这仍然是 turn, 只是这个 turn 足够小, 能够满足模型与用户实时交互的需求。
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/thinking-machines-lab/micro_time.png" alt="Time-aligned micro-turns">
  <figcaption>Figure 2: Time-aligned micro-turns interleave user input and model output in a shared temporal sequence. (Image source: Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration")</figcaption>
</figure>

***Encoder-free early fusion.*** 相比于传统模型的目标, 即 "理解内容是什么", interaction model 的目标是 "在连续互动中, 判断现在应该怎么协调行动", 两者之间存在显著差异。预训练好的 encoder 更注重提取关键信息, 往往会忽略对连续互动有帮助的细节 (例如: 时间、犹豫等)。因此, TML 采用的策略是让模型一开始就从统一序列中联合建模所有模态与时间。
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/thinking-machines-lab/fusion.png" alt="Encoder-free early fusion">
  <figcaption>Figure 3: Encoder-free early fusion jointly models modalities and timing from the beginning of the sequence. (Image source: Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration")</figcaption>
</figure>

## Benchmarks
TML 分别使用 *FD-bench* 衡量模型的 *interaction quality*, 使用 *Audio MultiChallenge* 衡量模型的 *intelligence*。结果如下图所示, 可以看到 TML-small 模型在交互质量上远超其他模型, 但在智力上略低于 GPT realtime-2.0:
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/thinking-machines-lab/exist_bench.png" alt="FD-bench and Audio MultiChallenge benchmark results">
  <figcaption>Figure 4: Comparison on FD-bench for interaction quality and Audio MultiChallenge for intelligence. (Image source: Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration")</figcaption>
</figure>

为了进一步衡量模型的交互能力, TML 从两个角度对模型进行测试:
- Time awareness and simultaneous speech: 模型是否具备精准的时间估计能力? 模型是否能够在合适的时间主动说话? TML 分别使用内部构造的 *TimeSpeak* 和 *CueSpeak* 这两个 bench 衡量这两种主动式音频能力。
- Visual proactivity: 模型是否能够根据动态变化的视觉信息, 在合适的时机主动发起对话? TML 使用三个已有的 benchmarks (*RepCount-A*, *ProactiveVideoQA*, *Charades*) 衡量模型的视觉主动能力。

可以看到, 在 interaction 能力上, TML-interaction-small 模型远超 GPT realtime-2.0。
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/thinking-machines-lab/new_bench.png" alt="Interaction benchmark results">
  <figcaption>Figure 5: Evaluation of time awareness, simultaneous speech, and visual proactivity. (Image source: Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration")</figcaption>
</figure>


## Future Work
**Long sessions.** 如果遇到超长视频或者音频, 肯定需要对模型的上下文进行管理, 而目前 TML 并没有给出明确做法。由于 TML 显式考虑了时间维度, 因此如何在上下文管理的同时精确保留时间属性, 可能是一个难点。

**Improved background agents.** TML 采用 *interaction model* 和 *background model* 进行协作, 并且也同意 "we have just scratched the surface in how the background agents can work together with the interaction model"。事实上, 这种做法与 multi-agent system 中的 planner-executor 模式很像, 只是 TML 采用的是最简单的结构。因此, 探索"后台 agent 如何与交互模型更紧密协作"是一个值得考虑的方向。

# References

<div class="references" markdown="1">

[1] Thinking Machines Lab, ["Interaction Models: A Scalable Approach to Human-AI Collaboration"](https://thinkingmachines.ai/blog/interaction-models/), Thinking Machines Lab: Connectionism, May 2026.

</div>

</div>
