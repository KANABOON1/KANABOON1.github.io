---
title: 'Frontier AI Articles Reading Notes'
date: 2026-05-14
permalink: /posts/frontier-ai-articles-reading/
read_time: 40
author: Muxin Fu
tags:
  - AI Research
  - Reading Notes
  - Frontier Articles
description: 'AI 前沿文章阅读笔记，记录 papers、research blogs、technical reports 和 benchmark reports 的阅读理解。'
---

<style>
.frontier-reading-post {
  --frp-ink-soft: #5d6865;
  --frp-border: rgba(0, 0, 0, 0.12);
  --frp-shadow: 0 18px 46px rgba(22, 42, 38, 0.12);
}
.frontier-reading-post p,
.frontier-reading-post li {
  line-height: 1.82;
}
.frontier-reading-post h1 {
  margin-top: 2.4em;
  padding-bottom: 0.28em;
  border-bottom: 1px solid var(--frp-border);
}
.frontier-reading-post h2 {
  margin-top: 2.0em;
}
.frontier-reading-post h3 {
  margin-top: 1.6em;
}
.frontier-reading-post .post-figure {
  margin: 2.1rem auto;
  text-align: center;
}
.frontier-reading-post .post-figure img {
  display: block;
  width: 80%;
  max-width: 860px;
  height: auto;
  margin: 0 auto;
  border: 1px solid var(--frp-border);
  border-radius: 16px;
  background: #fff;
  box-shadow: var(--frp-shadow);
}
.frontier-reading-post .post-figure--wide img {
  max-width: 980px;
}
.frontier-reading-post figcaption {
  max-width: 780px;
  margin: 0.85rem auto 0;
  font-size: 0.92em;
  line-height: 1.65;
}
.frontier-reading-post .references p {
  margin: 0.75rem 0;
  padding-left: 2.2rem;
  text-indent: -2.2rem;
  line-height: 1.68;
}
@media (max-width: 720px) {
  .frontier-reading-post .post-figure {
    margin: 1.6rem -0.25rem;
  }
  .frontier-reading-post .post-figure img {
    border-radius: 12px;
  }
}
</style>

* TOC
{:toc}

<div class="frontier-reading-post" markdown="1">

这篇 blog 用来记录我对 AI 前沿文章的阅读笔记。内容不局限于 paper，也包括 research blog、technical report、benchmark report、开源项目。

# Reading Index

| No. | Article | Source | Topic | Why it matters |
|---|---|---|---|---|---|
| 001 | Interaction Models: A Scalable Approach to Human-AI Collaboration | Thinking Machines Lab | Human-AI Interaction | Realtime interaction may become an important model capability axis. |
| 002 | On-Policy Distillation | Thinking Machines Lab | Post-training | On-policy distillation combines student-sampled trajectories with dense teacher supervision. |

# [Reading Note 001] Interaction Models: A Scalable Approach to Human-AI Collaboration [1]

***Special thanks to [Thinking Machines Lab](https://thinkingmachines.ai) for their insightful work.***

## Introduction
AI labs 经常将模型的自主运行能力视为最重要的能力。然而在很多场景下, agent 仍然需要与用户协作, 而当前机制往往限制了 agent 的主动交互性:
- 当前的 agent 系统几乎都是 turn-based。agent 在本轮任务完成前, 感知通道处于关闭状态; 直到本轮生成完成后, 感知才会重新开启。这一性质严重制约了模型的主动实时交互能力。
- 现有做法往往通过 harness 来增强系统与用户的交互性。然而, 这种方式难以 scale。*The Bitter Lesson* 告诉我们: 手工打造的系统往往会被基于计算的通用系统超越。

综上, 将主动交互能力内化到模型中是十分自然的:
> We think interactivity should scale alongside intelligence; the way we work with AI should not be treated as an afterthought.

## Methodology

### *System Overview*
整个系统分为两个部分: interaction model 和 background model [1]。interaction model 始终保持与用户的交互; 当问题超出简单推理的范围时, interaction model 会将任务交给后台异步运行的 background model (与 planner-executor 机制相同), 同时继续与用户交互, 并能够将 background model 返回的结果融入后续对话中。
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/frontier-ai-articles-reading/1_system.png" alt="System overview">
  <figcaption>Figure 1: System overview of the interaction model and the background model.<br>(Image source: Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration")</figcaption>
</figure>

### *The interaction model*

***Time-aligned micro-turns.*** 模型具备交互性的核心特征在于能够: perceiving and responding at the same time. 基于这个 insight, interaction model 的核心机制是 **Time-aligned micro-turns**。也就是说, 系统每 200ms 都会将这一阶段的用户内容输入给模型, 使模型的输入与输出交错在同一个序列中。这样一来, 在用户说话时, 模型也能持续感知, 并生成回应、插话、沉默等行为。本质上这仍然是 turn, 只是这个 turn 足够小, 能够满足模型与用户实时交互的需求。
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/frontier-ai-articles-reading/1_micro_time.png" alt="Time-aligned micro-turns">
  <figcaption>Figure 2: Time-aligned micro-turns interleave user input and model output in a shared temporal sequence.<br>(Image source: Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration")</figcaption>
</figure>

***Encoder-free early fusion.*** 相比于传统模型的目标, 即 "理解内容是什么", interaction model 的目标是 "在连续互动中, 判断现在应该怎么协调行动", 两者之间存在显著差异。预训练好的 encoder 更注重提取关键信息, 往往会忽略对连续互动有帮助的细节 (例如: 时间、犹豫等)。因此, TML 采用的策略是让模型一开始就从统一序列中联合建模所有模态与时间。
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/frontier-ai-articles-reading/1_fusion.png" alt="Encoder-free early fusion">
  <figcaption>Figure 3: Encoder-free early fusion jointly models modalities and timing from the beginning of the sequence.<br>(Image source: Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration")</figcaption>
</figure>

## Experiments
TML 分别使用 *FD-bench* 衡量模型的 *interaction quality*, 使用 *Audio MultiChallenge* 衡量模型的 *intelligence*。结果如下图所示, 可以看到 TML-small 模型在交互质量上远超其他模型, 但在智力上略低于 GPT realtime-2.0:
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/frontier-ai-articles-reading/1_exist_bench.png" alt="FD-bench and Audio MultiChallenge benchmark results">
  <figcaption>Figure 4: Comparison on FD-bench for interaction quality and Audio MultiChallenge for intelligence.<br>(Image source: Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration")</figcaption>
</figure>

为了进一步衡量模型的交互能力, TML 从两个角度对模型进行测试:
- Time awareness and simultaneous speech: 模型是否具备精准的时间估计能力? 模型是否能够在合适的时间主动说话? TML 分别使用内部构造的 *TimeSpeak* 和 *CueSpeak* 这两个 bench 衡量这两种主动式音频能力。
- Visual proactivity: 模型是否能够根据动态变化的视觉信息, 在合适的时机主动发起对话? TML 使用三个已有的 benchmarks (*RepCount-A*, *ProactiveVideoQA*, *Charades*) 衡量模型的视觉主动能力。

可以看到, 在 interaction 能力上, TML-interaction-small 模型远超 GPT realtime-2.0。
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/frontier-ai-articles-reading/1_new_bench.png" alt="Interaction benchmark results">
  <figcaption>Figure 5: Evaluation of time awareness, simultaneous speech, and visual proactivity.<br>(Image source: Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration")</figcaption>
</figure>


## Future Work
**Long sessions.** 如果遇到超长视频或者音频, 肯定需要对模型的上下文进行管理, 而目前 TML 并没有给出明确做法。由于 TML 显式考虑了时间维度, 因此如何在上下文管理的同时精确保留时间属性, 可能是一个难点。

**Improved background agents.** TML 采用 *interaction model* 和 *background model* 进行协作, 并且也同意 "we have just scratched the surface in how the background agents can work together with the interaction model"。事实上, 这种做法与 multi-agent system 中的 planner-executor 模式很像, 只是 TML 采用的是最简单的结构。因此, 探索"后台 agent 如何与交互模型更紧密协作"是一个值得考虑的方向。

# [Reading Note 002] On-Policy Distillation [2]

***Special thanks to [Thinking Machines Lab](https://thinkingmachines.ai) for their insightful work.***

## Introduction
LLMs 目前已经可以在垂直领域取得专家级的表现, 这样的结果通常是这四种能力的叠加: *perception of input*, *knowledge retrieval*, *plan selection* and *reliable execution*. 但是考虑到通用大模型庞大的计算开销, 使用更加轻量同时更加定制化的小模型往往更有优势. 因此, 为了在特定专家领域上让小模型具备甚至超过通用大模型的性能, 后训练是十分重要的一步. 当前对于 student model 的后训练大体可以分为两大范式:
- Off-policy training: 依赖外部构建的有标签的专家数据集让 student model 模仿;
- On-policy training: 从 student model 自己采样出数据, 然后给出相应的奖励.

然而, 这两种范式各自具有优缺点. 

off-policy training 通常以 SFT 的方式实现. 由于 off-policy training 是逐 token 进行监督的, 监督信号更加密集, 因此这种方式的优点是可以高效地让模型进行指令遵循, 数学推理等特定任务. 然而, off-policy training 的主要问题是 *Exposure Bias*, 即如果模型在早期就犯了一个教师模型不会犯的错误, 那么由于和训练时的上下文不同, 模型的错误会随着推理继续逐步累积, 进而导致最终错误. 产生这个问题的根本原因是在 off-policy training 时, 数据直接来源自教师模型 (外部收集也可以看成是教师模型的一种), 并非从学生模型采样得到, 因此在推理时很难有和教师模型一样的语境.

on-policy training via reinforcement learning. on-policy RL 的方法的优点是样本直接从学生模型采样, 这样学生模型以更加直接的方式学会改正自己的错误. 然而, RL 方法有很明显的问题在于其监督信号十分稀疏. 当学生模型错误地回答了问题时, 他只能被告知这道题他答错了, 但是并不知道错在哪里. 因此这种奖励的稀疏性限制了 RL 的应用场景.

使用国际象棋这个场景来打比方, off-policy training 就像让学生观看大师下象棋, 虽然很精妙但是学生在实践中往往不会遇到这种棋局; on-policy training 就像让学生自己下象棋, 虽然胜负仅直接来自于学生自己的下法, 但是教练仅仅会给出胜负结果, 并不会告诉学生哪一步做错了.

因此, 我们希望有一个方法, 能够兼具 off-policy training 的密集监督的特点和 on-policy training 的从模型采样的特性, 这个方法即 *On-Policy Distillation*.

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/frontier-ai-articles-reading/2_compare.png" alt="Comparison between SFT, RL, and on-policy distillation">
  <figcaption>Figure 6: Comparison between off-policy distillation, reinforcement learning, and on-policy distillation.<br>(Image source: Thinking Machines Lab, "On-Policy Distillation")</figcaption>
</figure>

## Methodology

### *Loss function: reverse KL*
On-policy distillation 的核心想法是: 从 student model 采样出轨迹, 并让 teacher model 对每一个 token 进行打分。具体来说, on-policy distillation 采用反向 KL 散度 (反向 KL 与 on-policy 这种从自身采样的方法更契合), 让学生模型在自己实际生成的轨迹上靠近 teacher model 的高概率行为。需要注意的是, reverse KL 是 mode-seeking 的, 因此它更倾向于压缩到 teacher model 认为最优的模式, 而不是完整覆盖 teacher model 的全部输出分布:
$$
\begin{aligned}
& D_{KL}(\pi_{\theta} \parallel \pi_{teacher}) \\
&= \int \pi_{\theta}(x) \log \frac{\pi_{\theta}(x)}{\pi_{teacher}(x)} dx \\
&= \Sigma_x \pi_{\theta}(x) \log \frac{\pi_{\theta}(x)}{\pi_{teacher}(x)} \\
&= \Sigma_x \pi_{\theta}(x) \log \frac{\prod_{t=1}^{T} \pi_{\theta}(x_t | x_{<t})}{\prod_{t=1}^T \pi_{teacher}(x_t | x_{<t})} \\
&= \Sigma_x \pi_{\theta}(x) \Sigma_{t=1}^T \log \frac{\pi_{\theta}(x_t | x_{<t})}{\pi_{teacher}(x_t | x_{<t})} \\
&= \Sigma_x \pi_{\theta}(x) \Sigma_{t=1}^T (\log \pi_{\theta}(x_t | x_{<t}) - \log \pi_{teacher}(x_t | x_{<t})) \\
&= \mathbb{E}_{x \sim \pi_{\theta}} [\Sigma_{t=1}^T \log \pi_{\theta}(x_t | x_{<t}) - \log \pi_{teacher}(x_t | x_{<t})] \\
\end{aligned}
$$

### *Implementation*
上述的 loss 计算逻辑可以用 python 代码表示如下:
```python
# Initialize teacher client (main):
teacher_client = service_client.create_sampling_client(
    base_model=teacher_config.base_model,
    model_path=teacher_config.load_checkpoint_path,
)

# Sample trajectories (main):
trajectories = do_group_rollout(student_client, env_group_builder)
sampled_logprobs = trajectories.loss_fn_inputs["logprobs"]

# Compute reward (compute_teacher_reverse_kl):
teacher_logprobs = teacher_client.compute_logprobs(trajectories)
reverse_kl = sampled_logprobs - teacher_logprobs
trajectories["advantages"] = -reverse_kl

# Train with RL (train_step):
training_client.forward_backward(trajectories, loss_fn="importance_sampling")
```

## Experiments
首先对 `Qwen3-8B-Base` 模型进行 mid-training, 得到性能-数据量曲线如下. 可以看到, 无论是全参数还是使用 LoRA, 性能都随着数据量实现了线性对数增长:
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/frontier-ai-articles-reading/2_mid_train.png" alt="Mid-training performance scaling">
  <figcaption>Figure 7: Mid-training performance scales with the amount of training data for full fine-tuning and LoRA.<br>(Image source: Thinking Machines Lab, "On-Policy Distillation")</figcaption>
</figure>

接下来, 以 400K 数据量的 checkpoint 作为出发点, 向后进行了一系列的对比实验。可以看出, on-policy distillation 在取得最好效果的同时, 其训练 compute cost 也明显低于 RL 方法.
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/frontier-ai-articles-reading/2_cost_compare.png" alt="Compute cost comparison between SFT, RL, and on-policy distillation">
  <figcaption>Figure 8: Performance and compute cost comparison across SFT, RL, and on-policy distillation.<br>(Image source: Thinking Machines Lab, "On-Policy Distillation")</figcaption>
</figure>

相比于 RL, on-policy distillation 的优势体现在以下几个方面:
- **Dense supervision greatly improves compute efficiency.** on-policy distillation 和 RL 都是通过 reverse KL 学习, 以此剪枝基础策略中存在的动作空间. 两者的区别在于奖励信号的稠密程度: RL 中的稠密程度为 $$O(1)$$; OPD 中的稠密程度为 $$O(N)$$. 由于这个优势, OPD 相比于 RL 需要更少的梯度更新就能取得更好的效果:
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/frontier-ai-articles-reading/2_efficiency.png" alt="Compute efficiency comparison between RL and on-policy distillation">
  <figcaption>Figure 9: Dense token-level supervision helps on-policy distillation reach strong performance with fewer updates.<br>(Image source: Thinking Machines Lab, "On-Policy Distillation")</figcaption>
</figure>

- **Distillation can effectively reuse training data for data efficiency.** 收集大规模训练的数据集可能昂贵又耗时, 因此我们希望能够重复利用已有数据集中的提示词。然而, 使用 RL 方法如果多次重复提示词, 模型往往会死记硬背最终答案; 而 OPD 则是让 student model 逼近 teacher model 的分布, 而不是记住最终答案, 因此可以多次利用同一个样本. 这里做了个有趣的实验: 在同一个 prompt 上连续训练 20 steps, 每一个 step 的 batch size 为 256, 发现 student model 仍然可以大致达到 teacher model 的水平:
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/frontier-ai-articles-reading/2_data.png" alt="Training on repeated prompts with on-policy distillation">
  <figcaption>Figure 10: On-policy distillation can reuse prompts while continuing to sample fresh student trajectories.<br>(Image source: Thinking Machines Lab, "On-Policy Distillation")</figcaption>
</figure>

- **RL searches in the space of semantic strategies.** RL 仍然通过梯度更新模型参数, 但它的有效搜索对象更接近语义策略: 模型通过 rollout 发现某类解题路径, reward 再把这些路径强化到参数中。这个过程需要反复采样、评估和 credit assignment, 因而 compute cost 较高。一旦 RL 找到了一个好策略, 则可以用 OPD 直接蒸馏学习 RL 找到的最终策略, 而不需要重走 RL 发现语义策略的完整过程, 这减少了大量的开销。我们可以打个比方: 在科学研究中, 我们花费了大量的时间和资源去寻找答案和探索新想法。一旦某个结果被发现, 通过自然语言将其表达出来并传授给他人就会简单得多。

- **On-policy learning as a tool for continual learning.** On-policy learning 相比于 off-policy learning 更不容易产生遗忘。然而, RL 只能用于规范模型的行为 (只能判断模型生成的内容的对错, 并不能直接注入新的领域知识); 而 SFT 之所以很难作为持续学习的框架, 是因为固定数据集会很快变成 off-policy 数据。即使数据集最初完全由模型自身输出构建, 单个 finite batch 的分布也不会和模型完整分布完全一致, 因此每个 batch 都会带来非零参数更新。模型一旦被更新, 后续尚未训练的数据就不再严格来自当前模型分布, 这种偏差会随着 batch 更新逐步累积, 最终损害模型既有行为。OPD 的关键在于持续从当前 student model 采样轨迹, 再用固定 teacher model 给这些当前轨迹打分, 因而可以在提供 dense supervision 的同时保持 on-policy。
<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/frontier-ai-articles-reading/2_on_policy_sft.png" alt="Comparison between on-policy distillation and on-policy SFT">
  <figcaption>Figure 11: On-policy distillation avoids the distribution drift that can appear when SFT reuses a fixed self-generated dataset.<br>(Image source: Thinking Machines Lab, "On-Policy Distillation")</figcaption>
</figure>

## Future Work
On-policy distillation 中 student model 是去拟合 teacher model 的分布, 这就决定了 student model 的性能基本不可能超过 teacher model. 因为一旦 student model 探索出了比 teacher model 更优质的思路, 也会因为受到 teacher model 的抑制. 另外, OPD 之后, 模型容易出现多样性崩塌的问题. 因为 OPD 中采用的 reverse KL 散度的目的是让 student model 拟合 teacher model 概率的最高的尖峰, 因此容易出现 Pass@1 很高, 但是 Pass@k 偏低的情况. 综上, 针对 OPD 的这些问题仍有改善的空间.

# References

<div class="references" markdown="1">

[1] Thinking Machines Lab, ["Interaction Models: A Scalable Approach to Human-AI Collaboration"](https://thinkingmachines.ai/blog/interaction-models/), Thinking Machines Lab: Connectionism, May 2026.

[2] Kevin Lu and Thinking Machines Lab, ["On-Policy Distillation"](https://thinkingmachines.ai/blog/on-policy-distillation/), Thinking Machines Lab: Connectionism, October 27, 2025.

</div>

</div>
