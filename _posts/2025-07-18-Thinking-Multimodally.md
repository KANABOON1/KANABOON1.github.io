---
title: 'Thinking Multimodally'
date: 2025-07-18
permalink: /posts/how-we-think/
read_time: 40
author: Muxin Fu
tags:
  - Test-Time Compute
  - Multimodal
description: 'AI model 往往缺乏自主的 "pause and think" 能力, 他们的回答主要依赖于预训练时学习到的固定的回答模式, 这种模式通常被称为 System 1 reasoning。与之相对, System 2 reasoning 更接近人类的思维方式...'
---

<style>
.how-we-think-post {
  --hwt-ink-soft: #5d6865;
  --hwt-border: rgba(0, 0, 0, 0.12);
  --hwt-shadow: 0 18px 46px rgba(22, 42, 38, 0.12);
}
.how-we-think-post p,
.how-we-think-post li {
  line-height: 1.82;
}
.how-we-think-post h1 {
  margin-top: 2.4em;
  padding-bottom: 0.28em;
  border-bottom: 1px solid var(--hwt-border);
}
.how-we-think-post h2 {
  margin-top: 2.0em;
}
.how-we-think-post h3 {
  margin-top: 1.6em;
}
.how-we-think-post .post-figure {
  margin: 2.1rem auto;
  text-align: center;
}
.how-we-think-post .post-figure img {
  display: block;
  width: 80%;
  max-width: 860px;
  height: auto;
  margin: 0 auto;
  border: 1px solid var(--hwt-border);
  border-radius: 16px;
  background: #fff;
  box-shadow: var(--hwt-shadow);
}
.how-we-think-post .post-figure--wide img {
  max-width: 980px;
}
.how-we-think-post figcaption {
  max-width: 780px;
  margin: 0.85rem auto 0;
  font-size: 0.92em;
  line-height: 1.65;
}
.how-we-think-post .references p {
  margin: 0.75rem 0;
  padding-left: 2.2rem;
  text-indent: -2.2rem;
  line-height: 1.68;
}
@media (max-width: 720px) {
  .how-we-think-post .post-figure {
    margin: 1.6rem -0.25rem;
  }
  .how-we-think-post .post-figure img {
    border-radius: 12px;
  }
}
</style>

* TOC
{:toc}

<div class="how-we-think-post" markdown="1">

***Special thanks to [Lilian Weng](https://scholar.google.com/citations?user=dCa-pW8AAAAJ&hl=en) for her inspiring blog posts, which motivated me to start writing my own.***

# Motivation

AI model 往往缺乏自主的 "pause and think" 能力，它们的回答主要依赖于预训练时学习到的固定回答模式，这种模式通常被称为 *System 1 reasoning*。与之相对，*System 2 reasoning* 更接近人类的思维方式：一个真正具备推理能力的模型，应能够根据题目的难度动态分配合适的 *computational resource*，在简单的问题上花费少量计算资源，在困难的问题上投入更多计算资源。

因此，模型在何种情况下需要引入 test-time compute 技术，以及应当采用何种具体方法，正是当前 test-time compute 领域的研究热点。

> We are given a prompt and a test-time compute budget within which to solve the problem. Under the abstraction above, there are different ways to utilize test-time computation. Each of these methods may be more or less effective depending on the specific problem given. How can we determine the most effective way to utilize test-time compute for a given prompt? And how well would this do against simply utilizing a much bigger pretrained model?

# A Unified Perspective on Test-Time Computation: Proposer and Verifier

*Scaling LLM Test-Time Compute Optimally* [(Snell et al., 2024)](https://arxiv.org/abs/2408.03314) 将 test-time compute 技术分为两类：

1. **Modifying the proposal distribution**：输入层面的改动。它通过改变 LLM 原有的 *proposal distribution*，也就是给定输入条件下模型对下一个 token 的概率分布，来影响后续生成。

2. **Optimizing the verifier**：输出层面的改动。它不直接改变 LLM 的 *proposal distribution*，而是针对同一个问题生成的 \(M\) 个并行回答，利用 **verifier** 来聚合、排序或选择最优答案。

对于第一类方法，常见做法包括：

- **Fine-Tuning**：直接对 LLM 的参数进行进一步训练，从而调整其生成分布。常用方法包括 slow-thinking supervised fine-tuning 和 reinforcement fine-tuning。
- **Self-Improvement**：让 LLM 在生成过程中不断进行自我检查与修正，通过迭代更新答案来逐步优化输出。

对于第二类方法，常见做法包括：

- **Reward Modeling**：PRM，即 process reward model，在 LLM 的长链推理过程中提供稠密奖励信号，帮助模型评估当前回答的质量。相比 ORM，即 outcome reward model，PRM 提供了更稠密的奖励环境，也更加符合人类的推理习惯：最终推理结果的正确通常依赖于每一步小推理的正确。
- **Search Methods**：非结构化搜索，如 best-of-N、beam search、lookahead search，以及结构化搜索，如 MCTS，可以依据 reward model 提供的奖励，有效平衡 LLM 推理过程中的**探索**和**利用**，从而引导模型探索并选择更高质量的推理路径。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/how-we-think/parallel_sampling.png" alt="Parallel sampling and sequential revision for test-time compute">
  <figcaption>Figure 1: Illustration of parallel sampling vs sequential revision. (Image source: Snell et al. 2024)</figcaption>
</figure>

# Thinking in Continuous Space

## Latent Tokens

Latent tokens 指在 LLM 训练或者推理过程中动态添加的一批**隐式** tokens。这些 tokens 往往不承载显式语义，其主要作用有两个方面：

- **扩展 LLM 的表达能力**：由于插入的 latent tokens 仅存在于 LLM 的隐藏状态中，不受具体 token 级别语义的制约，因此可以显著提升 LLM 的 expressive bandwidth。
- **增加计算资源**：它们为模型提供额外的“思考时间”和计算能力，从而有助于生成更复杂、更准确的推理过程。

在 *Deliberation in Latent Space* [(Liu et al., 2024)](https://arxiv.org/abs/2412.17747) 中，作者使用一个 coprocessor model 来接收 LLM 当前的 KV cache。该 coprocessor 根据传入的 KV cache 生成一批 latent tokens，并将它们插入到 LLM 的 KV cache 中，实现 cache augmentation 的效果。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/how-we-think/deliberation_via_cache.png" alt="Deliberation in latent space via differentiable cache augmentation">
  <figcaption>Figure 2: Deliberation in latent space via differentiable cache augmentation. (Image source: Liu et al. 2024)</figcaption>
</figure>

值得注意的是，这里仅对 coprocessor model 进行微调，而非直接微调 LLM 本身。这样做的目的是保留 LLM 在预训练阶段获得的能力，同时避免直接微调 LLM 导致灾难性遗忘。

在 *SoftCoT* [(Xu et al., 2025)](https://arxiv.org/abs/2502.12134) 中，作者将插入的 latent tokens 视为隐式 CoT，即 implicit form of chain-of-thought。在 LLM 正式开始回答之前，辅助模块会根据输入 prompt 先生成一批 latent tokens，并将它们拼接到 LLM 的输入 prompt 之后，然后再由 LLM 对问题进行回答。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/how-we-think/softcot.png" alt="SoftCoT framework for latent chain-of-thought reasoning">
  <figcaption>Figure 3: Illustration of SoftCoT, where latent tokens act as an implicit chain-of-thought before answer generation. (Image source: Xu et al. 2025)</figcaption>
</figure>

与 *Coconut* [(Hao et al., 2024)](https://arxiv.org/abs/2412.06769) 通过自回归方式逐个生成 latent token 不同，*SoftCoT* 中的 latent tokens 是通过一批 query tokens 获取对应位置的隐藏状态，然后通过一个 projection layer 将辅助模型的隐藏状态映射到 LLM 的隐藏状态中，实现对 LLM 的直接增强。

在很多 thinking token 的设置中，通常不会修改原始 LLM 参数，而是仅微调 thinking tokens 本身或者辅助模块的参数。实验结果表明，仅通过微调这些 thinking tokens 或者辅助模块的参数，就能够达到与直接微调 LLM 参数相当、甚至更优的效果。这一现象暗示：插入的 thinking tokens 拥有足够大的 expressive bandwidth，同时也说明 LLM 的表达能力在很大程度上确实受到 token-level supervision 的制约。

# Thinking with Image

Thinking-with-image 系列工作很大程度上沿用了 thinking-with-text 的核心思路：让多模态大语言模型（MLLM）在给定图像和问题后，**显式输出文本形式的推理步骤**，从而强化模型的推理能力。

*Multimodal-CoT* [(Zhang et al., 2023)](https://arxiv.org/abs/2302.00923) 将多模态问答任务拆解成两个阶段：rationale generation 和 answer inference。在第一阶段，作者将图片和问题同时输入给第一个 MLLM，该 MLLM 负责生成针对该问题的推理；在第二阶段，第二个 MLLM 接收来自第一个 MLLM 的推理，以及原始图片和问题，输出最终答案。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/how-we-think/multimodal_cot.png" alt="Multimodal-CoT two-stage rationale generation and answer inference">
  <figcaption>Figure 4: Multimodal-CoT decomposes multimodal reasoning into rationale generation and answer inference. (Image source: Zhang et al. 2023)</figcaption>
</figure>

该方法使用 SFT 训练模型。由于 *ScienceQA* 和 *A-OKVQA* 提供了人工标注的思维链，便于进行 SFT，因此作者选用了这两个数据集进行训练和测试。

与 SFT 训练模型相对，自从 *DeepSeek-R1* [(Guo et al., 2025)](https://arxiv.org/pdf/2501.12948) 证明强化学习能够显著提升模型的推理能力之后，也有很多工作尝试在多模态领域利用 GRPO 等强化学习后训练方法，从而赋予 MLLM 多模态推理能力。

*Visual-RFT* [(Liu et al., 2025)](https://arxiv.org/abs/2503.01785)、*VLM-R1* [(Shen et al., 2025)](https://arxiv.org/abs/2504.07615)、*R1-V* [(Chen et al., 2025)](https://deepagent.notion.site/rlvr-in-vlms) 等工作将 GRPO 方法从文本模态迁移到多模态任务中，并取得了显著效果，进一步证明强化学习算法同样可以增强 MLLM 的推理能力。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/how-we-think/visual_rft.png" alt="Visual-RFT framework for visual reinforcement fine-tuning">
  <figcaption>Figure 5: Visual-RFT applies reinforcement fine-tuning to visual reasoning tasks. (Image source: Liu et al. 2025)</figcaption>
</figure>

需要意识到，多模态推理方式与纯文本推理有所不同：人类在解决视觉问题时往往是边看边思考的。如果仅仅利用一次图片信息进行推理，就无法充分模拟这种“动态观察—推理”的过程，容易遗漏关键信息，导致推理不够全面。

*More Thinking, Less Seeing?* [(Liu et al., 2025)](https://arxiv.org/abs/2505.21523) 发现：如果仅将图像作为视觉输入送入多模态大模型的感知层，模型对视觉信息的关注度会非常低。尤其在 VQA 形式的 Chain-of-Thought 推理中，模型往往更倾向于依赖其生成的文本信息，从而进一步忽视图像内容，导致视觉信息严重丢失，并容易引发视觉幻觉（hallucination）。值得注意的是，这种现象在经过强化学习增强推理能力的模型中表现得更为严重。

因此，为了降低多模态大模型的幻觉并提升其推理能力，将图片这一视觉信息动态地融入 MLLM 的推理过程，而非单纯进行文本推理，是一种十分自然的思路。而一旦涉及到在推理过程中动态利用图片，通常就需要 agent 调用相应的图像处理工具。

*Visual Sketchpad* [(Hu et al., 2024)](https://arxiv.org/abs/2406.09403) 针对不同任务场景，例如数学几何题、复杂视觉推理，允许模型通过 Python 代码调用特定工具进行处理。例如在数学函数题中使用 matplotlib 绘制函数图像，让模型边思考边使用工具处理图片。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/how-we-think/visual_sketchpad.png" alt="Visual Sketchpad framework for visual chain-of-thought">
  <figcaption>Figure 6: Visual Sketchpad enables multimodal models to use sketches and visual tools during reasoning. (Image source: Hu et al. 2024)</figcaption>
</figure>

基于 ICL（in-context learning）的工具调用方式缺少模型对工具使用的自主探索，并且缺乏灵活性。因此，*VisualToolAgent* [(Huang et al., 2025)](https://arxiv.org/abs/2505.20289) 使用 GRPO 训练了一个 visual tool agent，专门用于根据当前任务选择适当的工具。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/how-we-think/visual_tool_agent.png" alt="VisualToolAgent reinforcement learning framework for visual tool selection">
  <figcaption>Figure 7: VisualToolAgent learns to select visual tools through reinforcement learning. (Image source: Huang et al. 2025)</figcaption>
</figure>

另外，实验表明这个通过 RL 训练的 agent 可以迁移到其他多模态大模型上，说明对工具的使用也是一种泛化的知识。

与基于 workflow 的方式不同，*DeepEyes* [(Zheng et al., 2025)](https://arxiv.org/abs/2505.14362) 提出了用端到端强化学习方式训练多模态大模型的 grounding 和推理能力。*DeepEyes* 利用强化学习训练模型自主判断、自主调用工具的能力：当模型认为当前线索不足时，会调用图片裁剪工具处理原始图片，以收集更多细节；当模型认为已有线索足够时，模型会输出问题的最终答案。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/how-we-think/deepeyes.png" alt="DeepEyes end-to-end reinforcement learning for thinking with images">
  <figcaption>Figure 8: DeepEyes trains the model to decide when to inspect image regions and when to answer. (Image source: Zheng et al. 2025)</figcaption>
</figure>

# Thinking with Video

视频领域同样涌现出了大量 thinking-with-video 相关工作。由于视频数据同时具有时间和空间两个属性，如何高效利用这两个属性，已成为当前研究的重心。

## Long-term Video Understanding

### Frame Sampling

从时间属性角度出发，视频数据包含大量帧，其中存在显著冗余信息，很多帧与问题无关。一个自然的思路是筛选视频中与题目有关的关键帧，而不是简单地进行等间距抽帧。

*CoS* [(Hu et al., 2025)](https://arxiv.org/abs/2502.06428) 将关键帧选择视为 **test-time visual prompt optimization**，通过调用 agent 对视频帧进行区分，将其划分为“有用帧”和“无用帧”。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/how-we-think/cos.png" alt="CoS chain-of-shot prompting for long video understanding">
  <figcaption>Figure 9: CoS treats key-shot selection as test-time visual prompt optimization for long-video understanding. (Image source: Hu et al. 2025)</figcaption>
</figure>

在此基础上，*Temporal CoT* [(Arnab et al., 2025)](https://arxiv.org/abs/2507.02001) 进一步将关键帧选择与文字标注融合，旨在增强视频输入特征。在 *Temporal CoT* 中，作者使用两个 VLM：第一个 VLM 筛选出关键帧并对这些帧进行文字标注；第二个 VLM 根据筛选出来的关键帧回答问题。

这样的做法与人类在回答视频问题时有一定程度上的相似：我们在看视频时，会停顿在关键时间点并记录笔记以防止遗忘；然后在回答问题时，会回到之前记录的重要帧，并结合已有笔记重新理解，最后做出回答。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/how-we-think/temporal_cot.png" alt="Temporal Chain of Thought for long-video understanding">
  <figcaption>Figure 10: Temporal CoT combines key-frame selection with textual notes for long-video reasoning. (Image source: Arnab et al. 2025)</figcaption>
</figure>

### Token Compression

Token compression 是另一条自然路线：不一定要选择少量关键帧，也可以把长视频压缩成更短、更结构化的视觉 token 表示。这样可以在保留关键时空信息的同时，降低上下文长度和计算成本。

### Tool-integrated Video Understanding

从空间属性角度来看，视频数据中每一帧都包含大量物体及其相互关系。并且与图像数据中静态的物体关系不同，视频数据呈现的是动态演化的物体关系，这使得建模更具挑战性。

*Video-of-Thought* [(Fei et al., 2025)](https://arxiv.org/abs/2501.03230) 提出了一个以 **video spatial-temporal scene graph (video STSG)** 为核心的多阶段推理框架。该方法采用 tool-using 范式：首先通过 STSG 捕捉视频中的物体及其动态交互关系，然后将 STSG 转为文本形式，以便后续推理模型进一步处理。

其核心思想依然属于 **test-time visual prompt optimization**，但创新点在于利用新颖的 STSG 结构对视频输入进行增强和预处理，从而更好地建模视频中的时空关系。

<figure class="post-figure post-figure--wide">
  <img src="/assets/posts/how-we-think/stsg.png" alt="Video spatial-temporal scene graph for Video-of-Thought">
  <figcaption>Figure 11: Video-of-Thought uses video spatial-temporal scene graphs to structure visual information before reasoning. (Image source: Fei et al. 2025)</figcaption>
</figure>

# References

<div class="references" markdown="1">

[1] Charlie Snell, et al. ["Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters."](https://arxiv.org/abs/2408.03314) arXiv preprint arXiv:2408.03314 (2024).

[2] Luyang Liu, et al. ["Deliberation in Latent Space via Differentiable Cache Augmentation."](https://arxiv.org/abs/2412.17747) arXiv preprint arXiv:2412.17747 (2024).

[3] Yige Xu, et al. ["SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs."](https://arxiv.org/abs/2502.12134) arXiv preprint arXiv:2502.12134 (2025).

[4] Shibo Hao, et al. ["Training Large Language Models to Reason in a Continuous Latent Space."](https://arxiv.org/abs/2412.06769) arXiv preprint arXiv:2412.06769 (2024).

[5] Zhuosheng Zhang, et al. ["Multimodal Chain-of-Thought Reasoning in Language Models."](https://arxiv.org/abs/2302.00923) Transactions on Machine Learning Research (2024).

[6] DeepSeek-AI. ["DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning."](https://arxiv.org/abs/2501.12948) Nature 645, 633-638 (2025).

[7] Liangchen Liu, et al. ["Visual-RFT: Visual Reinforcement Fine-Tuning."](https://arxiv.org/abs/2503.01785) arXiv preprint arXiv:2503.01785 (2025).

[8] Haozhan Shen, et al. ["VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model."](https://arxiv.org/abs/2504.07615) arXiv preprint arXiv:2504.07615 (2025).

[9] Liang Chen, et al. ["R1-V: Reinforcing Super Generalization Ability in Vision-Language Models with Less Than $3."](https://deepagent.notion.site/rlvr-in-vlms) Project report (2025).

[10] Chengzhi Liu, et al. ["More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models."](https://arxiv.org/abs/2505.21523) arXiv preprint arXiv:2505.21523 (2025).

[11] Yushi Hu, et al. ["Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models."](https://arxiv.org/abs/2406.09403) NeurIPS 2024.

[12] Zeyi Huang, et al. ["VisualToolAgent (VisTA): A Reinforcement Learning Framework for Visual Tool Selection."](https://arxiv.org/abs/2505.20289) arXiv preprint arXiv:2505.20289 (2025).

[13] Yiheng Zheng, et al. ["DeepEyes: Incentivizing \"Thinking with Images\" via Reinforcement Learning."](https://arxiv.org/abs/2505.14362) arXiv preprint arXiv:2505.14362 (2025).

[14] Jian Hu, et al. ["CoS: Chain-of-Shot Prompting for Long Video Understanding."](https://arxiv.org/abs/2502.06428) arXiv preprint arXiv:2502.06428 (2025).

[15] Anurag Arnab, et al. ["Temporal Chain of Thought: Long-Video Understanding by Thinking in Frames."](https://arxiv.org/abs/2507.02001) arXiv preprint arXiv:2507.02001 (2025).

[16] Hao Fei, et al. ["Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition."](https://arxiv.org/abs/2501.03230) ICML 2024.

[17] Rendy Satria Dalimunthe. ["Understanding test-time compute: A New Mechanism Allowing AI to Think Harder."](http://medium.com/@rendysatriadalimunthe/understanding-test-time-compute-a-new-mechanism-allowing-ai-to-think-harder-19e017abc540) Medium (2024).

</div>

</div>
