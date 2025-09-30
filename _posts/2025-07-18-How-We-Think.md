---
title: 'How We Think'
date: 2025-07-18
permalink: /posts/how-we-think/
read_time: 40
author: Muxin Fu
tags:
  - Test-Time Compute
  - Multimodal
description: 'AI model 往往缺乏自主的 "pause and think" 能力, 他们的回答主要依赖于预训练时学习到的固定的回答模式, 这种模式通常被称为 System 1 reasoning。与之相对, System 2 reasoning 更接近人类的思维方式...'
---

* TOC
{:toc}

***Special thanks to [Lilian Weng](https://scholar.google.com/citations?user=dCa-pW8AAAAJ&hl=en) for her inspiring blog posts, which motivated me to start writing my own.***

# Motivation
AI model 往往缺乏自主的 "pause and think" 能力, 他们的回答主要依赖于预训练时学习到的固定的回答模式, 这种模式通常被称为 *System 1 reasoning*。与之相对,  *System 2 reasoning* 更接近人类的思维方式: 一个真正具备推理能力的模型, 应能够根据题目的难度, 动态地分配合适的 *computational resource* —— 在简单的问题上花费少量的计算资源, 在困难的问题上花费较多的计算资源。

因此，模型在何种情况下需要引入 test-time compute 技术，以及应当采用何种具体方法，正是当前 test-time compute 领域的研究热点。
> We are given a prompt and a test-time compute budget within which to solve the problem. Under the abstraction above, there are different ways to utilize test-time computation. Each of these methods may be more or less effective depending on the specific problem given. How can we determine the most effective way to utilize test-time compute for a given prompt? And how well would this do against simply utilizing a much bigger pretrained model?

# A Unified Perspective on Test-Time Computation: Proposer and Verifier
**Scaling LLM test-time compute Optimally**[(Snell et al. 2024)](https://arxiv.org/abs/2408.03314) 将 test-time compute 技术分为两类:
1. **Modifying the proposal distribution**(输入层面)  
   这一类方法通过改变 LLM 原有的 *proposal distribution*(即在给定输入条件下，模型对下一个 token 的概率分布)。主要手段包括两种：  
   - **Fine-Tuning**：直接对 LLM 的参数进行进一步训练，从而调整其分布。常用方法: Slow-thinking Supervised Fine-Tuning, Reinforcemennt Fine-Tuning
   - **Self Improvement**：让 LLM 在生成过程中不断进行自我检查与修正，通过迭代更新答案来逐步优化输出。

2. **Optimizing the verifier**(输出层面)
   这一类方法并不改变 LLM 的 *proposal distribution*，而是针对同一个问题生成的 \(M\) 个并行回答，利用 **verifier** 来聚合或选择最优答案。  
   - **Reward Modeling**: PRM(process reward model) 的作用是在 LLM 的长链推理过程中，提供稠密奖励信号，从而帮助模型评估当前回答的质量。相比于 ORM(observation reward model), PRM 提供了更稠密的奖励环境, 并且 PRM 也更加符合人类的推理习惯——最终推理结果的正确依赖于每一步小推理的正确(避免了推理错误但是答案正确的情况)。
   - **Search Methods**: 非结构化搜索(best-of-N, beam search, lookahead search)以及结构化搜索(MCTS)算法依据 reward model 提供的奖励, 有效平衡了 LLM 推理过程中的`探索`和`利用`, 从而引导模型探索并选择更高质量的推理路径。
   ![alt text](/assets/posts/how-we-think/parallel_sampling.png)

# Thinking with Text

## Thinking in Tokens

## Thinking in Continuous Space

### Thinking tokens
Thinking tokens 指在 LLM 训练或者推理过程中, 动态添加的一批**隐式**的 tokens。这些 tokens 往往不承载显式语义, 其主要作用有两个方面:
- 扩展 LLM 的表达能力(expressive bandwidth): 由于插入的这批 latent tokens 仅存在于 LLM 的隐藏状态中, 不受具体 token 级别语义的制约, 因此可以显著提升 LLM 的表达能力。
- 增加计算资源: 它们为模型提供了额外的"思考时间"和计算能力，从而有助于生成更复杂、更准确的推理过程。

在 **Deliberation in Latent Space**[(Liu et al. 2024)](https://arxiv.org/abs/2412.17747) 中, 作者使用了一个 coprocessor model 来接收 LLM 当前的 kvcache。该 coprocessor 根据传入的 kvcache 生成一批 latent tokens, 并将它们插入到 LLM 的 kvcache 中, 实现 kvcache augmentation 的效果。

![alt text](/assets/posts/how-we-think/deliberation_via_cache.png)
值得注意的是, 这里仅对 coprocessor model 进行微调, 而非直接微调 LLM 本身。这样做的目的是保留 LLM 在预训练阶段获得的能力, 同时避免直接微调 LLM 导致灾难性遗忘。

在 **SoftCoT**[(Xu et al. 2025)](https://arxiv.org/abs/2502.12134) 中, 作者将插入的 latent tokens 视为隐式 CoT (implicit form of CoTs)。在 LLM 正式开始回答之前, 辅助模块会根据输入 prompt 先生成一批 latent tokens, 并将它们拼接到 LLM 的输入 prompt 之后, 然后再由 LLM 对问题进行回答。

![alt text](/assets/posts/how-we-think/softcot.png)

与 **Coconut**[(Hao et al. 2024)](https://arxiv.org/abs/2412.06769) 通过自回归方式逐个生成 latent token 不同, **SoftCoT** 中的 latent tokens 是通过一批 query tokens 获取对应位置的隐藏状态, 然后通过一个 projection layer 将辅助模型的隐藏状态映射到 LLM 的隐藏状态中, 实现对 LLM 的直接增强。 

在很多 thinking token 的设置中, 通常不会修改原始 LLM 参数, 而是仅微调 thinking tokens 本身或者辅助模块的参数。实验结果表明, 仅通过微调这些 thinking tokens 或者辅助模块的参数, 就能够达到与直接微调 LLM 参数相当, 甚至更优的效果。这一现象表明, 这暗示了插入的这些 thinking tokens 拥有足够大的 expressive bandwidth, 同时也暗示了 LLM 的表达能力在很大程度上确实受到 token-level supervision 的制约。

### Recurrent Architecture

# Thinking with Image
结合视觉的推理范式和单纯基于语言的推理范式有所不同: 人类在解决一个视觉类的问题时往往都是边看边想的。如果像传统的 VLM 一样只是一次性地将图片输入到 VLM 模型的感知模块转为 image embedding 之后变成了静态的 context, 则这种纯文本的推理方式在视觉领域上是有损的。在 **More Thinking, Less Seeing**[(Liu et al. 2025)](https://arxiv.org/abs/2505.21523) 发现: 如果仅将图像作为视觉输入送入多模态大模型的感知层, 模型对视觉信息的关注度会非常低。尤其在 VQA 形式的 Chain-of-Thought 推理中, 模型往往更倾向于依赖其生成的文本信息, 从而进一步忽视图像内容, 导致视觉信息严重丢失, 最终引发视觉幻觉(hallucination)。值得注意的是, 这种现象在经过强化学习以增强推理能力的模型中表现得更为严重。

因此, 为了降低多模态大模型幻觉并提高大模型的推理能力, 将图片这个视觉信息融入到多模态大模型的推理过程中(而不是单纯地进行文本推理)是十分自然的想法。

**Visual Sketchpad**[(Hu et al. 2024)](https://arxiv.org/abs/2406.09403) 中针对不同任务场景(例如数学几何题、复杂视觉推理)允许模型通过 python 代码的方式调用特定的工具进行处理(例如数学函数题使用 matplotlib 绘制函数图像) 边思考边使用工具处理图片。![Visual Sketchpad](/assets/posts/how-we-think/visual_sketchpad.png) 

基于 ICL (in-context learning) 的工具调用方式缺少了模型对工具使用的自身探索, 并且缺乏灵活性。因此, **VisualToolAgent**[(Huang et al. 2025)](https://arxiv.org/abs/2505.20289) 中使用了强化学习 GRPO 策略训练了一个 visual tool agent 专门用于根据当前的任务选择适当的工具。![Visual Tool Agent](/assets/posts/how-we-think/visual_tool_agent.png) 另外, 实验表明这个通过 RL 训练的 agent 可以迁移到其他的多模态大模型上, 说明了对工具的使用也是一种泛化的知识。

与基于 workflow 的方式不同, **DeepEyes**[(Zheng et al. 2025)](https://arxiv.org/abs/2505.20289) 提出了用端到端的强化学习方式训练多模态大模型的 Grounding 和推理能力。

# Thinking with Video


# References
1. [*Understanding test-time compute: A New Mechanism Allowing AI to “Think Harder”*](http://medium.com/@rendysatriadalimunthe/understanding-test-time-compute-a-new-mechanism-allowing-ai-to-think-harder-19e017abc540)