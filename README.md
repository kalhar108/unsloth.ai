# Name: Kalhar Mayurbhai Patel
# SJSU ID: 019140511
# Unsloth AI: Complete Tutorial Series
## 5 Colab Notebooks for Modern LLM Training

This repository contains 5 comprehensive Colab notebooks demonstrating different modern AI training techniques using Unsloth.ai. Each notebook is fully documented with detailed explanations, code comments, and real-world examples.

---

## 📚 Notebooks Overview

### 1️⃣ **Colab 1: Full Fine-tuning with SmolLM2-135M**
**File**: `colab1_full_finetuning_smollm2.ipynb`
- Colab link: https://colab.research.google.com/drive/14Ac_p4RaGY0XuFpn_yFM-lzcBzF7KNan?usp=sharing
- Video link: 
**What it covers:**
- Full fine-tuning (all parameters updated, not LoRA)
- Using SmolLM2-135M (135 million parameters)
- Chat template formatting
- Instruction-following dataset (Alpaca)
- Training configuration optimized for small models

**Key Learning Points:**
- Difference between full fine-tuning and LoRA
- When to use full fine-tuning vs parameter-efficient methods
- Chat template formats (`<|im_start|>` pattern)
- Dataset formatting for instruction tuning
- Training hyperparameters

**Video Recording Tips:**
- Show model loading and explain SmolLM2 architecture
- Demonstrate dataset formatting step-by-step
- Explain training arguments (batch size, learning rate, etc.)
- Run inference and show before/after comparison
- Discuss memory usage and training time

---

### 2️⃣ **Colab 2: LoRA Fine-tuning with SmolLM2-135M**
**File**: `colab2_lora_finetuning_smollm2.ipynb`
- Colab link: https://colab.research.google.com/drive/1uBB4oEEw8ufPsXrQq8xDd4XtwkEZhR4E?usp=sharing
Video link: 
**What it covers:**
- LoRA (Low-Rank Adaptation) parameter-efficient fine-tuning
- Same dataset as Colab 1 for direct comparison
- LoRA configuration (rank, alpha, target modules)
- Adapter saving and merging
- Memory efficiency benefits

**Key Learning Points:**
- What is LoRA and how it works
- LoRA hyperparameters (r, lora_alpha, target_modules)
- Trainable parameters: ~2M vs 135M (1-2% of full model)
- Adapter architecture and weight matrices
- When to use LoRA (7B+ models, limited GPU)

---

### 3️⃣ **Colab 3: DPO Reinforcement Learning**
**File**: `colab3_dpo_reinforcement_learning.ipynb`
Colab link: https://colab.research.google.com/drive/1jEuyL-hM7LkH59VpIMka7wUyLbA0diwS?usp=sharing
Video link: 
**What it covers:**
- Direct Preference Optimization (DPO)
- Preference dataset format (prompt, chosen, rejected)
- Anthropic HH-RLHF dataset
- Human alignment without reward models
- DPO loss function and beta parameter

**Key Learning Points:**
- DPO vs traditional RLHF
- Preference dataset structure
- How DPO learns from preferences
- Beta temperature parameter
- Alignment for helpfulness and harmlessness


---

### 4️⃣ **Colab 4: GRPO Reasoning Model Training**
**File**: `colab4_grpo_reasoning_model.ipynb`
Colab link: https://colab.research.google.com/drive/14Ac_p4RaGY0XuFpn_yFM-lzcBzF7KNan?usp=sharing
Video link: 
**What it covers:**
- Group Relative Policy Optimization (GRPO)
- Training reasoning models (like DeepSeek-R1, OpenAI o1)
- Automatic reward function for math problems
- Multiple generations per problem
- Self-verification and iterative improvement

**Key Learning Points:**
- GRPO vs DPO (generated vs pre-labeled data)
- How reasoning models are trained
- Reward function design
- Multiple sampling for reasoning diversity
- Answer extraction and verification


---

### 5️⃣ **Colab 5: Continued Pretraining**
**File**: `colab5_continued_pretraining.ipynb`
Colab link: https://colab.research.google.com/drive/14Ac_p4RaGY0XuFpn_yFM-lzcBzF7KNan?usp=sharing
Video link: 
**What it covers:**
- Continued pretraining for domain adaptation
- Teaching models new languages or domains
- Raw text corpus creation (Python code examples)
- Causal language modeling objective
- Before/after domain knowledge comparison

**Key Learning Points:**
- Pretraining vs fine-tuning vs continued pretraining
- Domain adaptation strategy
- Corpus creation for specific domains
- Training hyperparameters (lower LR, more epochs)
- Real-world examples (BloombergGPT, CodeLlama, BioGPT)


---

## 🎥 Video Recording Guidelines

### General Recording Tips:
1. **Screen Resolution**: Use 1920x1080 for clarity
2. **Font Size**: Increase Colab font size (Ctrl/Cmd + +)
3. **Audio**: Use good microphone, minimize background noise
4. **Pacing**: Speak clearly and not too fast
5. **Length**: Aim for 10-15 minutes per notebook

### Recommended Recording Structure:

#### Introduction (1-2 minutes)
- Introduce yourself
- State which Colab this is (1-5)
- Overview of what will be covered
- Mention prerequisites (if any)

#### Code Walkthrough (8-12 minutes)
1. **Setup Phase**
   - Show pip install commands
   - Explain import statements
   
2. **Configuration Phase**
   - Explain model loading
   - Discuss hyperparameters
   - Show dataset preparation

3. **Training Phase**
   - Run training (you can speed up video if training takes long)
   - Explain loss curves
   - Discuss training progress

4. **Inference Phase**
   - Show test prompts
   - Run inference
   - Discuss results
   - Compare outputs

5. **Saving Phase**
   - Show model saving
   - Explain file structure
   - Discuss deployment options

#### Conclusion (1-2 minutes)
- Summarize key learnings
- Mention real-world applications
- Suggest next steps
- Encourage viewers to try the notebook

### Recording Software Options:
- **OBS Studio** (Free, powerful)
- **Loom** (Easy to use, web-based)
- **Zoom** (Record screen share)
- **Camtasia** (Professional, paid)
- **ScreenFlow** (Mac, paid)

### Video Editing Tips:
- Cut out long pauses
- Speed up repetitive parts (2x speed)
- Add captions/subtitles if possible
- Include code snippets as overlays
- Add chapter markers for different sections

---

## 🚀 Running the Notebooks

### Prerequisites:
- Google Colab account (free tier works)
- GPU runtime enabled (Runtime → Change runtime type → T4 GPU)
- Basic Python knowledge

### Execution Order:
1. Start with **Colab 1** (Full Fine-tuning) - Foundation
2. Then **Colab 2** (LoRA) - Compare with Colab 1
3. Then **Colab 3** (DPO) - Learn alignment
4. Then **Colab 4** (GRPO) - Advanced reasoning
5. Finally **Colab 5** (Continued Pretraining) - Domain adaptation

### Runtime Estimates (on T4 GPU):
- Colab 1: ~5-10 minutes
- Colab 2: ~3-7 minutes (faster than full FT)
- Colab 3: ~5-10 minutes
- Colab 4: ~8-12 minutes (multiple generations)
- Colab 5: ~10-15 minutes (multiple epochs)

---

## 📊 Comparison Table

| Colab | Technique | Data Format | Trainable % | Use Case | Real Examples |
|-------|-----------|-------------|-------------|----------|---------------|
| 1 | Full FT | Q&A pairs | 100% | Small models | GPT-2 fine-tuning |
| 2 | LoRA | Q&A pairs | 1-2% | Large models | Llama adapters |
| 3 | DPO | Preferences | 1-2% | Alignment | Claude, GPT-4 |
| 4 | GRPO | Problems | 1-2% | Reasoning | o1, o3, DeepSeek-R1 |
| 5 | Continued PT | Raw text | 1-2% | Domain | BloombergGPT, CodeLlama |

---

## 🎓 Learning Path

### Beginner Level:
- Start with Colab 1 (Full Fine-tuning)
- Understand basic concepts
- Learn dataset formatting

### Intermediate Level:
- Move to Colab 2 (LoRA)
- Learn parameter efficiency
- Compare training approaches

### Advanced Level:
- Try Colab 3 (DPO) for alignment
- Explore Colab 4 (GRPO) for reasoning
- Use Colab 5 for domain specialization

---

## 🔧 Troubleshooting

### Common Issues:

**1. Out of Memory (OOM)**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`

**2. Slow Training**
- Ensure GPU is enabled (not CPU)
- Check GPU type (T4 or better)
- Reduce `max_steps` for quick tests

**3. Bad Results**
- Increase training steps/epochs
- Adjust learning rate
- Check dataset quality
- Ensure proper data formatting

**4. Import Errors**
- Re-run pip install cells
- Restart runtime
- Check Unsloth version

---

## 📚 Additional Resources

### Unsloth Documentation:
- Main Docs: https://docs.unsloth.ai/
- Getting Started: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide
- GitHub: https://github.com/unslothai/notebooks/

### Research Papers:
- **LoRA**: "LoRA: Low-Rank Adaptation of Large Language Models"
- **DPO**: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- **GRPO**: DeepSeek-R1 technical report

### Related Tutorials:
- Unsloth Blog: https://unsloth.ai/blog
- HuggingFace TRL: https://huggingface.co/docs/trl/
- Weights & Biases guides

---

## 🤝 Contributing

If you improve these notebooks or create video tutorials:
1. Share on GitHub
2. Tag @unslothai on Twitter/X
3. Post on HuggingFace forums
4. Submit to awesome-llm lists

---

## ⚖️ License

These notebooks are provided for educational purposes. 
- Code: MIT License
- Models: Follow individual model licenses (SmolLM2, etc.)
- Datasets: Follow dataset licenses (Alpaca, HH-RLHF, etc.)

---

## 🎯 Next Steps After Completing All Notebooks

1. **Try Larger Models**
   - Experiment with Llama 3.1 8B
   - Try Mistral 7B or Gemma 7B
   - Compare results with SmolLM2

2. **Use Your Own Data**
   - Prepare custom datasets
   - Try domain-specific tasks
   - Evaluate on your benchmarks

3. **Deploy Models**
   - Export to GGUF for Ollama
   - Deploy on HuggingFace Spaces
   - Serve with vLLM or TGI

4. **Advanced Techniques**
   - Combine DPO + GRPO
   - Multi-stage training pipeline
   - Quantization experiments

5. **Build Applications**
   - RAG systems with fine-tuned models
   - Task-specific AI assistants
   - Domain expert chatbots

---

## 📈 Performance Benchmarks

Expected performance improvements (approximate):

| Task | Base Model | After Colab 1-2 | After Colab 3 | After Colab 4-5 |
|------|------------|-----------------|---------------|-----------------|
| Instruction Following | 40% | 75% | 85% | 85% |
| Domain Knowledge | 30% | 60% | 65% | 90% |
| Reasoning | 25% | 45% | 50% | 75% |
| Safety/Alignment | 50% | 55% | 85% | 85% |

*Note: Actual results depend on data quality, training duration, and evaluation metrics*

---

## 🌟 Key Takeaways

### From All 5 Notebooks:

1. **Training Methods Matter**
   - Choose right method for your use case
   - LoRA for large models, Full FT for small
   - DPO for alignment, GRPO for reasoning

2. **Data Quality > Quantity**
   - Small, high-quality datasets work well
   - Format matters (Q&A vs raw text)
   - Domain relevance is crucial

3. **Hyperparameters Are Critical**
   - Learning rate affects convergence
   - Batch size impacts training stability
   - LoRA rank affects capacity

4. **Pipeline Approach**
   - Stage 1: Continued pretraining (domain)
   - Stage 2: Instruction tuning (task)
   - Stage 3: Alignment (preferences)

5. **Modern LLMs Are Accessible**
   - Can train on free Colab
   - Unsloth makes it fast
   - Community support available

---

