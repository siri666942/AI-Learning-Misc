# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rule

**调用工具，不要瞎猜。** Always Read a file before answering questions about its contents. Never rely on prior conversation context — the user may have edited the file since your last interaction. Reading a file is cheap; guessing wrong wastes the user's time and erodes trust.

任何值得持久记忆可以复用的经验都要主动记下来；

写完

## What This Repository Is

A personal deep learning study portfolio by Siri. It is **not** a software application — there is no build system, no package manager, no test suite, no CI/CD. The primary artifacts are Jupyter notebooks and Python scripts organized by learning resource.

## Language & Frameworks

- **Python** with **Jupyter notebooks** (.ipynb) as the primary medium
- **PyTorch** is the main deep learning framework (d2l, karpathy notebooks)
- **NumPy** used in Andrew Ng course assignments (from-scratch implementations)
- **Graphviz** for neural network visualization (karpathy/engine.ipynb)
- **scikit-learn**, **Matplotlib** for data handling and plotting

## Repository Structure

```
karpathy/          — Karpathy's micrograd: autograd engine + MLP from scratch (engine.ipynb)
d2l/               — "Dive into Deep Learning" (d2l.ai) textbook exercises, PyTorch
吴恩达/             — Andrew Ng Coursera course assignments (Python scripts + utilities)
Neural Networks and Deep Learning/  — Same course, notebook format with quiz answers
prac/              — Practical work: Kaggle store-sales competition, chatbot experiments
dataset/           — CIFAR-10, cat/not-cat images
data/              — FashionMNIST
```

## Running Code

There are no standard build/test/lint commands. To run notebooks:
- Open .ipynb files in Jupyter Lab/Notebook or VS Code
- Python scripts can be run directly: `python <script>.py`
- The `push.ps1` script handles `git add . && git commit -m "add" && git push origin main`

## Key Dependencies (not in any requirements file)

`torch`, `numpy`, `matplotlib`, `scikit-learn`, `graphviz`, `openai` (for prac/without_ai.py only)

## Conventions

- Notebook filenames match their source material chapter/section numbers (e.g., `3.5.ipynb`, `8.1.ipynb`)
- The `karpathy/engine.ipynb` notebook is the most self-contained and substantial piece — it implements a full micro-autograd `Value` class with backward pass, then `Neuron`/`Layer`/`MLP` on top, verified against PyTorch
- Comments and documentation are primarily in Chinese
- The README tracks learning progress across 4 resources: Karpathy (foundations), d2l.ai (main textbook), Andrew Ng (completed), fast.ai (not started)
