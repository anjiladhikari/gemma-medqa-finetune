# Gemma MedQA Finetune

## Current Goal

This project aims to build a local workflow for evaluating and fine-tuning a small Gemma-based model on a question-answering dataset using LoRA. The system is being developed step by step, with a focus on reusable scripts, clear project structure, and repeatable benchmarking before and after fine-tuning.

## What Has Been Done So Far

### 1. Project direction defined

The overall workflow for the project has been decided:

* explore the base model first
* test the model on sample questions without a system prompt
* prepare and clean the dataset
* create a reusable evaluation pipeline
* fine-tune using LoRA
* save the LoRA adapter for reuse
* evaluate the fine-tuned model
* compare results before and after fine-tuning

This gives the project a clear end-to-end structure instead of doing isolated experiments.

### 2. Local development approach chosen

The project will be developed locally rather than inside notebooks.
The planned stack is:

* **Backend:** FastAPI
* **Frontend:** React + Vite (later stage)
* **Model experimentation:** local Python scripts

At this stage, the main focus is on the model pipeline, not the frontend.

### 3. Practical project structure approach decided

Instead of creating a large number of folders and files at once, the approach is to:

* think about the final structure first
* create files and folders only when needed
* keep the code modular
* avoid wasting time on unused structure

This keeps development simpler and more flexible in the early stage.

### 4. Early model performance reviewed

Initial model results were checked using the available question-answering data.
Some early benchmark metrics already looked promising, including a strong semantic score (around **0.89**).

This suggests that the current setup already has useful potential, but it also raised an important question about possible overfitting, especially because the dataset size is relatively small (around **5k QA pairs**).

### 5. Need for proper evaluation identified

Even though the early metrics look good, it was noted that strong results alone are not enough unless they are measured on properly separated unseen data.

Because of that, the next important steps were identified as:

* clean the dataset
* prepare train/validation/test splits
* evaluate the base model on held-out data
* fine-tune only after the benchmark pipeline is ready

This helps ensure that improvements after fine-tuning are real and not just memorization.

## Current Status

The project is currently at the **data preparation stage**.

The next steps are:

1. document progress in the README
2. clean the raw QA dataset
3. split the data into train/validation/test
4. run baseline evaluation on the base model
5. start LoRA fine-tuning
6. evaluate and compare results after fine-tuning

## Development Principles

The project is being built with the following principles:

* local-first workflow
* reusable scripts for repeated experiments
* modular structure
* measurable evaluation before training
* saved LoRA adapters for efficient reuse
* clear comparison between base and fine-tuned model
