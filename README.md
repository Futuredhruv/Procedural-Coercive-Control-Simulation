# Procedural Synthetic Data Generation for Coercive Control Simulation

This repository contains the code, dataset, and validation metrics for our feasibility study on modelling implicit Narcissistic Personality Disorder (NPD) markers using procedural synthetic data.

## 📂 Repository Contents
* **`/dataset`**: Contains `scaled_forensic_corpus_v2.csv` (275 temporally-encoded conversational dyads).
* **`/code`**: Python/Jupyter scripts for data generation (Llama-3/Qwen), structural validation (SBERT), and transformer fine-tuning (DistilBERT).
* **`/figures`**: High-resolution vector graphics of attention maps and validation distributions.
* **`/manuscript`**: LaTeX source files and the compiled PDF of the paper.

## ⚙️ Methodology Highlight
To bypass the "Forensic Data Gap" without violating user privacy, we operationalised DSM-5 criteria into generative constraints. The pipeline introduces explicit temporal encoding (e.g., `<DELAY: 14_HOURS>`) to simulate the weaponisation of silence. The DistilBERT model achieved an 88% recall in detecting implicit abuse, significantly outperforming lexicon-based baselines (VADER).

## 📊 Quick Results
* **SBERT Semantic Overlap (Benign):** 0.404
* **SBERT Semantic Overlap (NPD):** 0.358 (p < .001)
* **DistilBERT Recall:** 0.88
* **Human-in-the-loop validation (Cohen's Kappa):** 0.84

## 📝 License & Usage
This code and dataset are provided for academic and forensic research purposes. Due to the sensitive nature of coercive control modelling, this system is designed strictly for human-in-the-loop investigative triage, not autonomous clinical diagnosis.
