
## Understanding Generalization in Role-Playing Models via Information Theory
 (https://arxiv.org/abs/2512.17270)


## A. Benchmark

The benchmark dataset is provided under `Benchmark/v15/`.

---

## B. R-EMID Implementation

### B.0 Environment Setup

1. **Dependencies**: (python=3.10) Install required packages via  
   ```bash
   pip install -r requirements.txt
   ```


2. **Model Preparation**: Place the LLM (e.g., Qwen3-8B) at  
   ```
   ../llm_path/Qwen/Qwen3-8B
   ```  
   *(Note: This path is relative to the project root and should reside outside the current repository directory.)*

---

### B.1 Computing R-EMID Metric (Direct Estimation)

To compute the R-EMID metric using the pre-trained estimator:

```bash
cd REMID_estimation
bash run.sh
```

---

### B.2 Training the R-EMID Estimator (Optional)

To train the R-EMID estimator from scratch (i.e., the CoRL algorithm described in the paper), run:

```bash
cd estimator_training
bash run_co_evolve.sh
```


---

> **Reference**  
>   
> ```bibtex
> @misc{li-2025-RPMG,
>       title={Understanding Generalization in Role-Playing Models via Information Theory}, 
>       author={Yongqi Li and Hao Lang and Fei Huang and Tieyun Qian and Yongbin Li},
>       year={2025},
>       eprint={2512.17270},
>       archivePrefix={arXiv},
>       primaryClass={cs.LG},
>       url={https://arxiv.org/abs/2512.17270}, 
> }
> ```
