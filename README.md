# DiffFPR: Diffusion Prior for Oversampled Fourier Phase Retrieval

[Ji Li](https://chilie.github.io/cv-en.html), [Chao Wang](https://scholar.google.com/citations?user=57qzWYMAAAAJ&hl=en&oi=sra).

This repository contains the code with the paper "DiffFPR: Diffusion Prior for Oversampled Fourier Phase Retrieval", which is accepted by ICML 2024.

This code is based on the [OpenAI Guided Diffusion](https://github.com/openai/guided-diffusion) and [DiffPIR](https://github.com/yuanzhi-zhu/DiffPIR).


___________

## Setting Up

### Clone and Install

```bash

git clone https://github.com/yuanzhi-zhu/DiffPIR.git

cd DiffPIR

pip install -r requirements.txt

```
  

### Model Download

From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" and paste it to ./model_zoo/
```
mkdir model_zoo
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./model_zoo/

```

Do not forget to rename "ffhq_10m" to "diffusion_ffhq_m" for code consistency.

  

### Inference Code

```python

python main_pr.py

```
  
  

## Citation

If you find this repo helpful, please cite:

  

```bibtex

@inproceedings{li2024diff, % DiffFPR

title={DiffFPR: Diffusion Prior for Oversampled Fourier Phase Retrieval},

author={Ji Li and Chao Wang},

booktitle={Proceedings of the 41th international conference on machine learning},

year={2024},

}

```

  
  

## Acknowledgments

```This work was partly supported by the fund from Capital Normal University.```