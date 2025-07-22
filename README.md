# EfficienceVSAdversarial

**Error Resiliency and Adversarial Robustness in CNNs: An Empirical Analysis**

This repository contains the code, experiments, and results presented in the following paper:

> M. Barbareschi, S. Barone, V. Casola, and S. Della Torca,  
> *Error Resiliency and Adversarial Robustness in Convolutional Neural Networks: An Empirical Analysis*,  
> IFIP International Internet of Things Conference (IFIP-IoT 2024), Springer, pp. 149–160.  
> [https://doi.org/10.1007/978-3-031-57202-3_12](https://doi.org/10.1007/978-3-031-57202-3_12)

## 📘 Overview

This work investigates the interplay between **approximate computing** and **adversarial robustness** in Convolutional Neural Networks (CNNs). It includes:

- Construction of Approximate Neural Networks (AxNNs) by replacing exact multipliers with approximate ones;
- Multi-Objective Optimization (via NSGA-II) to select optimal approximate configurations;
- Evaluation of multiple **white-box** (e.g., BIM, PGD, CW, DP) and **black-box** (e.g., One-Pixel) adversarial attacks on both original and approximate models.

## 🗂 Repository Structure

```
EfficienceVSAdversarial/
├── src/                     # All Python scripts live here
│   ├── generate_attacks.py
│   ├── evaluate_model.py
│   ├── build_axnn.py
│   ├── utils.py
│   └── ...
├── Models/                  # All the target models have to be here
├── Results/                 # Output figures, CSVs, statistics
│   ├── BlackBoxResults/
│   └── WhiteBoxResults/
└── README.md
```

## 🧪 Requirements

- Python ≥ 3.8
- TensorFlow
- numpy, pandas, matplotlib
- **inspectnn** ([PyPI link](https://pypi.org/project/inspectnn/))

Install required packages:

```bash
pip install -r requirements.txt
```

## 🧠 Models

- **MinNet**: a lightweight CNN
- **ResNet8**, **ResNet24**: standard ResNet variants
- Trained on **CIFAR-10**, using quantization-aware training
- AxNNs are built using 8-bit integer approximate multipliers from the EvoApprox8b library

## 🔐 Adversarial Attacks

Implemented attacks include:

- **White-box**:
  - PGD (Projected Gradient Descent)
  - BIM (Basic Iterative Method)
  - CW (Carlini & Wagner)
  - DP (DeepFool)
- **Black-box**:
  - One-Pixel Attack (using Differential Evolution)

## 📊 Results

The `Results/` folder contains:
- Boxplots, PSNR graphs, iteration counts
- Attack success rates in `.csv` format
- Transferability analysis

## 📄 Citation

If you use this repository, please cite the following paper:

```bibtex
@inproceedings{barbareschi2024error,
  title={Error Resiliency and Adversarial Robustness in Convolutional Neural Networks: An Empirical Analysis},
  author={Barbareschi, Mario and Barone, Salvatore and Casola, Valentina and Della Torca, Salvatore},
  booktitle={IFIP International Internet of Things Conference},
  pages={149--160},
  year={2024},
  organization={Springer},
  doi={10.1007/978-3-031-57202-3_12}
}
```

## 🧰 Notes

- All Python scripts are located in the `src/` subdirectory.
- This work makes use of the public `inspectnn` library to analyze internal CNN behaviors.

## 📜 License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

## 🙏 Acknowledgments

This work was carried out at the University of Napoli Federico II, with experiments performed in collaboration with the authors' research groups.
