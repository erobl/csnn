# Cell Scoring Neural Networks

This repository contains the source code used for [CSNN](https://www.medrxiv.org/content/10.1101/2023.02.07.23285606v1).

Steps to run:

1. Download the [data](https://drive.google.com/drive/folders/1VcmDOdBbG46ILRd99TM2ZsZHBpcMazZ6?usp=sharing)

2. Extract CLL_288.zip and B-ALL_178.zip to data/raw/: You should have the following structure in your `data/` folder:

```
📦csnn
 ┗ 📂data
   ┗📂raw
    ┣📂B-ALL
    ┣📂CLL_24
    ┣📂CLL_102
    ┗📂CLL_162
```

3. Install the requirements: `pip install -r requirements.txt`

4. Run `preprocess_ball.py`

5. Run `preprocess_cll.py`

6. Run the experiments:

| Experiment                   | Dataset | Algorithm   | Command                                                                    |
|------------------------------|---------|-------------|----------------------------------------------------------------------------|
| Hyperparameter search        | B-ALL   | CellCNN     | `python train_cnn_cv.py config/tuning_ball_cellcnn.yaml`                   |
|                              |         | DeepCellCNN | `python train_cnn_cv.py config/tuning_ball_cnn.yaml`                       |
|                              |         | CSNN-Class  | `python train_logistic_cv.py config/tuning_ball_logistic.yaml`             |
|                              |         | CSNN-Reg    | `python train_reg_cv.py config/tuning_ball_reg.yaml`                       |
|                              | CLL     | CellCNN     | `python train_cnn.py config/tuning_cll_cellcnn.yaml`                       |
|                              |         | DeepCellCNN | `python train_cnn.py config/tuning_cll_cnn.yaml`                           |
|                              |         | CSNN-Class  | `python train_logistic.py config/tuning_cll_logistic.yaml`                 |
|                              |         | CSNN-Reg    | `python train_reg.py config/tuning_cll_reg.yaml`                           |
| Test set evaluation          | B-ALL   | CellCNN     | `python train_cnn.py config/best_ball_cellcnn.yaml`                        |
|                              |         | DeepCellCNN | `python train_cnn.py config/best_ball_cnn.yaml`                            |
|                              |         | CSNN-Class  | `python train_logistic.py config/best_ball_logistic.yaml`                  |
|                              |         | CSNN-Reg    | `python train_reg.py config/best_ball_reg.yaml`                            |
|                              | CLL     | CellCNN     | `python train_cnn.py config/best_cll_cellcnn.yaml`                         |
|                              |         | DeepCellCNN | `python train_cnn.py config/best_cll_cnn.yaml`                             |
|                              |         | CSNN-Class  | `python train_logistic.py config/best_cll_logistic.yaml`                   |
|                              |         | CSNN-Reg    | `python train_reg.py config/best_cll_reg.yaml`                             |
| No initialization ablation   | B-ALL   | CSNN-Class  | `python train_logistic_ablation_no_init.py config/best_ball_logistic.yaml` |
|                              |         | CSNN-Reg    | `python train_reg_ablation_no_init.py config/best_ball_reg.yaml`           |
|                              | CLL     | CSNN-Class  | `python train_logistic_ablation_no_init.py config/best_cll_logistic.yaml`  |
|                              |         | CSNN-Reg    | `python train_reg_ablation_no_init.py config/best_cll_reg.yaml`            |
| Initialization only ablation | B-ALL   | N/A         | `python train_ablation_init_only.py config/best_ball_reg.yaml`             |
|                              | CLL     | N/A         | `python train_ablation_init_only config/best_cll_reg.yaml`                 |