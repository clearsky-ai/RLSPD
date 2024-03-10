# Reinforcement Learning-based Streaming Process Discovery under Concept Drift

## Setup

```bash
conda env create -f environment.yml
```

Due to different machine environments, especially CUDA versions, some dependencies (such as PyTorch and PM4PY) may not be installed with the above command. If the above command fails, please follow the instructions below to install the dependencies manually.

1. Create a python environment

    ```bash
    conda create -n RLSPD python=3.9.16
    conda activate RLSPD 
    ```

2. Install pytorch

    Following the official website's guidance (<https://pytorch.org/get-started/locally/>), install the corresponding PyTorch version based on your CUDA version. For our experiments, we use torch 1.12.1+cu113. The installation command is as follows:

    ```bash
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

3. Install pm4py

    Follow the official website's instructions(<https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.9/install.html>) to install pm4py and its related dependencies. For our experiments, we use pm4py 2.7.5. The installation command is as follows:

    ```bash
    pip install pm4py==2.7.5
    ```

4. Install other related dependencies

    ```bash
    pip install anytree==2.8.0
    ```

## Training and Testing

```bash
python main.py
```
