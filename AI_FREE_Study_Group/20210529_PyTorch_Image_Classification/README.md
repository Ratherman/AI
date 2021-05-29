# AI FREE 周末讀書會: PyTorch/ Image Classification 
> A step-by-step workthrough tutorial

> Eric

> 2021/05/29

# Specify Env:
- Python 3.7.10
- PyTorch 1.7.1 (cpu version) (gpu version works actually well)

# Outline:
- Step 01: Installation
- Step 02: Get Dataset
- Step 03: Arrange File Structure
- Step 04: Have a look of image
- Step 05: Define Custom Dataset Class
- Step 06: Define DataLoader w.r.t Dataset Class
- Step 07: Define LeNet5 - like Structure
- Step 08: Define Loss Function
- Step 09: Training Phase
- Step 10: Plot Accuracy & Loss Curves
- Step 11: Testing Phase

# Start:

- Step 01: Installation
    - Anaconda:
    - Conda Create Env
        - 示範使用 CPU，請使用這個，Python 3.7 和 PyTorch 1.7.X 是好朋友 ^_^
        - 確認新環境建置完成
        - 啟動新環境

        ```bash
        conda create --name "PT_CPU_1.7.1_DEMO" python=3.7
        ```

        ```python
        conda env list
        ```

        ```python
        conda activate PT_CPU_1.7.1_DEMO
        ```

    - PyTorch: 1.7.X, and other useful packages (I spent 10 mins on this)
        - 核心: 安裝 pytorch 1.7.1 CPU Version
        - Other Useful Packages
            - 1. GUI 介面，方便寫程式 → jupyter notebook
            - 2. 用來讀 Metadata.csv → pandas
            - 3. 知道 loop 的進度 → tqdm
            - 4. 讀影像，其他選擇如 PIL，總之這邊使用 OpenCV → opencv
            - 5. 畫圖 → matplotlib
            - 6. 矩陣運算 → numpy

        ```python
        conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
        ```

        ```python
        conda install jupyter notebook
        ```

        ```python
        conda install pandas
        ```

        ```python
        conda install tqdm
        ```

        ```python
        conda install -c conda-forge opencv
        ```

        ```python
        conda install matplotlib
        ```

        ```python
        conda install numpy
        ```

    - 示範上考慮到大家不一定都有顯卡，所以使用 CPU，若想要裝 higher version (e.g. 1.8.X) or GPU Version，請參考以下網址。
    - PyTorch Installation Link: [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)
- Step 02: Get Dataset

    ### Temp: Download From Google Cloud

    - AIdea AOI Detection: [https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4)

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/80b68975-ddc8-4dd2-852c-a68eff14faf4/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/80b68975-ddc8-4dd2-852c-a68eff14faf4/Untitled.png)

    - Download Page:

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/932feece-6f53-4fc4-8803-d2a73eff44e8/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/932feece-6f53-4fc4-8803-d2a73eff44e8/Untitled.png)

- Step 03: Arrange File Structure
    - 預期的專案結構如下:
        - . / test_images/
        . / train_images/
        . / test.csv
        . / train.csv
        . / [LeNet5.py](http://lenet5.py) ← 這個 Python File 是要自己新建的

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9116a3d5-e8a4-4897-b785-d368048db4a3/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9116a3d5-e8a4-4897-b785-d368048db4a3/Untitled.png)

    - train_images

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/49828915-67be-4170-9db0-7ec9283b9f49/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/49828915-67be-4170-9db0-7ec9283b9f49/Untitled.png)

    - train.csv

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/32de0040-2e5e-4e0e-8f60-b8a564054fe0/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/32de0040-2e5e-4e0e-8f60-b8a564054fe0/Untitled.png)

---

- 在 PyTorch，Train 的時候提供 Batch 需要使用 DataLoader，而要使用 DataLoader 前，會需要先完成自定義的 Dataset Class。
- 下方正式進入程式碼，請大家移動 cd 至專案路徑當中，並開啟 jupyter notebook

```bash
jupyter notebook
```

然後點選: LeNet5.py

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b82f1a42-c02d-4678-9ba1-2d9fae0cf8cc/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b82f1a42-c02d-4678-9ba1-2d9fae0cf8cc/Untitled.png)