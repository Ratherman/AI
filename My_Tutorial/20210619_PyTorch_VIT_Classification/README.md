# Visual Transformer
* [[Paper Link]](https://arxiv.org/abs/2010.11929) An Image is Worth 16 x 16 Words: Transformers for Image Recognition at Scale.
* [[Github Link]]() Not yet decided

# Paper Note:
<details>
<summary> Abstract </summary>

1. Alexey Dosovitskiy (Google Research, Grain Team)
2. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place.
3. We show that the reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.
4. Pre-trained on large amounts of data first, and then transferred to small-size dataset.
</details>
<details>
<summary> Introduction </summary>

1. With the models and datasets growing, there is still no sign of saturating performance.
2. In large-scale image recognition, classic ResNet-like architectures are still state of the art.
3. We split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer.
4. Image patches are treated the same way as tokens (words) in an NLP application.
5. Only use ImageNet Dataset, the performance is a bit worse than ResNet because ... Transformers lack some of the "inductive biases inherent to CNNs", such as "translation equivariance" and "locality", and therefore do not generalize well. The situation changes if use larger datasets (14M - 300M images).
6. Datasets
    * ImageNet
    * ImageNet-Real
    * CIFAR-100
    * VTAB
</details>
<details>
<summary> Related Work </summary>

1. Transformers were for machine translation (2017), and have since become the state of the art method in many NLP tasks.
    * BERT (2019) uses a denoising self-supervised pre-training task.
    * GPT (2020) uses language modeling as its pre-training task.
2. Naive application of self-attention to images would require that each pixel attends to every other pixel: Quadratic cost.=
3. Model Overview: Use the image from [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/images/vit.gif)
    * <img src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/vit.gif" width="500">
    * <img src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/vit.png" width="500">

</details>
<details>
<summary> Method </summary>

1. Vision Transformer (ViT)
    * The standard Tranformer receives as input a 1D sequence of token embeddings.
    * Handel 2D images:
        1. Reshape the image from (H, W, C) into patch'ES' N x (P, P, C), where N = H x W / P^2
        2. Flatten the patchES and map to D dimensions with a trainable linear projection.
        3. Refer to the output of this projection as the patch embeddings.
    * "Class" Token
        1. Similar to BERT's Class token.
        2. Prof. Hung-yi Lee comes to rescue! (It's a 50 min video, but the first 15 min is enough for our understanding of CLS token.)
            * [機器學習2021】自督導式學習 (Self-supervised Learning) (二) – BERT簡介](https://www.youtube.com/watch?v=gh0hewYkjgo)
            * 2021/4/16
            * <img src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/self-supervised-learning.png" width="500">
            * <img src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/Bert-Review.png" width="500">
            * <img src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/Next-sentence-prediction.png" width="500">
    * Position embeddings are added to the patch embeddings to retain positional information.
    * Transformer Encoder:
        1. MSA: Multiheaded Self-Attention.
        2. MLP: Multi-Layer Perceptron.
        3. LN: Layernorm. (Before Every Block)
        4. Residual connections. (After Every Block)
    * Inductive Bias, Hybrid Architecture: These concepts exist in 4D world, but I lived happily in 3D world already, so. XD
2. Fine-Tuning And Higher Resolution

</details>
<details>
<summary> Experiments </summary>

</details>
<details>
<summary> Conclusion </summary>

</details>
<details>
<summary> Appendix: Multihead Self-attention </summary>

</details>
<details>
<summary> Appendix: Experiment Details </summary>

</details>
<details>
<summary> Appendix: Additional Results </summary>

</details>
<details>
<summary> Appendix: Additional Analyses </summary>

</details>

# Code Blocks & Explanations