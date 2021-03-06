# Vision Transformer
* [[Paper Link]](https://arxiv.org/abs/2010.11929) An Image is Worth 16 x 16 Words: Transformers for Image Recognition at Scale.
* [[PyTorch Tutorial Link]](https://pytorch.org/tutorials/beginner/vt_tutorial.html) 
* [[Ref Code Link 01]](https://github.com/jeonsworld/ViT-pytorch) jeonsworld/ViT-pytorch
* [[Ref Code Link 02]](https://github.com/google-research/vision_transformer/tree/master/vit_jax) google-research/vision_transformer
* [[Ref Cpde Link 03]](https://github.com/lukemelas/PyTorch-Pretrained-ViT) lukemelas/PyTorch-Pretrained-ViT
* [[Ref Code Link 04]](https://github.com/asyml/vision-transformer-pytorch) asyml/vision-transformer-pytorch
* [[Ref Code Link 05 => Code]](https://github.com/jacobgil/vit-explain) jacobgil/vit-explain
* [[Ref Code Link 05 => XAI]](https://jacobgil.github.io/deeplearning/vision-transformer-explainability) Exploring Explainability for Vision Transformers
* [[Ref Code Link 06 => YouTube]](https://www.youtube.com/watch?v=ovB0ddFtzzA) Vision Transformer in PyTorch
* [[Ref Code Link 06 => Code]](https://github.com/rwightman/pytorch-image-models) rwightman/pytorch-image-models
* [[Ref Code link 06 => YouTuber]](https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/vision_transformer) jankrepl/mildlyoverfitted
# Paper Review: 
<details>
<summary> Abstract </summary>

1. Alexey Dosovitskiy (Google Research, Brain Team)
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
    * Pre-trained Datasets
        * ILSVRC-2012 ImageNet Dataset (1K classes, 1.3M images)
        * Superset ImageNet-21k Dataset (21K classes, 14M images)
        * JFT (18K classes, 303M images)
    * Transfer to ...
        * ImageNet-Real
        * CIFAR10/100
        * Oxford-IIIT Pets
        * Oxford Flowers-102
</details>
<details>
<summary> Related Work </summary>

1. Transformers were for machine translation (2017), and have since become the state of the art method in many NLP tasks.
    * BERT (2019) uses a denoising self-supervised pre-training task.
    * GPT (2020) uses language modeling as its pre-training task.
2. Naive application of self-attention to images would require that each pixel attends to every other pixel: Quadratic cost.
3. Model Overview: Use the image from [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/images/vit.gif)
    <p align="center">
      <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/vit.gif">
    </p>
    <p align="center">
      <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/vit.png">
    </p>
</details>
<details>
<summary> Method </summary>

1. Vision Transformer (ViT)
    * The standard Tranformer receives as input a 1D sequence of token embeddings.
    * Handel 2D images:
        1. Reshape the image from (H, W, C) into patch'ES' N x (P, P, C), where N = H x W / P^2
        2. Flatten the patchES and map to D dimensions with a trainable linear projection.
        3. Refer to the output of this projection as the patch embeddings.
    * "Class" Token: `CLS`
        1. Similar to BERT's Class token.
        2. Prof. Hung-yi Lee comes to rescue! (It's a 50 min video, but the first 15 min is enough for our understanding of CLS token.)
            * [???????????????2021????????????????????? (Self-supervised Learning) (???) ??? BERT??????](https://www.youtube.com/watch?v=gh0hewYkjgo)
            * 2021/4/16
            <p align = "center">
              <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/self-supervised-learning.png">
            </p>
            <p align = "center">
              <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/Bert-Review.png">
            </p>
            <p align = "center">
              <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/Next-sentence-prediction.png">
            </p>
    * Position embeddings are added to the patch embeddings to retain positional information.
    * Transformer Encoder:
        1. MSA: Multiheaded Self-Attention.
        2. MLP: Multi-Layer Perceptron.
        3. LN: Layernorm. (Before Every Block)
        4. Residual connections. (After Every Block)
        <p align = "center">
          <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/Functions.png">
        </p>
    * Inductive Bias, Hybrid Architecture: These concepts exist in 4D world, but I lived happily in 3D world already, so. XD
2. Fine-Tuning And Higher Resolution
    * Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks.
    * For this, we remove the pre-trained prediction head and attach a zero-initialized D x K feedforward layer, where K is the number of downstream classes.
    * It's always beneficial to fine-tune at higher resolution than pre-training.
    * When feeding images of higher resolution, we keep the patch size the same, which results in a larger effective sequence length.
    * The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints).
    * To use pre-trained position embeddings, they perform 2D interpolation.
</details>
<details>
<summary> Experiments </summary>

1. Setup
    * Datasets
    * Model Variants: Vit-Base, Vit-Large, Vit-Huge
    * Baseline: ResNet
    * Training & Fine-tuning:
        * Use Adam with Beta1=0.9, Beta2 = 0.999, a batch size of 4096 and apply a high weight decay of 0.1, which we found to be useful for transfer of all models.
        * Use a linear learing rate and decay.
2. Comparison to SOTA
    <p align = "center">
      <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/compare_sota.png">
    </p>
3. Pre-Training Data Requirements
    * How crucial is the dataset size?
    * Pre-train ViT models on datasets of increaseing size: ImageNet, ImageNet-21K, and JFT-300M.
    <p align = "center">
      <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/BiT-ViT.png">
    </p>
4. Scaling Study
    * Vision Transformers generally outperform ResNets with the same computational budget.
    * Hybrids improve upon pure Transformers for smaller model sizes, but the gap vanishes for larger models.
    <p align = "center">
      <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/Flops.png">
    </p>
5. Inspecting Vision Transformer
    * To begin understand how the Vision Transformer processes image data, we analyze its internal representations.
    * The first layer of the Vision Transformer linearly projects the flattened patches into lower-dimensional space. 
        * Fig. `left` shows the top principal components of the learned embedding filters.
    * After the projection, a learned position embedding is added to the patch representations.
        * Fig. `center` shows the model learns to encode distance within the image in the similarity of position embeddings.
        * Closer patches tend to have more similar position embeddings.
        * Patches in the same row/column have similar embeddings.
        <p align = "center">
          <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/fig7.png">
        </p>

        * What is Positional Encoding?
            * Prof. Hung-yi Lee comes to rescue! (It's a 45 min video, only 5 mins between 20:00 - 25:00 should be enough to have a feel of positional encoding.)
            * [???????????????2021????????????????????? (Self-attention) (???)](https://www.youtube.com/watch?v=gmsMY5kc-zw)
                <p align = "center">
                  <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/positional-encoding.png">
                </p>
            * Each position has a unique positional vector e^i.
            * hand-crafted
            * learned from data
    * We compute the average distance in image space across which information is integrated, based on the attention weights.
        * This "attention distance" is analogous to receptive field size in CNNs.
        * Globally, we find that the model attends to image regions that are semantically relevant for classification.
        <p align = "center">
          <img width="750" src="https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210619_PyTorch_VIT_Classification/imgs/fig6.png">
        </p>
    
6. Self-Supervision
    * We also perform a preliminary exploration on `masked patch prediction` for self-supervision, mimicking the masked language modeling task used in BERT.
    * With self-supervised pre-training, our smaller ViT-B/16 model achieves 79.9% accuracy on ImageNet, a significant improvement of 2% to training from scratch, but still 4% behind supervised pre-training.
</details>
<details>
<summary> Conclusion </summary>

* We do not introduce image-specific inductive biases into the architecture apart from the initial patch extraction step.
* We interpret an image as a sequence of patches and process it by a standard Transformer encoder as used in NLP.
* Challenges:
    * Apply ViT to other computer vision tasks, such as detection and segmentation.
    * Another challenge is to continue exploring self-supervised pre-training methods.
</details>
<hr>
Let's leave it there. ^_^ By "it", I mean Appendix.
<hr>
<details>
<summary> Appendix </summary>

1. Multihead Self-attention
2. Experiment Details
3. Additional Results
4. Additional Analyses
</details>

# Code Blocks & Explanations
* There are 5 modules I'll cover in the following contents.
* From small module to large module.
* Besides knowing how to make vision transformer work, there will be also one more section/ module to introduce XAI.

## Module 1: Patch Embedding Module

## Module 2: Attention Module

## Module 3: Multilayer Perceptron

## Module 4: Block Module

## Module 5: Vision Transformer