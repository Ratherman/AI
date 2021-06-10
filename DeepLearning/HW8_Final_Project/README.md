# Take Note

# Reading Papers:
## Deep Learning for Single Image Super-Resolution：A Brief Review
* Deep Architectures For SISR
    * Section A: SRCNN
        * A three-layer CNN.
        * The filter size of each layer are [`64 x 1 x 9 x 9`, `32 x 64 x 5 x 5`, `1 x 32 x 5 x 5`]
        * The functions of these three nonlinear transformations are `(1) Patch extraction, (2) Nonlinear mapping, (3) Reconstuction`.
        * The loss function：mean square error (MSE).
        * We argue that its acclaim is owing to the CNN's strong capability of learning valid representations from big data in an end-to-end manner.
    * Section B: SOTA Deep SISR Networks
        <details>
        <summary> Learning Effective Upsampling with CNN </summary>
        
        * FSRCNN is the first work to use normal deconvolution layer to reconstruct HR images from LR feature maps.
        * ESPCN
            * Efficient Subpixel Convolution Layer
            * Rather than increasing resolution by explicitly enlarging feature maps as the deconvolution layer does, ESPCN extends the channels of the output features for storing the extra points to increase resolution and then rearranges these points to obtain the HR output through a specific mapping criterion.
        </details>
        
        <details>
        <summary> The Deeper, The Better </summary>

        * VDSR is the first very deep model used in SISR.
            * The second contribution is the residual learning.
        * DRCN
            * To overcome the difficulties of training a deep recursive CNN, a multisupervised strategy is applied, and the result can be regarded as the fusion of 16 intermediate results.
        * SRResNet
            * It composed of 16 residual units (a residual unit consists of two nonlinear convolutions with residual learning.)
        * DRRN
        * DRCN
            * Then, to accommodate parameter reduction, each block shares the same parameters and is reused recursively.
        * EDSR
            * Remove the usage of BN.
            * Increases the number of output features of each layer on a large scale.
        * MDSR
            * Achieve the multiscale architecture.
        * SRDenseNet
        </details>

        <details>
        <summary> Combining Properties of the SISR process with the Design of the CNN Frame </summary>

        * Combining sparse coding with deep NN
        * Learning to ensemble by NN
        * Deep architectures with progressive methodology
            * DEGREE
            * LapSRN: Generate SR of different scales progressively.
            * PixelSR: 
                * Leverage conditional autoregresive models to generate SR pixel-by-pixel.
                * First applies conditional PixelCNN to SISR.
        * Deep architectures with backprojection
        * Usage of additional information from LR
            * DEGREE: takes the edge map of LR as another input.
            * SFT-GAN: extra semantic information of LR for better perceptual quality.
        * Reconstruction-based frameworks based on priors offered by deep NN
            * IRCNN: they first trained a series of CNN-based denoisers with different noise levels and took backprojection as the reconstruction part.
        * Deep architectures with internal examples
            * ZSSR: 
                * the first literature combining deep architectures with interal-example learning.
                * However, this approach will increase runtime immensely.

        </details>

    * Section C: Comparisons among Different Models and Discussion.
        * PSNR/SSIM for measuring reconstruction quality
        * Number of parameters of NN for measuring storage efficiency (Params)
        * Number of composite multiply-accumulate operations for measuring computational efficiency (Mult&Adds)
        * MAYBE use ESPCN because the parameters is much smaller than others and its Mult&Adds is also smaller.
        * MAYBE use SRGAN because it's GAN-based.

    * Section D: Optimization Objectives for DL-Based SISR
        * Benchmark of Optimization Objectives for DL-based SISR
            * MSE favors a high PSNR.
        * Objective Functions Based on non-Gaussian Additive Noises
        * Optimizing Forward KLD with Nonparametric Estimation
    
    * Section E: Characters of Different Objective Fucntions
        
    * Section F: Trends And Challenges
        * Lighter Deep Architectures for Efficient SISR.
        * More Effective DL Algorithms for Large-scale SISR and SISR with Unknow Corruption.
        * Theoretical Understanding of Deep Models for SISR.
        * More Rational Assessment Criteria for SISR in Different Applications.

## SRCNN: 
* [Paper Link](https://arxiv.org/abs/1501.00092): Image Super-Resolution Using Deep Convolutional Networks
* [Github Link](https://github.com/yjn870/SRCNN-pytorch): SRCNN-pytorch


## ESPCN:
* [Paper Link](https://arxiv.org/abs/1609.05158): Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
* [Github Link](https://github.com/Lornatang/ESPCN-PyTorch/blob/a3804d810e1416356c9e2b0bbb1619e39fa858d4/espcn_pytorch/model.py#L18): ESPCN-PyTorch

## SRGAN:
* [Paper Link](https://arxiv.org/abs/1609.04802):Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
* [Github Link](https://github.com/Lornatang/SRGAN-PyTorch)