# Take Note

# Reading Papers:
### Deep Learning for Single Image Super-Resolution：A Brief Review
* Deep Architectures For SISR
    <details>
    <summary> Section A: SRCNN </summary>

    * A three-layer CNN.
    * The filter size of each layer are [`64 x 1 x 9 x 9`, `32 x 64 x 5 x 5`, `1 x 32 x 5 x 5`]
    * The functions of these three nonlinear transformations are `(1) Patch extraction, (2) Nonlinear mapping, (3) Reconstuction`.
    * The loss function：mean square error (MSE).
    * We argue that its acclaim is owing to the CNN's strong capability of learning valid representations from big data in an end-to-end manner.
    </details>
    <details>
    <summary> Section B: SOTA Deep SISR Networks </summary>

    * Learning Effective Upsampling with CNN 
        * FSRCNN is the first work to use normal deconvolution layer to reconstruct HR images from LR feature maps.
        * ESPCN
            * Efficient Subpixel Convolution Layer
            * Rather than increasing resolution by explicitly enlarging feature maps as the deconvolution layer does, ESPCN extends the channels of the output features for storing the extra points to increase resolution and then rearranges these points to obtain the HR output through a specific mapping criterion.
        
    * The Deeper, The Better
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

    * Combining Properties of the SISR process with the Design of the CNN Frame
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
    <details>
    <summary> Section C: Comparisons among Different Models and Discussion. </summary>
    
    * PSNR/SSIM for measuring reconstruction quality
    * Number of parameters of NN for measuring storage efficiency (Params)
    * Number of composite multiply-accumulate operations for measuring computational efficiency (Mult&Adds)
    * MAYBE use ESPCN because the parameters is much smaller than others and its Mult&Adds is also smaller.
    * MAYBE use SRGAN because it's GAN-based.
    </details>
    <details>
    <summary> Section D: Optimization Objectives for DL-Based SISR </summary>

    * Benchmark of Optimization Objectives for DL-based SISR
        * MSE favors a high PSNR.
    * Objective Functions Based on non-Gaussian Additive Noises
    * Optimizing Forward KLD with Nonparametric Estimation
    
    * Section E: Characters of Different Objective Fucntions
    </details>
    <details>        
    <summary> Section F: Trends And Challenges </summary>
    
    * Lighter Deep Architectures for Efficient SISR.
    * More Effective DL Algorithms for Large-scale SISR and SISR with Unknow Corruption.
    * Theoretical Understanding of Deep Models for SISR.
    * More Rational Assessment Criteria for SISR in Different Applications.
    </details>
# SRCNN: 
* [Paper Link](https://arxiv.org/abs/1501.00092): Image Super-Resolution Using Deep Convolutional Networks
* [REF Github Link](https://github.com/yjn870/SRCNN-pytorch): SRCNN-pytorch
* [Google Colab Src Link](https://colab.research.google.com/drive/16nYeVokmDM_1cc_bi6z0Zu-MX9MRTbU3#scrollTo=GgR0KL5H2zv8): Google Colab SRC Link
# ESPCN:
* [Paper Link](https://arxiv.org/abs/1609.05158): Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
* [REF Github Link](https://github.com/Lornatang/ESPCN-PyTorch/blob/a3804d810e1416356c9e2b0bbb1619e39fa858d4/espcn_pytorch/model.py#L18): ESPCN-PyTorch
### Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
* Abstract
    * In this paper, we present the first convolutional neural network (CNN) capable of real-time SR of 1080p videos on a single K2 GPU.
    * We introduce an efficient sub-pixel convolution layer which learns an array of upscaling filters to upscale the final LR feature maps into the HR output.

1. Introduction
    <details>
    <summary> 1.0 Preface </summary>

    * The global SR problem assumes LR data to be a low-pass filtered (blurred), downsampled and nisy version of HR data.
    * It's a highly ill-posed problem, due to the loss of high-frequency information that occurs during the non-invertible low-pass filtering and subsampling operations.
    </details>
    <details>
    <summary> 1.1 Related Work </summary>

    * Sparse coding is an effective mechanism that assumes any natural image can be sparsely represented in a transform domain.
    * This transform domain is usually a dictionary of image atoms, which can be learnt through a training process that tries to discover the correspondence between LR and HR patches.
    </details>
    <details>
    <summary> 1.2 Motivations and contributions </summary>

    * To super-resolve a LR image into HR space, it is necessary to increase the resolution of the LR image to match that of the HR image at some point.
    * In this paper, contrary to previous works, we propose to increase the resolution from LR to HR only at the very end of the network and super-resolbe HR data from LR feature maps.
    </details>
    <details>
    <summary> 1.3 Advantage </summary>

    * For a network with L layers, we learn n_L-1 upscaling filters for the n_L-1 feature maps as opposed to one upscaling filter for the input image.
    * Thus, the network is capable of learning a better and more complex LR to HR mapping compared to a single fixed filter upscaling at the first layer.
    </details>

2. Method
    <details>
    <summary> 2.0 Preface </summary>

    * The downsampling operation is deterministic and known: to produce I^LR from I^HR, we first convolve I^HR using a Gaussian filter - then downsample the image by a factor of r.
    * In general, both I^LR and I^HR can have C color channels, thus they are represented as real-valued tensors of size H x W x C and rH x rW x C, respectively.
    * To solve the SISR problem, the SRCNN recovers from an upscaled and interpolated version of I^LR instead of I^LR.
    * In ESPCN, they avoid upscaling I^LR before feeding it into the network. Instead, they first apply a l layer convolutional neural network directly to the LR image, and then apply a sub-pixel convolution layer that upscaled the LR feature maps to produce I^SR.
    </details>
    <details>
    <summary> 2.1 Deconvolution layer </summary>
    
    * The addition of a deconvolution layer is a popular choice for recovering resolution from max-pooling adn other image down-sample layers.
    * It's trival to show that the bicubic interpolation used in SDRCNN is a psecial case of the decovolution layer.
    </details>
    <details>
    <summary> 2.2 Efficient sub-pixel convolution layer </summary>

    * PS is an periodic shuffling operator that rearranges the elements of a H x W x C * r^2 tensor to a tensor of shape rH x rW x C.
    * Given a training set consisting of HR image examples, we generatethe corresponding LR images, and calculate the pixel-wise mean squared error (MSE) of the reconstructuion as an objective function to train the network.
    </details>
    <details>

3. Experiments
    <details>
    <summary> 3.1 Datasets </summary>
    
    * For their final models, they use 50,000 randomly selected images from ImageNet for the Training.
    * We only consider the luminance channel in YCbCr colour space in this section because humans are more sensitive to luminance changes.
    </details>
    <details>
    <summary> 3.2 Implementation details </summary>

    * The training stops after no improvement of the cost function is observed after 100 epochs.
    * Initial learning rate is set to 0.01 and final learning rate is set to 0.0001 and updated gradually when the improvement of the cost function is smaller than a threshold miu.The final layer learns 10 times slower.
    </details>
    <details>
    <summary> 3.3 Image super-resolution results </summary>
    
    * It is noticeable that despite each filter is independent in LR space, out independent filters is actually smooth in the HR space after PS.
    * Compared to SRCNN's last layer filters, their final layer filters has complex patterns for different feature maps, it also has much richer and more meaningful representations.
    * Result suggest that tanh function performs better for SISR compared to relu.
    </details>
    <details>
    <summary> 3.4 Run time evaluations </summary>

    * The mean PSNR (dB) of different methods. Benchmarks: Set5, Set14, BSD300, BSD500
    </details>

4. Conclusion
    * We propose a novel sub-pixel convolution layer which is capable of super-resolving LR data into HR space with very little additional computational cost compared to a deconvolution layer at training time.
# SRGAN:
* [Paper Link](https://arxiv.org/abs/1609.04802):Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
* [REF Github Link](https://github.com/Lornatang/SRGAN-PyTorch)