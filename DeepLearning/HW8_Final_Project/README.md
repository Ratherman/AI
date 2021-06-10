# Take Note

# Reading Papers:
## Deep Learning for Single Image Super-Resolution：A Brief Review
* Deep Architectures For SISR
    <details>
    <summary> SRCNN </summary>
    
    * A three-layer CNN.
    * The filter size of each layer are [`64 x 1 x 9 x 9`, `32 x 64 x 5 x 5`, `1 x 32 x 5 x 5`]
    * The functions of these three nonlinear transformations are `(1) Patch extraction, (2) Nonlinear mapping, (3) Reconstuction`.
    * The loss function：mean square error (MSE).
    * We argue that its acclaim is owing to the CNN's strong capability of learning valid representations from big data in an end-to-end manner.
    </details>

    * SOTA Deep SISR Networks
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
