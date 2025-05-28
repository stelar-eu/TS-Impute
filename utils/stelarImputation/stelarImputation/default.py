__all__ = ['default_imputation_params', 'default_hyperparameter_tuning_imputation_params']


def default_imputation_params() -> dict:
    """
    Provides default parameters for various imputation algorithms.

    :return: A dictionary containing default parameters for each imputation algorithm.
    """
    default_params = {
        "iterativesvd": {
            "rank": 3  # initial rank estimation for low-rank matrix approximation
        },
        "rosl": {
            "rank": 3,  # initial rank estimation
            "reg": 0.05  # Regularization parameter on l1-norm controlling sparsity
        },
        "cdmissingvaluerecovery": {
            "truncation": 2,  # Number of components to retain in decomposition
            "eps": 0.000001  # Convergence threshold for iterative updates
        },
        "ogdimpute": {
            "truncation": 3  # Number of components to retain in decomposition
        },
        "nmfmissingvaluerecovery": {
            "truncation": 3  # Number of components to retain in decomposition
        },
        "dynammo": {
            "H": 5,  # Number of hidden states in the model
            "max_iter": 100,  # Maximum number of iterations for convergence
            "fast": True  # Flag to use a faster approximation method - use woodbury lemma for inverse of the matrix
        },
        "grouse": {
            "max_rank": 3  # Maximum rank for the subspace estimation
        },
        "softimpute": {
            "max_rank": 3  # Maximum rank for the low-rank approximation
        },
        "pca_mme": {
            "truncation": 3,  # Number of principal components to retain
            "single_block": False  # Flag to process data in a single block
        },
        "svt": {
            "tauScale": 0.7  # Scaling factor for the threshold in singular value thresholding
        },
        "spirit": {
            "k0": 1,  # The number of hidden variables
            "w": 6,  # The order of the autoregressive models
            "lambda": 1  # The exponential forgetting factor
        },
        "tkcm": {
            "truncation": 1  # Number of components to retain in decomposition
        },
        "linearimpute": {
            # No parameters required for this method
        },
        "meanimpute": {
            # No parameters required for this method
        },
        "zeroimpute": {
            # No parameters required for this method
        },
        "naive": {
            # No parameters required for this method
        },
        "saits": {
            "n_layers": 4,  # Number of layers in the 1st and 2nd DMSA blocks in the SAITS model
            "d_model": 64,  # Dimension of the model’s backbone; input dimension of the multi-head DMSA layers
            "d_ffn": 64,  # Dimension of the feed-forward network within each Transformer layer
            "n_heads": 2,  # Number of attention heads in the multi-head attention mechanism
            "d_k": 32,  # Dimension of the key vectors in the attention mechanism
            "d_v": 32,  # Dimension of the value vectors in the attention mechanism
            "dropout": 0.1,  # Dropout rate applied to prevent overfitting
            "epochs": 100,  # Number of training epochs
            "batch_size": 32,  # Number of samples per batch during training
            "ORT_weight": 1,  # Weight for the ORT (Observed Reconstruction Task) loss component
            "MIT_weight": 1,  # Weight for the MIT (Missing Imputation Task) loss component
            "attn_dropout": 0,  # Dropout rate specifically for the attention mechanism
            "diagonal_attention_mask": True,
            # Whether to apply a diagonal mask in self-attention to prevent a position from attending to itself
            "num_workers": 0,
            # Number of subprocesses for data loading; 0 means data loading is done in the main process
            "patience": 10,  # Number of epochs with no improvement after which training will be stopped early
            "lr": 0.008610459441905952,  # Learning rate for the optimizer
            "weight_decay": 1e-5,  # Weight decay (L2 penalty) for regularization
            "device": None  # Device to run the model on; None defaults to GPU and if there is no GPU then to CPU
        },
        "brits": {
            "rnn_hidden_size": 128,  # Hidden size of the RNN
            "batch_size": 32,
            "epochs": 100,
            "num_workers": 0,
            "patience": 10,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "device": None
        },
        "csdi": {
            "n_layers": 4,  # Number of Transformer layers
            "n_channels": 32,  # Channels in the neural network
            "n_heads": 4,
            "d_time_embedding": 64,  # Dimension of time embedding
            "d_feature_embedding": 3,  # Dimension of feature embedding
            "d_diffusion_embedding": 64,  # Dimension for diffusion embedding
            "is_unconditional": False,  # Whether the model is unconditional
            "beta_start": 0.0001,  # Start value for beta schedule
            "beta_end": 0.1,  # End value for beta schedule
            "epochs": 100,
            "batch_size": 32,
            "n_diffusion_steps": 50,  # Number of diffusion steps
            "num_workers": 0,
            "patience": 10,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "device": None
        },
        "usgan": {
            "rnn_hidden_size": 128,
            "lambda_mse": 1,  # Loss weight for MSE component
            "hint_rate": 0.7,  # Hint rate in GAN training
            "dropout": 0.1,
            "G_steps": 1,  # Generator update steps per iteration
            "D_steps": 1,  # Discriminator update steps per iteration
            "epochs": 100,
            "batch_size": 32,
            "num_workers": 0,
            "patience": 10,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "device": None
        },
        "mrnn": {
            "rnn_hidden_size": 64,
            "batch_size": 32,
            "epochs": 100,
            "num_workers": 0,
            "patience": 10,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "device": None
        },
        "gpvae": {
            "latent_size": 8,  # Latent variable size
            "encoder_sizes": [64, 64],  # Layer sizes of the encoder
            "decoder_sizes": [64, 64],  # Layer sizes of the decoder
            "kernel": "cauchy",  # Kernel type for Gaussian process e.g. 'cauchy', 'diffusion', 'rbf' or 'matern'
            "beta": 0.2,  # KL-divergence weight
            "M": 1,  # Number of Monte Carlo samples
            "K": 1,  # Number of GP samples
            "sigma": 1.005,  # GP noise parameter
            "length_scale": 7.0,  # Length scale in GP kernel
            "kernel_scales": 1,  # Scale factor for kernel
            "window_size": 24,  # Size of the input window
            "batch_size": 32,
            "epochs": 100,
            "num_workers": 0,
            "patience": 10,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "device": None
        },
        "timesnet": {
            "n_layers": 2,
            "top_k": 3,  # Top-K attention selection
            "d_model": 64,
            "d_ffn": 64,
            "n_kernels": 1,  # Number of convolution kernels
            "dropout": 0.05,
            "apply_nonstationary_norm": False,  # Whether to normalize for non-stationarity
            "batch_size": 32,
            "epochs": 100,
            "num_workers": 0,
            "patience": 10,
            "lr": 0.009978373271498147,
            "weight_decay": 1e-5,
            "device": None
        },
        "nonstationary_transformer": {
            "n_layers": 2,
            "d_model": 32,
            "d_ffn": 64,
            "n_heads": 4,
            "n_projector_hidden_layers": 2,  # Hidden layers in the output projector
            "d_projector_hidden": [32, 32],  # Dimensions of hidden layers
            "dropout": 0.2,
            "epochs": 100,
            "batch_size": 32,
            "ORT_weight": 1,
            "MIT_weight": 1,
            "num_workers": 0,
            "patience": 10,
            "lr": 0.007037859132857435,
            "weight_decay": 1e-5,
            "device": None
        },
        "autoformer": {
            "n_layers": 2,
            "d_model": 64,
            "d_ffn": 64,
            "n_heads": 2,
            "factor": 3,  # Series decomposition factor
            "moving_avg_window_size": 25,  # Window size for moving average
            "dropout": 0.1,
            "epochs": 100,
            "batch_size": 32,
            "ORT_weight": 1,
            "MIT_weight": 1,
            "num_workers": 0,
            "patience": 10,
            "lr": 0.0009696220076141339,
            "weight_decay": 1e-5,
            "device": None
        }
    }

    return default_params


def default_hyperparameter_tuning_imputation_params() -> dict:
    """
    Provides default hyperparameter tuning parameters for various imputation algorithms.

    :return: A dictionary containing hyperparameter tuning parameters for each imputation algorithm.
    """
    default_params: dict = {
        "cdmissingvaluerecovery": {
            "truncation": [
                1,
                2,
                3,
                4,
                5
            ],
            "eps": [
                0.00005,
                0.01
            ]
        },
        "pca_mme": {
            "truncation": [
                1,
                2,
                3,
                4,
                5,
                10,
                20,
                50
            ],
            "single_block": [
                False,
                True
            ]
        },
        "softimpute": {
            "max_rank": [
                1,
                2,
                3,
                4
            ]
        },
        "grouse": {
            "max_rank": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10
            ]
        },
        "iterativesvd": {
            "rank": [
                1,
                2,
                3,
                4,
                5
            ]
        },
        "rosl": {
            "rank": [
                1,
                2,
                3,
                4,
                5
            ],
            "reg": [
                0.001,
                0.1
            ]

        },
        "dynammo": {
            "H": [
                1,
                25
            ],
            "max_iter": [
                100
            ],
            "fast": [
                False,
                True
            ]
        },
        "ogdimpute": {
            "truncation": [
                1,
                2,
                3,
                4,
                5
            ]
        },
        "nmfmissingvaluerecovery": {
            "truncation": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10
            ]
        },
        "svt": {
            "tauScale": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9
            ]
        },
        "tkcm": {
            "truncation": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10
            ]
        },
        "spirit": {
            "k0": [
                1,
                2,
                3,
                4,
                5
            ],
            "w": [
                1,
                2,
                3,
                4,
                5,
                6
            ],
            "lambda": [
                0.00005,
                1
            ]
        },
        "saits": {
            "n_layers": [
                1,
                2,
                3
            ],
            "d_model": [
                8,
                16,
                32,
                64,
                128
            ],
            "d_ffn": [
                8,
                16,
                32,
                64,
                128
            ],
            "n_heads": [
                1,
                2,
                3
            ],
            "d_k": [
                8,
                16,
                32,
                64
            ],
            "d_v": [
                8,
                16,
                32,
                64
            ],
            "dropout": [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5
            ],
            "epochs": [
                100
            ],
            "batch_size": [
                32
            ],
            "ORT_weight": [
                1
            ],
            "MIT_weight": [
                1
            ],
            "attn_dropout": [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5
            ],
            "diagonal_attention_mask": [
                True,
                False
            ],
            "num_workers": [
                0
            ],
            "patience": [
                10
            ],
            "lr": [
                0.00005,
                0.01
            ],
            "weight_decay": [
                1e-6,
                1e-3
            ],
            "device": None
        },
        "timesnet": {
            "n_layers": [
                1,
                2,
                3
            ],
            "d_model": [
                8,
                16,
                32,
                64,
                128
            ],
            "d_ffn": [
                8,
                16,
                32,
                64,
                128
            ],
            "n_heads": [
                1,
                2,
                3
            ],
            "top_k": [
                1,
                2,
                3
            ],
            "n_kernels": [
                3,
                4,
                5
            ],
            "dropout": [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5
            ],
            "epochs": [
                100
            ],
            "batch_size": [
                32
            ],
            "attn_dropout": [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5
            ],
            "apply_nonstationary_norm": [
                True,
                False
            ],
            "num_workers": [
                0
            ],
            "patience": [
                10
            ],
            "lr": [
                0.00005,
                0.01
            ],
            "weight_decay": [
                1e-6,
                1e-3
            ],
            "device": None
        },
        "nonstationary_transformer": {
            "n_layers": [
                1,
                2,
                3
            ],
            "d_model": [
                8,
                16,
                32,
                64,
                128
            ],
            "d_ffn": [
                8,
                16,
                32,
                64,
                128
            ],
            "n_heads": [
                1,
                2,
                3
            ],
            "n_projector_hidden_layers": [
                2
            ],
            "d_projector_hidden": [
                [
                    16,
                    16
                ],
                [
                    32,
                    32
                ],
                [
                    64,
                    64
                ]
            ],
            "dropout": [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5
            ],
            "epochs": [
                100
            ],
            "batch_size": [
                32
            ],
            "ORT_weight": [
                1
            ],
            "MIT_weight": [
                1
            ],
            "num_workers": [
                0
            ],
            "patience": [
                10
            ],
            "lr": [
                0.00005,
                0.01
            ],
            "weight_decay": [
                1e-6,
                1e-3
            ],
            "device": None
        },
        "autoformer": {
            "n_layers": [
                1,
                2,
                3
            ],
            "d_model": [
                8,
                16,
                32,
                64,
                128
            ],
            "d_ffn": [
                8,
                16,
                32,
                64,
                128
            ],
            "n_heads": [
                1,
                2,
                3
            ],
            "factor": [
                3
            ],
            "moving_avg_window_size": [
                5,
                13,
                25
            ],
            "dropout": [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5
            ],
            "epochs": [
                100
            ],
            "batch_size": [
                32
            ],
            "ORT_weight": [
                1
            ],
            "MIT_weight": [
                1
            ],
            "num_workers": [
                0
            ],
            "patience": [
                10
            ],
            "lr": [
                0.00005,
                0.01
            ],
            "weight_decay": [
                1e-6,
                1e-3
            ],
            "device": None
        },
        "brits": {
            "rnn_hidden_size": [
                8,
                16,
                32,
                64,
                128
            ],
            "epochs": [
                100
            ],
            "batch_size": [
                32
            ],
            "num_workers": [
                0
            ],
            "patience": [
                10
            ],
            "lr": [
                0.00005,
                0.01
            ],
            "weight_decay": [
                1e-6,
                1e-3
            ],
            "device": None
        },
        "csdi": {
            "n_layers": [
                1,
                2,
                3,
                4
            ],
            "n_channels": [
                8,
                16,
                32,
                64
            ],
            "n_heads": [
                1,
                2,
                3,
                4
            ],
            "d_time_embedding": [
                8,
                16,
                32,
                64
            ],
            "d_feature_embedding": [
                1,
                2,
                3,
                4
            ],
            "d_diffusion_embedding": [
                8,
                16,
                32,
                64
            ],
            "is_unconditional": [
                False,
                True
            ],
            "beta_start": [
                0.00001,
                0.0001,
                0.001,
            ],
            "beta_end": [
                0.05,
                0.1,
                0.2,
                0.3
            ],
            "n_diffusion_steps": [
                10,
                20,
                30,
                40,
                50
            ],
            "epochs": [
                100
            ],
            "batch_size": [
                32
            ],
            "num_workers": [
                0
            ],
            "patience": [
                10
            ],
            "lr": [
                0.00005,
                0.01
            ],
            "weight_decay": [
                1e-6,
                1e-3
            ],
            "device": None
        },
        "usgan": {
            "rnn_hidden_size": [
                8,
                16,
                32,
                64,
                128
            ],
            "lambda_mse": [
                0.1,
                1
            ],
            "hint_rate": [
                0.1,
                0.9
            ],
            "G_steps": [
                1
            ],
            "D_steps": [
                1
            ],
            "dropout": [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5
            ],
            "epochs": [
                100
            ],
            "batch_size": [
                32
            ],
            "num_workers": [
                0
            ],
            "patience": [
                10
            ],
            "lr": [
                0.00005,
                0.01
            ],
            "weight_decay": [
                1e-6,
                1e-3
            ],
            "device": None
        },
        "mrnn": {
            "rnn_hidden_size": [
                8,
                16,
                32,
                64,
                128
            ],
            "epochs": [
                100
            ],
            "batch_size": [
                32
            ],
            "num_workers": [
                0
            ],
            "patience": [
                10
            ],
            "lr": [
                0.00005,
                0.01
            ],
            "weight_decay": [
                1e-6,
                1e-3
            ],
            "device": None
        },
        "gpvae": {
            "latent_size": [
                1,
                4,
                8,
                16
            ],
            "encoder_sizes": [
                [
                    16,
                    16
                ],
                [
                    32,
                    32
                ],
                [
                    64,
                    64
                ]
            ],
            "decoder_sizes": [
                [
                    16,
                    16
                ],
                [
                    32,
                    32
                ],
                [
                    64,
                    64
                ]
            ],
            "kernel": [
                'cauchy'
            ],
            "beta": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5
            ],
            "M": [
                1,
                2
            ],
            "K": [
                1,
                2
            ],
            "sigma": [
                1,
                1.005,
                1.1,
                1.2
            ],
            "length_scale": [
                1,
                3,
                5,
                7,
                9
            ],
            "kernel_scales": [
                1
            ],
            "window_size": [
                3,
                9,
                12,
                24
            ],
            "epochs": [
                100
            ],
            "batch_size": [
                32
            ],
            "num_workers": [
                0
            ],
            "patience": [
                10
            ],
            "lr": [
                0.00005,
                0.01
            ],
            "weight_decay": [
                1e-6,
                1e-3
            ],
            "device": None
        }
    }

    return default_params
