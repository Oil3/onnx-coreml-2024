# Method registry January 2025
# Contains every public CoreML neural networks ops - coremltools==8.1
#    from nn_builder_methods import method_registry

method_registry = {
    add_acos: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add an acos layer to the model that computes element-wise arc-cosine for the input tensor.',
    },
    add_acosh: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add an acosh layer to the model that computes element-wise inverse hyperbolic cosine for the input tensor.',
    },
    add_activation: {
        params: [
            name,
            non_linearity,
            input_name,
            output_name,
            paramss=None,
            input_rank=None,
            input_shape=None,
            output_rank=None,
            output_shape=None,
        ],
        param_docs: {
        },
        description: 'Add an activation layer to the model.',
    },
    add_add_broadcastable: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add an add_broadcastable layer to the model that performs element-wise addition operation with broadcast support.',
    },
    add_argmax: {
        params: [
            name,
            input_name,
            output_name,
            axis,
            keepdims=True,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int,
                description: 'axis along which the argmax is computed. Negative indexing is supported.',
            },
            keepdims: {
                type: bool, optional,
                description: 'if true, output rank is same as input rank, default: true.',
            },
        },
        description: 'Add an argmax layer to the model that returns the indices of the maximum value along a specified axis in the input tensor.',
    },
    add_argmin: {
        params: [
            name,
            input_name,
            output_name,
            axis,
            keepdims=True,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int,
                description: 'axis along which the argmin is computed. Negative indexing is supported.',
            },
            keepdims: {
                type: bool, optional,
                description: 'if true, output rank is same as input rank, default: true.',
            },
        },
        description: 'Add an argmin layer to the model that returns the indices of the minimum value along a specified axis in the input tensor.',
    },
    add_argsort: {
        params: [
            name,
            input_name,
            output_name,
            axis=0,
            descending=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int, optional,
                description: 'axis along which to compute the sorting indices',
            },
            descending: {
                type: bool, optional,
                description: 'order of sorting',
            },
        },
        description: 'Add an argsort layer to the model.',
    },
    add_asin: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add an asin layer to the model that computes element-wise arc-sine for the input tensor.',
    },
    add_asinh: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add an asinh layer to the model that computes element-wise inverse hyperbolic sine for the input tensor.',
    },
    add_atan: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add an atan layer to the model that computes element-wise arc-tangent for the input tensor.',
    },
    add_atanh: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add an atanh layer to the model that computes element-wise inverse hyperbolic tangent for the input tensor.',
    },
    add_batched_mat_mul: {
        params: [
            name,
            input_names,
            output_name,
            transpose_a=False,
            transpose_b=False,
            weight_matrix_rows=0,
            weight_matrix_columns=0,
            W=None,
            bias=None,
            int_8_dynamic_quantize=False,
            is_quantized_weight=False,
            quantization_type='linear',
            nbits=8,
            quant_scale=None,
            quant_bias=None,
            quant_lut=None,
        ],
        param_docs: {
        },
        description: 'Add a N-D Batched Matrix Multiplication layer with NumPy-like broadcasting.',
    },
    add_batchnorm: {
        params: [
            name,
            channels,
            gamma,
            beta,
            mean=None,
            variance=None,
            input_name='data',
            output_name='out',
            compute_mean_var=False,
            instance_normalization=False,
            epsilon=1e-05,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            channels: {
                type: int,
                description: 'Number of channels of the input blob.',
            },
            gamma: {
                type: numpy.array,
                description: 'Values of gamma. Must be numpy array of shape (channels, ) .',
            },
            beta: {
                type: numpy.array,
                description: 'Values of beta. Must be numpy array of shape (channels, ) .',
            },
            mean: {
                type: numpy.array,
                description: 'Means of the input blob on each channel. Must be numpy array of shape (channels, ) .',
            },
            variance: {
                type: ,
                description: 'Variances of the input blob on each channel. Must be numpy array of shape (channels, ) .',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            compute_mean_var: {
                type: bool,
                description: 'Set to True if mean and variance is to be computed from the input data.',
            },
            instance_normalization: {
                type: bool,
                description: 'Set compute_mean_var and this to True to perform
instance normalization. That is, mean and variance are computed
from the single input instance.',
            },
            epsilon: {
                type: float,
                description: 'Value of epsilon. Defaults to 1e-5 if not specified.',
            },
        },
        description: 'Add a batch normalization layer.',
    },
    add_bias: {
        params: [
            name,
            b,
            input_name,
            output_name,
            shape_bias=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            b: {
                type: int or numpy.array,
                description: 'Bias to add to the input.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            shape_bias: {
                type: list of int,
                description: 'List of ints that specifies the shape of the bias parameter
(if present). Can be [1] , [C] , [1,H,W] , or [C,H,W] .',
            },
        },
        description: 'Add a bias layer to the model.',
    },
    add_bidirlstm: {
        params: [
            name,
            W_h,
            W_x,
            b,
            W_h_back,
            W_x_back,
            b_back,
            hidden_size,
            input_size,
            input_names,
            output_names,
            inner_activation='SIGMOID',
            cell_state_update_activation='TANH',
            output_activation='TANH',
            peep=None,
            peep_back=None,
            output_all=False,
            forget_bias=False,
            coupled_input_forget_gate=False,
            cell_clip_threshold=50000.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            W_h: {
                type: [numpy.array],
                description: 'List of recursion weight matrices for the forward layer.
The ordering is [R_i, R_f, R_o, R_z] ,
where R_i , R_f , R_o , and R_z are weight matrices at
input gate, forget gate, output gate and cell gate.
The shapes of these matrices are (hidden_size, hidden_size) .',
            },
            W_x: {
                type: [numpy.array],
                description: 'List of input weight matrices for the forward layer. The ordering
is [W_i, W_f, W_o, W_z] ,
where W_i , W_f , W_o , and W_z are weight matrices at
input gate, forget gate, output gate and cell gate.
The shapes of these matrices are (hidden_size, input_size) .',
            },
            b: {
                type: [numpy.array],
                description: 'List of biases for the forward layer. The ordering is [b_i, b_f, b_o, b_z] ,
where b_i , b_f , b_o , and b_z are biases at input
gate, forget gate, output gate and cell gate.
If None , biases are ignored. Otherwise the shapes of the biases
are (hidden_size, ) .',
            },
            W_h_back: {
                type: [numpy.array],
                description: 'List of recursion weight matrices for the backward layer. The
ordering is [R_i, R_f, R_o, R_z] ,
where R_i , R_f , R_o , and R_z are weight matrices
at input gate, forget gate, output gate and cell gate.
The shapes of these matrices are (hidden_size, hidden_size) .',
            },
            W_x_back: {
                type: [numpy.array],
                description: 'List of input weight matrices for the backward layer. The ordering
is [W_i, W_f, W_o, W_z]` ,
where W_i , W_f , W_o , and W_z are weight matrices
at input gate, forget gate, output gate and cell gate.
The shapes of these matrices are (hidden_size, input_size) .',
            },
            b_back: {
                type: [numpy.array],
                description: 'List of biases for the backward layer. The ordering is [b_i, b_f, b_o, b_z] ,
where b_i , b_f , b_o , and b_z are biases at input
gate, forget gate, output gate and cell gate.
The shapes of the biases (hidden_size) .',
            },
            hidden_size: {
                type: int,
                description: 'Number of hidden units. This is equal to the number of channels of output shape.',
            },
            input_size: {
                type: int,
                description: 'Number of the number of channels of input shape.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer, in the order of [x, h_input, c_input, h_reverse_input, c_reverse_input] .',
            },
            output_names: {
                type: list of str,
                description: 'The output blob names of this layer, in the order of [y, h_output, c_output, h_reverse_output, c_reverse_output] .',
            },
            inner_activation: {
                type: str,
                description: 'Inner activation function used at input and forget gate. Can be one
of the following options:
[ \'RELU\' , \'TANH\' , \'SIGMOID\' , \'SCALED_TANH\' , \'SIGMOID_HARD\' , \'LINEAR\' ].
Defaults to \'SIGMOID\' .',
            },
            cell_state_update_activation: {
                type: str,
                description: 'Cell state update activation function used at the cell state update gate.
Can be one of the following options:
[ \'RELU\' , \'TANH\' , \'SIGMOID\' , \'SCALED_TANH\' , \'SIGMOID_HARD\' , \'LINEAR\' ].
Defaults to \'TANH\' .',
            },
            output_activation: {
                type: str,
                description: 'Activation function used at the output gate. Can be one of the following options:
[ \'RELU\' , \'TANH\' , \'SIGMOID\' , \'SCALED_TANH\' , \'SIGMOID_HARD\' , \'LINEAR\' ].
Defaults to \'TANH\' .',
            },
            peep: {
                type: [numpy.array] or None,
                description: 'List of peephole vectors for the forward layer. The ordering
is [p_i, p_f, p_o] ,
where p_i , p_f , and p_o are peephole vectors at input
gate, forget gate, and output gate.
The shapes of the peephole vectors are (hidden_size,) . Defaults to None .',
            },
            peep_back: {
                type: [numpy.array] or None,
                description: 'List of peephole vectors for the backward layer. The ordering
is [p_i, p_f, p_o] ,
where p_i , p_f , and p_o are peephole vectors at input
gate, forget gate, and output gate.
The shapes of the peephole vectors are (hidden_size,) . Defaults to None .',
            },
            output_all: {
                type: boolean,
                description: 'Whether the LSTM layer should output at every time step. Defaults to False . If False , the output is the result after the final state update. If True , the output is a sequence, containing outputs at all time steps.',
            },
            forget_bias: {
                type: boolean,
                description: 'If True , a vector of 1s is added to forget gate bias. Defaults to False .',
            },
            coupled_input_forget_gate: {
                type: boolean,
                description: 'If True , the input gate and forget gate is coupled. That is, the
forget gate is not used.
Defaults to False .',
            },
            cell_clip_threshold: {
                type: float,
                description: 'The limit on the maximum and minimum values on the cell state.
Defaults to 50.0.',
            },
        },
        description: 'Add a Bi-directional LSTM layer to the model.',
    },
    add_branch: {
        params: [
            name,
            input_name,
            if_branch=None,
            else_branch=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            if_branch: {
                type: NeuralNetwork,
                description: 'Neural network to execute if the absolute value of the input tensor is greater than 1e-6.',
            },
            else_branch: {
                type: NeuralNetwork, optional,
                description: 'Neural network to execute if the absolute value of the input tensor is less than 1e-6.',
            },
        },
        description: 'Add a branch layer to the model that provides the functionality of branching or an if-else block.',
    },
    add_broadcast_to_dynamic: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a broadcast_to_dynamic layer to the model that broadcasts a tensor to a compatible shape.',
    },
    add_broadcast_to_like: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a broadcast_to_like layer to the model that broadcasts a tensor to a compatible shape.',
    },
    add_broadcast_to_static: {
        params: [
            name,
            input_name,
            output_name,
            output_shape,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            output_shape: {
                type: list of int or tuple of int,
                description: 'The target shape of the output tensor.',
            },
        },
        description: 'Add a broadcast_to_static layer to the model that broadcasts a tensor to a compatible shape.',
    },
    add_categorical_distribution: {
        params: [
            name,
            input_name,
            output_name,
            num_samples,
            is_logits=True,
            eps=1e-10,
            temperature=1.0,
            seed=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            num_samples: {
                type: int,
                description: 'List of dimensions for the reduce operations.',
            },
            is_logits: {
                type: bool, optional,
                description: 'If true, the input is log probabilities. If false, the input is
probabilities, default: True',
            },
            eps: {
                type: float, optional,
                description: 'Epsilon parameter for categorical distribution, default 1e-10.',
            },
            temperature: {
                type: float, optional,
                description: 'Temperature parameter for categorical distribution, default 1.0.',
            },
            seed: {
                type: int, optional,
                description: 'Used to create a random seed for the distribution. default -1 (random).',
            },
        },
        description: 'Add a categorical_distribution layer to the model that fills the output tensor with random values from categorical distribution.',
    },
    add_ceil: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a ceil layer to the model that performs element-wise ceil operation on the input tensor that rounds the value to the smallest integer not less than x.',
    },
    add_clamped_relu: {
        params: [
            name,
            input_name,
            output_name,
            alpha=0.0,
            beta=6.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            alpha: {
                type: float, optional,
                description: 'slope of the output when input is negative, default: 0.0.',
            },
            beta: {
                type: float, optional,
                description: 'Upper bound on the output value, default: 6.0.',
            },
        },
        description: 'Add a clamped relu layer to the model.',
    },
    add_clip: {
        params: [
            name,
            input_name,
            output_name,
            min_value=0.0,
            max_value=1.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            min_value: {
                type: float, optional,
                description: 'Lower bound / minimum value for clip, default: 0.0.',
            },
            max_value: {
                type: float, optional,
                description: 'Upper bound / maximum value for clip, default: 1.0.',
            },
        },
        description: 'Add a clip layer to the model that performs element-wise clip operation.',
    },
    add_concat_nd: {
        params: [
            name,
            input_names,
            output_name,
            axis,
            interleave=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int,
                description: 'Axis to perform the concat operation on.',
            },
            interleave: {
                type: Unknown,
                description: '(Only available in Core ML Specification >= 5 (iOS >= 14, macOS >= 11.0)
If true, concatenate by interleaving the inputs',
            },
        },
        description: 'Add a concat_nd layer to the model that performs concatenation along the given axis.',
    },
    add_constant_pad: {
        params: [
            name,
            input_names,
            output_name,
            value=0.0,
            pad_to_given_output_size_mode=False,
            pad_amounts=[],
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob name(s) of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            value: {
                type: float,
                description: 'value to be used for padding.',
            },
            pad_to_given_output_size_mode: {
                type: bool,
                description: 'if true, pad_amounts are interpreted as output shapes (see example in NeuralNetwork.proto)',
            },
            pad_amounts: {
                type: [int], optional,
                description: 'must be non negative. Amount to pad in each dimension. Length of the list must be twice the input/output rank.
Not required if second input is present.',
            },
        },
        description: 'Add a constant pad layer.',
    },
    add_convolution: {
        params: [
            name,
            kernel_channels,
            output_channels,
            height,
            width,
            stride_height,
            stride_width,
            border_mode,
            groups,
            W,
            b,
            has_bias,
            is_deconv=False,
            output_shape=None,
            input_name='data',
            output_name='out',
            dilation_factors=[1,1],
            padding_top=0,
            padding_bottom=0,
            padding_left=0,
            padding_right=0,
            same_padding_asymmetry_mode='BOTTOM_RIGHT_HEAVY',
            **kwargs,
        ],
        param_docs: {
        },
        description: 'Add a convolution layer to the network.',
    },
    add_convolution3d: {
        params: [
            name,
            input_channels,
            output_channels,
            depth,
            height,
            width,
            W,
            b,
            has_bias,
            groups=1,
            stride_depth=1,
            stride_height=1,
            stride_width=1,
            dilation_width=1,
            dilation_height=1,
            dilation_depth=1,
            is_deconv=False,
            output_shape=None,
            padding_mode='valid',
            padding_front=0,
            padding_back=0,
            padding_top=0,
            padding_bottom=0,
            padding_left=0,
            padding_right=0,
            input_name='data',
            output_name='out',
        ],
        param_docs: {
        },
        description: 'Add a 3 dimensional convolution layer to the network.',
    },
    add_copy: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a copy layer to the model that copies its input tensor to the output tensor.',
    },
    add_cos: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a cos layer to the model that computes element-wise cosine for the input tensor.',
    },
    add_cosh: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a osh layer to the model that computes element-wise hyperbolic cosine for the input tensor.',
    },
    add_crop: {
        params: [
            name,
            left,
            right,
            top,
            bottom,
            offset,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            left: {
                type: int,
                description: 'Number of elements to be cropped on the left side of the input blob.
When the crop layer takes 2 inputs, this parameter is ignored.',
            },
            right: {
                type: int,
                description: 'Number of elements to be cropped on the right side of the input blob.
When the crop layer takes 2 inputs, this parameter is ignored.',
            },
            top: {
                type: int,
                description: 'Number of elements to be cropped on the top of the input blob.
When the crop layer takes 2 inputs, this parameter is ignored.',
            },
            bottom: {
                type: int,
                description: 'Number of elements to be cropped on the bottom of the input blob.
When the crop layer takes 2 inputs, this parameter is ignored.',
            },
            offset: {
                type: list of int,
                description: 'Offset along the height and width directions when the crop layer takes 2 inputs. Must be a list of length 2.
When the crop layer takes 1 input, this parameter is ignored.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer. Must be either a list of 1 string (1 input crop layer),
or a list of 2 strings (2-input crop layer).',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a cropping layer to the model.',
    },
    add_crop_resize: {
        params: [
            name,
            input_names,
            output_name,
            target_height=1,
            target_width=1,
            mode='STRICT_ALIGN_ENDPOINTS_MODE',
            normalized_roi=False,
            box_indices_mode='CORNERS_HEIGHT_FIRST',
            spatial_scale=1.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'Must be a list of two names: image feature map and crop indices/RoI input. First input corresponds to a blob with shape [1, Batch, C, H_in, W_in] .
This represents a batch of input image feature data with C channels. The second input shape must be [N, 1, 4, 1, 1] or [N, 1, 5, 1, 1] .
This represents the bounding box coordinates for N patches/RoIs. N : number of patches/RoIs to be extracted. If RoI shape = [N, 1, 4, 1, 1] , the channel axis corresponds
to the four coordinates specifying the bounding box.
All the N~ RoIs are extracted from all the batches of the input. If RoI shape = [N, 1, 5, 1, 1] , the first element of the
channel axis specifies the input batch id from which to extract the RoI and
must be in the interval [0, Batch - 1] . That is, n -th RoI is
extracted from the RoI[n,0,0,0] -th input batch id.
The last four elements of the channel axis specify the
bounding box coordinates.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            target_height: {
                type: int,
                description: 'Output height dimension.',
            },
            target_width: {
                type: int,
                description: 'Output width dimension.',
            },
            mode: {
                type: str,
                description: 'The following values are supported: \'STRICT_ALIGN_ENDPOINTS_MODE\' , \'ALIGN_ENDPOINTS_MODE\' , \'UPSAMPLE_MODE\' , \'ROI_ALIGN_MODE\' . This parameter determines the sampling grid used for bilinear interpolation.',
            },
            normalized_roi: {
                type: bool,
                description: 'If true the bounding box coordinates must be in the interval [0, 1] .
They are scaled by (input_height - 1) , (input_width - 1) ;
that is, based on the input spatial dimensions. If false the bounding box coordinates must be in the interval [0, input_height - 1] and [0, input_width - 1] ,
respectively for height and width dimensions.',
            },
            box_indices_mode: {
                type: str,
                description: 'The following values are supported: \'CORNERS_HEIGHT_FIRST\' , \'CORNERS_WIDTH_FIRST\' , \'CENTER_SIZE_HEIGHT_FIRST\' , \'CENTER_SIZE_WIDTH_FIRST\' . Representation used to interpret the bounding box coordinates (RoI) input. \'CORNERS_HEIGHT_FIRST\' : [h_start, w_start, h_end, w_end] \'CORNERS_WIDTH_FIRST\' : [w_start, h_start, w_end, h_end] \'CENTER_SIZE_HEIGHT_FIRST\' : [h_center, w_center, box_height, box_width] \'CENTER_SIZE_WIDTH_FIRST\' : [w_center, h_center, box_width, box_height]',
            },
            spatial_scale: {
                type: float,
                description: 'Additional spatial scale that multiplies the bounding box coordinates.
Generally used while implementing the RoI Align layer,
which uses unnormalized RoI coordinates along with a spatial scale less than or equal to 1.',
            },
        },
        description: 'Add a crop resize layer to the model.',
    },
    add_cumsum: {
        params: [
            name,
            input_names,
            output_name,
            axis=-1,
            reverse=False,
            exclusive=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int, optional,
                description: 'Axis to perform the operation, default: -1.',
            },
            reverse: {
                type: bool, optional,
                description: 'if true, cumsum is performed in the opposite direction, default: False.',
            },
            exclusive: {
                type: bool, optional,
                description: 'whether to perform exclusive or inclusive cumulative summation, default: False.',
            },
        },
        description: 'Add a cum sum layer to the model computes the cumulative sum values of the input along a given axis.',
    },
    add_custom: {
        params: [
            name,
            input_names,
            output_names,
            custom_proto_spec=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names to this layer.',
            },
            output_names: {
                type: list of str,
                description: 'The output blob names from this layer.',
            },
            custom_proto_spec: {
                type: CustomLayerParams,
                description: 'A protobuf CustomLayerParams message. This can also be left blank and filled in later.',
            },
        },
        description: 'Add a custom layer.',
    },
    add_divide_broadcastable: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a divide_broadcastable layer to the model that performs element-wise division operation with broadcast support.',
    },
    add_elementwise: {
        params: [
            name,
            input_names,
            output_name,
            mode,
            alpha=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'A list of input blob names of this layer. The input blobs should have the same shape.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            mode: {
                type: str,
                description: 'A string specifying the mode of the elementwise layer. It can be one of the following: \'CONCAT\' : Concatenate input blobs along the channel axis. \'SEQUENCE_CONCAT\' : Concatenate input blobs along the sequence axis. \'ADD\' : Perform an element-wise summation over the input blobs. \'MULTIPLY\' : Perform an element-wise multiplication over the input blobs. \'DOT\' : Compute the dot product of the two input blobs.
In this mode, the length of input_names should be 2. \'COS\' : Compute the cosine similarity of the two input blobs.
In this mode, the length of input_names should be 2. \'MAX\' : Compute the element-wise maximum over the input blobs. `\'MIN\'` : Compute the element-wise minimum over the input blobs. \'AVE\' : Compute the element-wise average over the input blobs.',
            },
            alpha: {
                type: float,
                description: 'if mode == \'ADD\' and there is only one input_name , alpha is added to the input. if mode == \'MULTIPLY\' and there is only one input_name , alpha is multiplied to the input.',
            },
        },
        description: 'Add an element-wise operation layer to the model.',
    },
    add_embedding: {
        params: [
            name,
            W,
            b,
            input_dim,
            output_channels,
            has_bias,
            input_name,
            output_name,
            is_quantized_weight=False,
            quantization_type='linear',
            nbits=8,
            quant_scale=None,
            quant_bias=None,
            quant_lut=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            W: {
                type: float32 numpy.array or bytes(),
                description: 'Weight matrix of shape (output_channels, input_dim) .
If W is of type bytes() (quantized to 1-8 bits), other
quantization related arguments must be provided as well (see below).',
            },
            b: {
                type: numpy.array,
                description: 'Bias vector of shape (output_channels, ) .',
            },
            input_dim: {
                type: int,
                description: 'Size of the vocabulary (1 + maximum integer index of the words).',
            },
            output_channels: {
                type: int,
                description: 'Number of output channels.',
            },
            has_bias: {
                type: boolean,
                description: 'Whether the bias vector of this layer is ignored in the spec . If True, the bias vector of this layer is not ignored. If False, the bias vector is ignored.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            Quantization arguments expected, when ``W`` is of type ``bytes()``: {
                type: ,
                description: '',
            },
            is_quantized_weight: {
                type: bool,
                description: 'Set it to true when W is of type bytes() , representing quantized weights.',
            },
            quantization_type: {
                type: str,
                description: 'When weights are quantized (that is, W is of type bytes() ),
this should be either "linear" or "lut" .',
            },
            nbits: {
                type: int,
                description: 'Should be between 1 and 8 (inclusive). Number of bits per weight value.',
            },
            quant_scale: {
                type: numpy.array(dtype=numpy.float32),
                description: 'Scale vector to be used with linear quantization.
Must be of length either 1 or output_channels.',
            },
            quant_bias: {
                type: numpy.array(dtype=numpy.float32),
                description: 'Bias vector to be used with linear quantization.
Must be of length either 1 or output_channels.',
            },
            quant_lut: {
                type: numpy.array(dtype=numpy.float32),
                description: 'The LUT (look up table) to be used with LUT quantization.
Must be of length 2^n bits.',
            },
        },
        description: 'Add an embedding layer to the model.',
    },
    add_embedding_nd: {
        params: [
            name,
            input_name,
            output_name,
            vocab_size,
            embedding_size,
            W,
            b=None,
            is_quantized_weight=False,
            quantization_type='linear',
            nbits=8,
            quant_scale=None,
            quant_bias=None,
            quant_lut=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            vocab_size: {
                type: int,
                description: 'Size of the vocabulary (1 + maximum integer index of the words).',
            },
            embedding_size: {
                type: int,
                description: 'Size of the embedded vector.',
            },
            W: {
                type: float32 numpy.array or bytes(),
                description: 'Weight matrix of shape (embedding_size, vocab_size).
If W is of type bytes(), i.e. quantized to 1-8 bits, other quantization
related arguments must be provided as well (see below).',
            },
            b: {
                type: numpy.array , optional,
                description: 'Bias vector of shape (embedding_size, ).',
            },
            Quantization arguments expected, when W is of type bytes(): {
                type: ,
                description: '',
            },
            is_quantized_weight: {
                type: bool,
                description: 'Set it to true when W is of type bytes(), representing quantized weights',
            },
            quantization_type: {
                type: str,
                description: 'When weights are quantized (i.e. W is of type bytes()), this should be either “linear” or “lut”.',
            },
            nbits: {
                type: int,
                description: 'Should be between 1 and 8 (inclusive). Number of bits per weight value.',
            },
            quant_scale: {
                type: numpy.array(dtype=numpy.float32),
                description: 'scale vector to be used with linear quantization. Must be of length either 1 or embedding_size.',
            },
            quant_bias: {
                type: numpy.array(dtype=numpy.float32),
                description: 'bias vector to be used with linear quantization. Must be of length either 1 or embedding_size.',
            },
            quant_lut: {
                type: numpy.array(dtype=numpy.float32),
                description: 'the LUT (look up table) to be used with LUT quantization. Must be of length 2^nbits.',
            },
        },
        description: 'Add an embedding layer to the model that performs a matrix lookup and optionally adds a bias.',
    },
    add_equal: {
        params: [
            name,
            input_names,
            output_name,
            alpha=0.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            alpha: {
                type: float, optional,
                description: 'y = x1 != alpha, if only one input is provided, default: 0.',
            },
        },
        description: 'Add an equal layer to the model that performs the element-wise equal (=) operation.',
    },
    add_erf: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add an erf function (gaussian error function) layer to the model.',
    },
    add_exp2: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add an exp2 layer to the model that performs element-wise experiential operation.',
    },
    add_expand_dims: {
        params: [
            name,
            input_name,
            output_name,
            axes,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int,
                description: 'Dimensions the operation perform on.',
            },
        },
        description: 'Add an expand dims layer to the model that increases the rank of the input tensor by adding unit dimensions.',
    },
    add_fill_dynamic: {
        params: [
            name,
            input_name,
            output_name,
            value=0.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            value: {
                type: float, optional,
                description: 'A scalar value for the fill operation, default: 0.',
            },
        },
        description: 'Add a fill_dynamic layer to the model that outputs a tensor filled with a scalar value.',
    },
    add_fill_like: {
        params: [
            name,
            input_name,
            output_name,
            value=0.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            value: {
                type: float, optional,
                description: 'A scalar value for the fill operation, default 0.',
            },
        },
        description: 'Add a fill_like layer to the model outputs a tensor filled with a scalar value.',
    },
    add_fill_static: {
        params: [
            name,
            output_name,
            output_shape,
            value=0.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            output_shape: {
                type: list of int or tuple of int,
                description: 'The target shape of the output tensor.',
            },
            value: {
                type: float, optional,
                description: 'A scalar value for the fill operation, default 0.',
            },
        },
        description: 'Add a fill_static layer to the model that outputs a tensor filled with a scalar value given shape as parameter.',
    },
    add_flatten: {
        params: [
            name,
            mode,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            mode: {
                type: int,
                description: 'If mode == 0, the flatten layer is in CHANNEL_FIRST mode. If mode == 1, the flatten layer is in CHANNEL_LAST mode.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a flatten layer.',
    },
    add_flatten_to_2d: {
        params: [
            name,
            input_name,
            output_name,
            axis=1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The of input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int, optional,
                description: 'Axis to perform the operation, default: 1.',
            },
        },
        description: 'Add a flatten_to_2d layer to the model that flattens the input tensor into a 2-dimensional matrix.',
    },
    add_floor: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a floor layer to the model that performs element-wise floor operation on the input tensor that rounds the value to the largest integer not greater than x.',
    },
    add_floor_div_broadcastable: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a floor_div_broadcastable layer to the model that performs floor division operation with broadcast support.',
    },
    add_gather: {
        params: [
            name,
            input_names,
            output_name,
            axis=0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int, optional,
                description: 'The axis the operation perform on, default: 0.',
            },
        },
        description: 'Add a gather layer to the model that gathers elements or slices from data and store to a tensor whose shape is defined by indices from the input.',
    },
    add_gather_along_axis: {
        params: [
            name,
            input_names,
            output_name,
            axis=0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int, optional,
                description: 'The axis the operation perform on, default: 0.',
            },
        },
        description: 'Add a gather_along_axis layer to the model that gathers elements or slices from data and store to a tensor whose shape is defined by indices from the input along the given axis into the output tensor.',
    },
    add_gather_nd: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a gather layer to the model that gathers elements or slices from data and store to a tensor whose shape is defined by indices from the input.',
    },
    add_gelu: {
        params: [
            name,
            input_name,
            output_name,
            mode='EXACT',
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            mode: {
                type: str, optional,
                description: 'Gelu mode in [EXACT | TANH_APPROXIMATION | SIGMOID_APPROXIMATION], default EXACT.',
            },
        },
        description: 'Add a GELU (gaussian error linear unit) activation layer, which is: 0.5 * x * (1 + erf(x / sqrt(2))) .',
    },
    add_get_shape: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a get_shape layer to the model.',
    },
    add_global_pooling3d: {
        params: [
            name,
            input_name,
            output_name,
            pooling_type,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            pooling_type: {
                type: str,
                description: 'Type of pooling performed. Can either be \'MAX\' OR \'AVERAGE\' .',
            },
        },
        description: 'Add a layer to pool three spatial dimensions down to one value.',
    },
    add_greater_than: {
        params: [
            name,
            input_names,
            output_name,
            use_greater_than_equal=False,
            alpha=0.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            use_greater_than_equal: {
                type: bool, optional,
                description: 'Whether or not to allow greater than or equal to, default: false.',
            },
            alpha: {
                type: float, optional,
                description: 'y = x1 != alpha, if only one input is provided, default: 0.',
            },
        },
        description: 'Add a greater_than layer to the model that performs the element-wise greater-than (>) operation or greater-than-or-equal-to (>=) operation.',
    },
    add_gru: {
        params: [
            name,
            W_h,
            W_x,
            b,
            hidden_size,
            input_size,
            input_names,
            output_names,
            activation='TANH',
            inner_activation='SIGMOID_HARD',
            output_all=False,
            reverse_input=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            W_h: {
                type: [numpy.array],
                description: 'List of recursion weight matrices. The ordering is [R_z, R_r, R_o] ,
where R_z , R_r and R_o are weight matrices at update gate,
reset gate and output gate.
The shapes of these matrices are (hidden_size, hidden_size) .',
            },
            W_x: {
                type: [numpy.array],
                description: 'List of input weight matrices. The ordering is [W_z, W_r, W_o] ,
where W_z , W_r , and W_o are weight matrices at update gate,
reset gate and output gate.
The shapes of these matrices are (hidden_size, input_size) .',
            },
            b: {
                type: [numpy.array] or None,
                description: 'List of biases of the GRU layer. The ordering is [b_z, b_r, b_o] ,
where b_z , b_r , and b_o are biases at update gate,
reset gate and output gate.
If None , biases are ignored. Otherwise the shapes of the biases are (hidden_size, ) .',
            },
            hidden_size: {
                type: int,
                description: 'Number of hidden units. This is equal to the number of channels of output shape.',
            },
            input_size: {
                type: int,
                description: 'Number of the number of channels of input shape.',
            },
            activation: {
                type: str,
                description: 'Activation function used at the output gate. Can be one of the following options:
[ \'RELU\' , \'TANH\' , \'SIGMOID\' , \'SCALED_TANH\' , \'SIGMOID_HARD\' , \'LINEAR\' ].
Defaults to \'TANH\' .
See add_activation for more detailed description.',
            },
            inner_activation: {
                type: str,
                description: 'Inner activation function used at update and reset gates.
Can be one of the following options:
[ \'RELU\' , \'TANH\' , \'SIGMOID\' , \'SCALED_TANH\' , \'SIGMOID_HARD\' , \'LINEAR\' ].
Defaults to \'SIGMOID_HARD\' .
See add_activation for more detailed description.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names list of this layer, in the order of [x, h_input] .',
            },
            output_names: {
                type: list of str,
                description: 'The output blob names list of this layer, in the order of [y, h_output] .',
            },
            output_all: {
                type: boolean,
                description: 'Whether the recurrent layer should output at every time step. If False, the output is the result after the final state update. If True, the output is a sequence, containing outputs at all time steps.',
            },
            reverse_input: {
                type: boolean,
                description: 'Whether the recurrent layer should process the input sequence in the reverse order. If False, the input sequence order is not reversed. If True, the input sequence order is reversed.',
            },
        },
        description: 'Add a Gated-Recurrent Unit (GRU) layer to the model.',
    },
    add_inner_product: {
        params: [
            name,
            W,
            b,
            input_channels,
            output_channels,
            has_bias,
            input_name,
            output_name,
            int_8_dynamic_quantize=False,
            is_quantized_weight=False,
            quantization_type='linear',
            nbits=8,
            quant_scale=None,
            quant_bias=None,
            quant_lut=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            W: {
                type: numpy.array or bytes(),
                description: 'Weight matrix of shape (output_channels, input_channels) .
If W is of type bytes() (quantized), other quantization
related arguments must be provided as well (see below).',
            },
            b: {
                type: numpy.array,
                description: 'Bias vector of shape: (output_channels, ) .',
            },
            input_channels: {
                type: int,
                description: 'Number of input channels.',
            },
            output_channels: {
                type: int,
                description: 'Number of output channels.',
            },
            has_bias: {
                type: boolean,
                description: 'Whether the bias vector of this layer is ignored in the spec. If True, the bias vector of this layer is not ignored. If False, the bias vector is ignored.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            Quantization arguments, used when ``W`` is of type ``bytes()``: {
                type: ,
                description: 'int_8_dynamic_quantize: boolean Whether to quantize and dequantize before and after inner product, respectively.
Expects byte weights, representing int8 values, if True.
See NeuralNetwork.proto for other validation conditions. is_quantized_weight: bool, optional Set it to true when W is of type bytes() , representing
quantized weights, default: false. quantization_type: str When weights are quantized (that is, W is of type bytes() ),
this should be either "linear" or "lut" . nbits: int Should be between 1 and 8 (inclusive). Number of bits per weight
value. Only applicable when weights are quantized. quant_scale: numpy.array(dtype=numpy.float32) scale vector to be used with linear quantization. Must be of
length either 1 or output_channels. quant_bias: numpy.array(dtype=numpy.float32) bias vector to be used with linear quantization. Must be of
length either 1 or output_channels. quant_lut: numpy.array(dtype=numpy.float32) the LUT (look up table) to be used with LUT quantization.
Must be of length 2^n bits.',
            },
        },
        description: 'Add an inner product layer to the model.',
    },
    add_l2_normalize: {
        params: [
            name,
            input_name,
            output_name,
            epsilon=1e-05,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            epsilon: {
                type: float,
                description: 'small bias to avoid division by zero.',
            },
        },
        description: 'Add L2 normalize layer.',
    },
    add_layer_normalization: {
        params: [
            name,
            input_name,
            output_name,
            normalized_shape,
            gamma,
            beta,
            eps=1e-05,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            normalized_shape: {
                type: list of int or tuple of int,
                description: 'Input shape from an expected input of size.',
            },
            gamma: {
                type: WeightParams,
                description: 'Weight parameters.',
            },
            beta: {
                type: WeightParams,
                description: 'Bias parameters.',
            },
            eps: {
                type: float, optional,
                description: 'Constant value added to the denominator, default: 1e-5.',
            },
        },
        description: 'Add a layer normalization layer to the model that applies layer normalization over the input tensor.',
    },
    add_less_than: {
        params: [
            name,
            input_names,
            output_name,
            use_less_than_equal=False,
            alpha=0.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            use_less_than_equal: {
                type: bool, optional,
                description: 'Whether or not to allow less than or equal to, default: false.',
            },
            alpha: {
                type: float, optional,
                description: 'y = x1 != alpha, if only one input is provided, default: 0.',
            },
        },
        description: 'Add a less_than layer to the model that performs the element-wise less-than (<) operation or less-than-or-equal-to (<=) operation.',
    },
    add_load_constant: {
        params: [
            name,
            output_name,
            constant_value,
            shape,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            constant_value: {
                type: numpy.array,
                description: 'value of the constant as a numpy array.',
            },
            shape: {
                type: list of int or tuple of int,
                description: 'List of ints representing the shape of the constant. Must be of length 3: [C,H,W]',
            },
        },
        description: 'Add a load constant layer.',
    },
    add_load_constant_nd: {
        params: [
            name,
            output_name,
            constant_value,
            shape,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            constant_value: {
                type: numpy.array(),
                description: 'value of the constant as a numpy array.',
            },
            shape: {
                type: list of int or tuple of int,
                description: 'List of ints representing the shape of the constant.',
            },
        },
        description: 'Add a load_constant layer that loads data as a parameter and provides it as an output.',
    },
    add_logical: {
        params: [
            name,
            input_names,
            output_name,
            mode,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            mode: {
                type: str,
                description: 'Logical operation mode in [AND | OR | XOR | NOT].',
            },
        },
        description: 'Add a logical layer to the model that performs element-wise logical and/or/xor/not operation.',
    },
    add_loop: {
        params: [
            name,
            body_network=None,
            input_name=None,
            condition=None,
            condition_network=None,
            max_iterations=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            body_network: {
                type: NeuralNetwork,
                description: 'Neural network to execute for the body of the loop.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            condition: {
                type: str, optional,
                description: 'Condition of the loop.',
            },
            condition_network: {
                type: NeuralNetwork, optional,
                description: 'Neural network to execute for the condition of the loop.',
            },
            max_iterations: {
                type: int, optional,
                description: 'Maximum number of iterations of the loop.',
            },
        },
        description: 'Add a loop layer to the model that provides the functionality of a for loop, or a while loop.',
    },
    add_loop_break: {
        params: [
            name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
        },
        description: 'Add a loop_break layer to the model that terminates the loop that contains this layer.',
    },
    add_loop_continue: {
        params: [
            name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
        },
        description: 'Add a loop_continue layer to the model that stops the current loop iteration and continue on the next iteration.',
    },
    add_lower_triangular: {
        params: [
            name,
            input_name,
            output_name,
            k=0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The of input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            k: {
                type: int, optional,
                description: 'Diagonal below which to zero elements, default: 0 (main diagonal),
k < 0 is lower it and k > 0 is upper.',
            },
        },
        description: 'Add a lower_triangular layer to the model that copies a tensor setting everything outside lower triangular to zero.',
    },
    add_lrn: {
        params: [
            name,
            input_name,
            output_name,
            alpha,
            beta,
            local_size,
            k=1.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            alpha: {
                type: float,
                description: 'multiplicative constant in the denominator.',
            },
            beta: {
                type: float,
                description: 'exponent of the normalizing term in the denominator.',
            },
            k: {
                type: float,
                description: 'bias term in the denominator. Must be positive.',
            },
            local_size: {
                type: int,
                description: 'size of the neighborhood along the channel axis.',
            },
        },
        description: 'Add a LRN (local response normalization) layer.',
    },
    add_matrix_band_part: {
        params: [
            name,
            input_name,
            output_name,
            num_lower=-1,
            num_upper=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The of input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            num_lower: {
                type: int, optional,
                description: 'Number of lower sub-diagonals to keep.
Default: -1 (keep entire lower triangle).',
            },
            num_upper: {
                type: int, optional,
                description: 'Number of upper sub-diagonals to keep.
Default: -1 (keep entire upper triangle).',
            },
        },
        description: 'Add a matrix_band_part layer to the model that copies a tensor setting everything outside a central band in each inner-most matrix to zero.',
    },
    add_max_broadcastable: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a max_broadcastable layer to the model that performs element-wise maximum operation with broadcast support.',
    },
    add_min_broadcastable: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a min_broadcastable layer to the model that performs element-wise minimum operation with broadcast support.',
    },
    add_mod_broadcastable: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a mod_broadcastable layer to the model that performs element-wise modular operation with broadcast support.',
    },
    add_multiply_broadcastable: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a multiply_broadcastable layer to the model that performs element-wise multiplication operation with broadcast support.',
    },
    add_mvn: {
        params: [
            name,
            input_name,
            output_name,
            across_channels=True,
            normalize_variance=True,
            epsilon=1e-05,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            across_channels: {
                type: boolean,
                description: 'If False, each channel plane is normalized separately
If True, mean/variance is computed across all C, H and W dimensions',
            },
            normalize_variance: {
                type: boolean,
                description: 'If False, only mean subtraction is performed.',
            },
            epsilon: {
                type: float,
                description: 'small bias to avoid division by zero.',
            },
        },
        description: 'Add an MVN (mean variance normalization) layer.',
    },
    add_nms: {
        params: [
            name,
            input_names,
            output_names,
            iou_threshold=0.5,
            score_threshold=0.0,
            max_boxes=1,
            per_class_suppression=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer. Must be at least 2, and maximum 5.',
            },
            output_names: {
                type: list of str,
                description: 'The output blob names of this layer. Must be of length 4 exactly.',
            },
            iou_threshold: {
                type: float,
                description: 'intersection over union threshold for suppression. Ignored if 3rd input is present.',
            },
            score_threshold: {
                type: float,
                description: 'threshold for selecting boxes to be used for NMS algorithm. Ignored if 4th input is present.',
            },
            max_boxes: {
                type: int,
                description: 'maximum number of boxes to output. Ignored if 5th input is present.',
            },
            per_class_suppression: {
                type: bool,
                description: 'If true, boxes are organized into classes and suppression is applied to each class group separately',
            },
        },
        description: 'Add a non maximum suppression layer.',
    },
    add_not_equal: {
        params: [
            name,
            input_names,
            output_name,
            alpha=0.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            alpha: {
                type: float, optional,
                description: 'y = x1 != alpha, if only one input is provided, default: 0.',
            },
        },
        description: 'Add a not_equal layer to the model that performs the element-wise not equal (!=) operation.',
    },
    add_one_hot: {
        params: [
            name,
            input_names,
            output_name,
            one_hot_vector_size=None,
            axis=-1,
            on_value=1.0,
            off_value=0.0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            one_hot_vector_size: {
                type: int > 0,
                description: 'size of the one hot vector.',
            },
            axis: {
                type: int, optional,
                description: 'refers to the axis in the output tensor, default: -1.',
            },
            on_value: {
                type: float, optional,
                description: 'Constant value on locations represented by first input, default: 1.0.',
            },
            off_value: {
                type: float, optional,
                description: 'Constant value at all other locations, default: 0.0.',
            },
        },
        description: 'Add a one hot layer to the model that computes the one hot representation of the input tensor.',
    },
    add_optionals: {
        params: [
            optionals_in,
            optionals_out,
        ],
        param_docs: {
            optionals_in: {
                type: list of str,
                description: 'List of inputs that are optionals.',
            },
            optionals_out: {
                type: list of str,
                description: 'List of outputs that are optionals.',
            },
        },
        description: 'Add optional inputs and outputs to the model spec.',
    },
    add_padding: {
        params: [
            name,
            left=0,
            right=0,
            top=0,
            bottom=0,
            value=0,
            input_name='data',
            output_name='out',
            padding_type='constant',
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            left: {
                type: int,
                description: 'Number of elements to be padded on the left side of the input blob.',
            },
            right: {
                type: int,
                description: 'Number of elements to be padded on the right side of the input blob.',
            },
            top: {
                type: int,
                description: 'Number of elements to be padded on the top of the input blob.',
            },
            bottom: {
                type: int,
                description: 'Number of elements to be padded on the bottom of the input blob.',
            },
            value: {
                type: float,
                description: 'Value of the elements padded. Used only when padding_type = \'constant\' .',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            padding_type: {
                type: str,
                description: 'Type of the padding. Can be one of \'constant\' , \'reflection\' , or \'replication\' .',
            },
        },
        description: 'Add a padding layer to the model that performs padding along spatial dimensions.',
    },
    add_permute: {
        params: [
            name,
            dim,
            input_name,
            output_name,
        ],
        param_docs: {
        },
        description: 'Add a permute layer.',
    },
    add_pooling: {
        params: [
            name,
            height,
            width,
            stride_height,
            stride_width,
            layer_type,
            padding_type,
            input_name,
            output_name,
            exclude_pad_area=True,
            is_global=False,
            padding_top=0,
            padding_bottom=0,
            padding_left=0,
            padding_right=0,
            same_padding_asymmetry_mode='BOTTOM_RIGHT_HEAVY',
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            height: {
                type: int,
                description: 'Height of pooling region.',
            },
            width: {
                type: int,
                description: 'Width of pooling region.',
            },
            stride_height: {
                type: int,
                description: 'Stride along the height direction.',
            },
            stride_width: {
                type: int,
                description: 'Stride along the width direction.',
            },
            layer_type: {
                type: str,
                description: 'Type of pooling performed. Can either be \'MAX\' , \'AVERAGE\' , or \'L2\' .',
            },
            padding_type: {
                type: str,
                description: 'Option for the type of padding and output blob shape. Can be either \'VALID\' , \'SAME\' , or \'INCLUDE_LAST_PIXEL\' .',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            exclude_pad_area: {
                type: boolean,
                description: 'Whether to exclude padded area in the \'AVERAGE\' pooling operation,
default: true. This flag is only used with average pooling. If True, the value of the padded area will be excluded. If False, the padded area will be included.',
            },
            is_global: {
                type: boolean,
                description: 'Whether the pooling operation is global. Defaults to False. If True, the pooling operation is global. The pooling region
is of the same size of the input blob.
Parameters height , width , stride_height , and stride_width will be ignored. If False, the pooling operation is not global.',
            },
            padding_top, padding_bottom, padding_left, padding_right: {
                type: int,
                description: 'Values of height (top, bottom) and width (left, right) padding
to be used if padding type is "VALID" or "INCLUDE_LAST_PIXEL" .',
            },
            same_padding_asymmetry_mode: {
                type: str.,
                description: 'Type of asymmetric padding to be used when padding_type = \'SAME\' .
Can be either \'BOTTOM_RIGHT_HEAVY\' or \'TOP_LEFT_HEAVY\' .',
            },
        },
        description: 'Add a pooling layer to the model that performs spatial pooling.',
    },
    add_pooling3d: {
        params: [
            name,
            input_name,
            output_name,
            pooling_type,
            kernel_depth,
            kernel_height,
            kernel_width,
            stride_depth,
            stride_height,
            stride_width,
            padding_mode='valid',
            custom_padding_front=0,
            custom_padding_back=0,
            custom_padding_top=0,
            custom_padding_bottom=0,
            custom_padding_left=0,
            custom_padding_right=0,
            average_pooling_count_excludes_padding=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            pooling_type: {
                type: str,
                description: 'Type of pooling performed. Can either be \'MAX\' OR \'AVERAGE\' .',
            },
            kernel_depth: {
                type: int,
                description: 'Depth of the pooling region.',
            },
            kernel_height: {
                type: int,
                description: 'Height of pooling region.',
            },
            kernel_width: {
                type: int,
                description: 'Width of pooling region.',
            },
            stride_depth: {
                type: int,
                description: 'Stride along the depth direction',
            },
            stride_height: {
                type: int,
                description: 'Stride along the height direction.',
            },
            stride_width: {
                type: int,
                description: 'Stride along the width direction.',
            },
            padding_mode: {
                type: str,
                description: 'Option for the padding type and output blob shape.
Can be \'VALID\' , \'SAME\' , or \'CUSTOM\' .',
            },
            custom_padding_front: {
                type: int,
                description: 'Padding before the input in the depth direction.',
            },
            custom_padding_back: {
                type: int,
                description: 'Padding after the input in the depth direction.',
            },
            custom_padding_top: {
                type: int,
                description: 'Padding before the input in the height direction.',
            },
            custom_padding_bottom: {
                type: int,
                description: 'Padding after the input in the height direction.',
            },
            custom_padding_left: {
                type: int,
                description: 'Padding before the input in the width direction.',
            },
            custom_padding_right: {
                type: int,
                description: 'Padding after the input in the width direction.',
            },
            average_pooling_count_excludes_padding: {
                type: boolean,
                description: 'If true, exclude zeros from padding in average pooling.
Can only be true for AVERAGE padding.',
            },
        },
        description: 'Add a pooling layer to the model that performs spatial pooling across three dimensions.',
    },
    add_pow_broadcastable: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a pow_broadcastable layer to the model that performs element-wise power operation with broadcast support.',
    },
    add_random_bernoulli_dynamic: {
        params: [
            name,
            input_names,
            output_name,
            prob=0.5,
            seed=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            prob: {
                type: float, optional,
                description: 'Probabilities for Bernoulli distribution, default: 0.5.',
            },
            seed: {
                type: int, optional,
                description: 'Used to create a random seed for the distribution. default -1 (random).',
            },
        },
        description: 'Add a random_bernoulli_dynamic layer to the model that fills the output tensor with random values from Bernoulli distribution.',
    },
    add_random_bernoulli_like: {
        params: [
            name,
            input_name,
            output_name,
            prob=0.5,
            seed=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            prob: {
                type: float, optional,
                description: 'Probabilities for Bernoulli distribution, default: 0.5.',
            },
            seed: {
                type: int, optional,
                description: 'Used to create a random seed for the distribution. default -1 (random).',
            },
        },
        description: 'Add a random_bernoulli_like layer to the model that fills the output tensor with random values from Bernoulli distribution.',
    },
    add_random_bernoulli_static: {
        params: [
            name,
            output_name,
            output_shape,
            prob=0.5,
            seed=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            output_shape: {
                type: list of int or tuple of int,
                description: 'Target shape of the output tensor.',
            },
            prob: {
                type: float, optional,
                description: 'Probabilities for Bernoulli distribution, default: 0.5.',
            },
            seed: {
                type: int, optional,
                description: 'Used to create a random seed for the distribution. default -1 (random).',
            },
        },
        description: 'Add a random_bernoulli_static layer to the model that fills the output tensor with random values from Bernoulli distribution.',
    },
    add_random_normal_dynamic: {
        params: [
            name,
            input_names,
            output_name,
            mean=0.0,
            stddev=0.0,
            seed=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            mean: {
                type: float, optional,
                description: 'The mean of the normal distribution, default: 0.0.',
            },
            stddev: {
                type: float, optional,
                description: 'The standard deviation of the normal distribution, default: 1.0.',
            },
            seed: {
                type: int, optional,
                description: 'Used to create a random seed for the distribution. Default -1 (random).',
            },
        },
        description: 'Add a random_normal_dynamic layer to the model that fills the output tensor with random values from normal distribution.',
    },
    add_random_normal_like: {
        params: [
            name,
            input_name,
            output_name,
            mean=0.0,
            stddev=0.0,
            seed=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            mean: {
                type: float, optional,
                description: 'The mean of the normal distribution, default: 0.0.',
            },
            stddev: {
                type: float, optional,
                description: 'The standard deviation of the normal distribution, default: 1.0.',
            },
            seed: {
                type: int, optional,
                description: 'Used to create a random seed for the distribution, default -1 (random).',
            },
        },
        description: 'Add a random_normal_like layer to the model that fills the output tensor with random values from normal distribution.',
    },
    add_random_normal_static: {
        params: [
            name,
            output_name,
            output_shape,
            mean=0.0,
            stddev=0.0,
            seed=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            output_shape: {
                type: list of int or tuple of int,
                description: 'Target shape of the output tensor.',
            },
            mean: {
                type: float, optional,
                description: 'The mean of the normal distribution, default: 0.0.',
            },
            stddev: {
                type: float, optional,
                description: 'The standard deviation of the normal distribution, default: 1.0.',
            },
            seed: {
                type: int, optional,
                description: 'Used to create a random seed for the distribution. Default -1 (random).',
            },
        },
        description: 'Add a random_normal_static layer to the model that fills the output tensor with random values from normal distribution.',
    },
    add_random_uniform_dynamic: {
        params: [
            name,
            input_names,
            output_name,
            minval=0.0,
            maxval=1.0,
            seed=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            minval: {
                type: float, optional,
                description: 'Lower bound / minimum value of the uniform distribution, default: 0.0.',
            },
            maxval: {
                type: float, optional,
                description: 'Upper bound / maximum value of the uniform distribution, default: 1.0.',
            },
            seed: {
                type: int, optional,
                description: 'Used to create a random seed for the distribution. default -1 (random).',
            },
        },
        description: 'Add a random_uniform_dynamic layer to the model that fills the output tensors with random values from uniform distribution.',
    },
    add_random_uniform_like: {
        params: [
            name,
            input_name,
            output_name,
            minval=0.0,
            maxval=1.0,
            seed=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            minval: {
                type: float, optional,
                description: 'Lower bound / minimum value of the uniform distribution, default: 0.0.',
            },
            maxval: {
                type: float, optional,
                description: 'Upper bound / maximum value of the uniform distribution, default: 1.0.',
            },
            seed: {
                type: int, optional,
                description: 'Used to create a random seed for the distribution. default -1 (random).',
            },
        },
        description: 'Add a random_uniform_like layer to the model that fills the output tensors with random values from uniform distribution.',
    },
    add_random_uniform_static: {
        params: [
            name,
            output_name,
            output_shape,
            minval=0.0,
            maxval=1.0,
            seed=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            output_shape: {
                type: list of int or tuple of int,
                description: 'Target shape of the output tensor.',
            },
            minval: {
                type: float, optional,
                description: 'Lower bound / minimum value of the uniform distribution, default: 0.0.',
            },
            maxval: {
                type: float, optional,
                description: 'Upper bound / maximum value of the uniform distribution, default: 1.0.',
            },
            seed: {
                type: int, optional,
                description: 'Used to create a random seed for the distribution. default -1 (random).',
            },
        },
        description: 'Add a random_uniform_static layer to the model that fills the output tensors with random values from uniform distribution.',
    },
    add_range_dynamic: {
        params: [
            name,
            input_names,
            output_name,
            start=0,
            step=1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names.
If input size == 1: end is input, start and step are read from parameters
If input size == 2: end, start are inputs, step is read from parameters
If input size == 3: start, end, step are all inputs, none of the parameters are used.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            start: {
                type: int, optional,
                description: 'Range parameter: start. Ignored if start is provided as input, default: 0.',
            },
            step: {
                type: int, optional,
                description: 'Range parameter: step. Ignored if step is provided as input, default: 1.',
            },
        },
        description: 'Add a range_dynamic layer that returns a tensor that contains evenly spaced values.',
    },
    add_range_static: {
        params: [
            name,
            output_name,
            input_names=None,
            end=1,
            start=0,
            step=1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            end: {
                type: int, optional,
                description: 'Range parameter: end, default: 1.',
            },
            start: {
                type: int, optional,
                description: 'Range parameter: start, default: 0.',
            },
            step: {
                type: int, optional,
                description: 'Range parameter: step size, default: 1.',
            },
        },
        description: 'Add a range_static layer that returns a tensor that contains evenly spaced values.',
    },
    add_rank_preserving_reshape: {
        params: [
            name,
            input_name,
            output_name,
            output_shape,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            output_shape: {
                type: list of int or tuple of int,
                description: 'Determines the shape of the output blob.
0: copy the dimension of the input to output
-1: calculate dimensions from the rest of the shape',
            },
        },
        description: 'Add a rank_preserving_reshape layer to the model that reshapes the input tensor without altering the rank of the tensor.',
    },
    add_reduce: {
        params: [
            name,
            input_name,
            output_name,
            axis,
            mode,
            epsilon=1e-06,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: str,
                description: 'dimensions along which the reduction operation is applied.
Allowed values: ‘CHW’, ‘HW’, ‘C’, ‘H’, ‘W’',
            },
            mode: {
                type: str,
                description: 'Reduction operation to be applied.
Allowed values:
‘sum’, ‘avg’, ‘prod’, ‘logsum’, ‘sumsquare’, ‘L1’, ‘L2’, ‘max’, ‘min’, ‘argmax’.
‘argmax’ is only supported with axis values ‘C’, ‘H’ and ‘W’.',
            },
            epsilon: {
                type: float,
                description: 'number that is added to the input when ‘logsum’ function is applied.',
            },
        },
        description: 'Add a reduce layer.',
    },
    add_reduce_l1: {
        params: [
            name,
            input_name,
            output_name,
            axes=None,
            keepdims=True,
            reduce_all=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int, optional,
                description: 'List of dimensions for the reduce operations.
Each should be in range [-rank(input), rank(input)), default: None (reduce_all)',
            },
            keepdims: {
                type: bool, optional,
                description: 'Whether or not to retain the reduced dimensions with length 1, default: true.',
            },
            reduce_all: {
                type: bool, optional,
                description: 'Whether or not to reduce on all axes, default: false.',
            },
        },
        description: 'Add a reduce_l1 layer to the model that reduces the input tensor using l1_normalization(elements across given dimensions) .',
    },
    add_reduce_l2: {
        params: [
            name,
            input_name,
            output_name,
            axes=None,
            keepdims=True,
            reduce_all=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int, optional,
                description: 'List of dimensions for the reduce operations.
Each should be in range [-rank(input), rank(input)), default: None (reduce_all)',
            },
            keepdims: {
                type: bool, optional,
                description: 'Whether or not to retain the reduced dimensions with length 1, default: true.',
            },
            reduce_all: {
                type: bool, optional,
                description: 'Whether or not to reduce on all axes, default: false.',
            },
        },
        description: 'Add a reduce_l2 layer to the model that reduces the input tensor using l2_normalization(elements across given dimensions) .',
    },
    add_reduce_logsum: {
        params: [
            name,
            input_name,
            output_name,
            axes=None,
            keepdims=True,
            reduce_all=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int, optional,
                description: 'List of dimensions for the reduce operations.
Each should be in range [-rank(input), rank(input)), default: None (reduce_all)',
            },
            keepdims: {
                type: bool, optional,
                description: 'Whether or not to retain the reduced dimensions with length 1, default: true.',
            },
            reduce_all: {
                type: bool, optional,
                description: 'Whether or not to reduce on all axes, default: false.',
            },
        },
        description: 'Add a reduce_logsum layer to the model that reduces the input tensor using log(sum(elements across given dimensions)).',
    },
    add_reduce_logsumexp: {
        params: [
            name,
            input_name,
            output_name,
            axes=None,
            keepdims=True,
            reduce_all=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int, optional,
                description: 'List of dimensions for the reduce operations.
Each should be in range [-rank(input), rank(input)), default: None (reduce_all)',
            },
            keepdims: {
                type: bool, optional,
                description: 'Whether or not to retain the reduced dimensions with length 1, default: true.',
            },
            reduce_all: {
                type: bool, optional,
                description: 'Whether or not to reduce on all axes, default: false.',
            },
        },
        description: 'Add a reduce_logsumexp layer to the model that computes log(sum(exp(tensor))) and reduces along the given axis.',
    },
    add_reduce_max: {
        params: [
            name,
            input_name,
            output_name,
            axes=None,
            keepdims=True,
            reduce_all=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int, optional,
                description: 'List of dimensions for the reduce operations.
Each should be in range [-rank(input), rank(input)), default: None (reduce_all)',
            },
            keepdims: {
                type: bool, optional,
                description: 'Whether or not to retain the reduced dimensions with length 1, default: true.',
            },
            reduce_all: {
                type: bool, optional,
                description: 'Whether or not to reduce on all axes, default: false.',
            },
        },
        description: 'Add a reduce_max layer to the model that reduces the input tensor using max(elements across given dimensions) .',
    },
    add_reduce_mean: {
        params: [
            name,
            input_name,
            output_name,
            axes=None,
            keepdims=True,
            reduce_all=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int, optional,
                description: 'List of dimensions for the reduce operations.
Each should be in range [-rank(input), rank(input)), default: None (reduce_all)',
            },
            keepdims: {
                type: bool, optional,
                description: 'Whether or not to retain the reduced dimensions with length 1, default: true.',
            },
            reduce_all: {
                type: bool, optional,
                description: 'Whether or not to reduce on all axes, default: false.',
            },
        },
        description: 'Add a reduce_mean layer to the model that reduces the input tensor using mean(elements across given dimensions) .',
    },
    add_reduce_min: {
        params: [
            name,
            input_name,
            output_name,
            axes=None,
            keepdims=True,
            reduce_all=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int, optional,
                description: 'List of dimensions for the reduce operations.
Each should be in range [-rank(input), rank(input)), default: None (reduce_all)',
            },
            keepdims: {
                type: bool, optional,
                description: 'Whether or not to retain the reduced dimensions with length 1, default: true.',
            },
            reduce_all: {
                type: bool, optional,
                description: 'Whether or not to reduce on all axes, default: false.',
            },
        },
        description: 'Add a reduce_min layer to the model that reduces the input tensor using min(elements across given dimensions) .',
    },
    add_reduce_prod: {
        params: [
            name,
            input_name,
            output_name,
            axes=None,
            keepdims=True,
            reduce_all=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int, optional,
                description: 'List of dimensions for the reduce operations.
Each should be in range [-rank(input), rank(input)), default: None (reduce_all)',
            },
            keepdims: {
                type: bool, optional,
                description: 'Whether or not to retain the reduced dimensions with length 1, default: true.',
            },
            reduce_all: {
                type: bool, optional,
                description: 'Whether or not to reduce on all axes. If axes list is empty, it will
be set to true, default: false.',
            },
        },
        description: 'Add a reduce_prod layer to the model that reduces the input tensor using prod(elements across given dimensions) .',
    },
    add_reduce_sum: {
        params: [
            name,
            input_name,
            output_name,
            axes=None,
            keepdims=True,
            reduce_all=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int, optional,
                description: 'List of dimensions for the reduce operations.
Each should be in range [-rank(input), rank(input)) , default: None ( reduce_all ).',
            },
            keepdims: {
                type: bool, optional,
                description: 'Whether or not to retain the reduced dimensions with length 1, default: true.',
            },
            reduce_all: {
                type: bool, optional,
                description: 'Whether or not to reduce on all axes, default: false.',
            },
        },
        description: 'Add a reduce_sum layer to the model that reduces the input tensor using sum(elements across given dimensions) .',
    },
    add_reduce_sumsquare: {
        params: [
            name,
            input_name,
            output_name,
            axes=None,
            keepdims=True,
            reduce_all=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int, optional,
                description: 'List of dimensions for the reduce operations.
Each should be in range [-rank(input), rank(input)), default: None (reduce_all)',
            },
            keepdims: {
                type: bool, optional,
                description: 'Whether or not to retain the reduced dimensions with length 1, default: true.',
            },
            reduce_all: {
                type: bool, optional,
                description: 'Whether or not to reduce on all axes, default: false.',
            },
        },
        description: 'Add a reduce_sumsquare layer to the model that reduces the input tensor using sum(square(elements across given dimensions)) .',
    },
    add_reorganize_data: {
        params: [
            name,
            input_name,
            output_name,
            mode='SPACE_TO_DEPTH',
            block_size=2,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            mode: {
                type: str,
                description: 'If mode == ‘SPACE_TO_DEPTH’: data is moved from the spatial to the channel dimension.
Input is spatially divided into non-overlapping blocks of size block_size X block_size
and data from each block is moved to the channel dimension.
Output CHW dimensions are: [C * block_size * block_size, H/block_size, C/block_size]. If mode == ‘DEPTH_TO_SPACE’: data is moved from the channel to the spatial dimension.
Reverse of the operation ‘SPACE_TO_DEPTH’.
Output CHW dimensions are: [C/(block_size * block_size), H * block_size, C * block_size]. If mode == ‘PIXEL_SHUFFLE’:  data is moved from the channel to the spatial dimension.
Reverse of the operation ‘SPACE_TO_DEPTH’.
Output CHW dimensions are: [C/(block_size * block_size), H * block_size, C * block_size].',
            },
            block_size: {
                type: int,
                description: 'Must be greater than 1. Must divide H and W, when mode is ‘SPACE_TO_DEPTH’. (block_size * block_size)
must divide C when mode is ‘DEPTH_TO_SPACE’ or ‘PIXEL_SHUFFLE’.',
            },
        },
        description: 'Add a data reorganization layer of type “SPACE_TO_DEPTH” or “DEPTH_TO_SPACE”.',
    },
    add_reshape: {
        params: [
            name,
            input_name,
            output_name,
            target_shape,
            mode,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            target_shape: {
                type: tuple,
                description: 'Shape of the output blob. The product of target_shape must be equal
to the shape of the input blob.
Can be either length 3 (C,H,W) or length 4 (Seq,C,H,W).',
            },
            mode: {
                type: int,
                description: 'If mode == 0, the reshape layer is in CHANNEL_FIRST mode. If mode == 1, the reshape layer is in CHANNEL_LAST mode.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a reshape layer.',
    },
    add_reshape_dynamic: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a reshape_dynamic layer to the model that reshapes a tensor.',
    },
    add_reshape_like: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a reshape_like layer to the model that reshapes a tensor.',
    },
    add_reshape_static: {
        params: [
            name,
            input_name,
            output_name,
            output_shape,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            output_shape: {
                type: list of int or tuple of int,
                description: 'Target shape of the output tensor.',
            },
        },
        description: 'Add a reshape_static layer to the model that reshapes a tensor.',
    },
    add_resize_bilinear: {
        params: [
            name,
            input_name,
            output_name,
            target_height=1,
            target_width=1,
            mode='ALIGN_ENDPOINTS_MODE',
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            target_height: {
                type: int,
                description: 'Output height dimension.',
            },
            target_width: {
                type: int,
                description: 'Output width dimension.',
            },
            mode: {
                type: str,
                description: 'Following values are supported: ‘STRICT_ALIGN_ENDPOINTS_MODE’, ‘ALIGN_ENDPOINTS_MODE’, ‘UPSAMPLE_MODE’, ‘ROI_ALIGN_MODE’.
This parameter determines the sampling grid used for bilinear interpolation.',
            },
        },
        description: 'Add a resize bilinear layer to the model.',
    },
    add_reverse: {
        params: [
            name,
            input_name,
            output_name,
            reverse_dim=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            reverse_dim: {
                type: list of int or tuple of int,
                description: 'Reverse along the dimension, default [1].',
            },
        },
        description: 'Add a reverse layer to the model that reverses specific dimensions of the input tensor.',
    },
    add_reverse_sequence: {
        params: [
            name,
            input_names,
            output_name,
            batch_axis=0,
            seq_axis=-1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            batch_axis: {
                type: int, optional,
                description: 'Slices input along the dimension batch_axis, default 0.',
            },
            seq_axis: {
                type: int, optional,
                description: 'Reverse along the dimension seq_axis, default: -1.',
            },
        },
        description: 'Add a reverse sequence layer to the model that reverses variable length slices.',
    },
    add_round: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a round layer to the model that performs element-wise round operation on the input tensor that rounds the value to the nearest integer.',
    },
    add_scale: {
        params: [
            name,
            W,
            b,
            has_bias,
            input_name,
            output_name,
            shape_scale=None,
            shape_bias=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            W: {
                type: int or numpy.array,
                description: 'Scale of the input.',
            },
            b: {
                type: int or numpy.array,
                description: 'Bias to add to the input.',
            },
            has_bias: {
                type: boolean,
                description: 'Whether the bias vector of this layer is ignored in the spec .',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            shape_scale: {
                type: list of int or tuple of int,
                description: 'List of ints that specifies the shape of the scale parameter.
Can be [1] , [C] , [1,H,W] , or [C,H,W] .',
            },
            shape_bias: {
                type: list of int,
                description: 'List of ints that specifies the shape of the bias parameter
(if present). Can be [1] , [C] , [1,H,W] , or [C,H,W] .',
            },
        },
        description: 'Add a scale layer to the model.',
    },
    add_scatter: {
        params: [
            name,
            input_names,
            output_name,
            axis=0,
            mode='UPDATE',
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int,
                description: 'The axis the operation perform on, default: 0.',
            },
            mode: {
                type: str, optional,
                description: 'Scatter accumulation mode in [UPDATE | ADD | SUB | MUL | DIV | MAX | MIN], default: UPDATE.',
            },
        },
        description: 'Add a scatter layer to the model that scatters data into a new tensor according to indices from the input.',
    },
    add_scatter_along_axis: {
        params: [
            name,
            input_names,
            output_name,
            axis=0,
            mode='UPDATE',
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int,
                description: 'The axis to perform on, default: 0.',
            },
            mode: {
                type: str, optional,
                description: 'Scatter accumulation mode in [UPDATE | ADD | SUB | MUL | DIV | MAX | MIN], default: UPDATE',
            },
        },
        description: 'Add a scatter_along_axis layer to the model that scatters data into a new tensor according to indices from the input along the given axis into the output tensor.',
    },
    add_scatter_nd: {
        params: [
            name,
            input_names,
            output_name,
            mode='UPDATE',
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            mode: {
                type: str, optional,
                description: 'Scatter accumulation mode in [UPDATE | ADD | SUB | MUL | DIV | MAX | MIN], default: UPDATE',
            },
        },
        description: 'Add a scatter layer to the model that scatters data into a new tensor according to indices from input.',
    },
    add_sequence_repeat: {
        params: [
            name,
            nrep,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            nrep: {
                type: int,
                description: 'Number of repetitions of the input blob along the sequence axis.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a sequence repeat layer to the model.',
    },
    add_sign: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a sign layer to the model that performs element-wise sign operation (+1 for positive values, -1 for negative values, 0 for zeroes).',
    },
    add_simple_rnn: {
        params: [
            name,
            W_h,
            W_x,
            b,
            hidden_size,
            input_size,
            activation,
            input_names,
            output_names,
            output_all=False,
            reverse_input=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            W_h: {
                type: numpy.array,
                description: 'Weights of the recurrent layer’s hidden state.
Must be of shape (hidden_size, hidden_size) .',
            },
            W_x: {
                type: numpy.array,
                description: 'Weights of the recurrent layer’s input.
Must be of shape (hidden_size, input_size) .',
            },
            b: {
                type: numpy.array or None,
                description: 'Bias of the recurrent layer’s output. If None , bias is ignored.
Otherwise it must be of shape (hidden_size, ) .',
            },
            hidden_size: {
                type: int,
                description: 'Number of hidden units. This is equal to the number of channels of output shape.',
            },
            input_size: {
                type: int,
                description: 'Number of the number of channels of input shape.',
            },
            activation: {
                type: str,
                description: 'Activation function name. Can be one of the following option:
[ \'RELU\' , \'TANH\' , \'SIGMOID\' , \'SCALED_TANH\' , \'SIGMOID_HARD\' , \'LINEAR\' ].
See add_activation for more detailed description.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names list of this layer, in the order of [x, h_input] .',
            },
            output_names: {
                type: list of str,
                description: 'The output blob names list of this layer, in the order of [y, h_output] .',
            },
            output_all: {
                type: boolean,
                description: 'Whether the recurrent layer should output at every time step. If False, the output is the result after the final state update. If True, the output is a sequence, containing outputs at all time steps.',
            },
            reverse_input: {
                type: boolean,
                description: 'Whether the recurrent layer should process the input sequence in the reverse order. If False, the input sequence order is not reversed. If True, the input sequence order is reversed.',
            },
        },
        description: 'Add a simple recurrent layer to the model.',
    },
    add_sin: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a sin layer to the model that computes element-wise sine for the input tensor.',
    },
    add_sinh: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a sinh layer to the model that computes element-wise hyperbolic sine for the input tensor.',
    },
    add_slice: {
        params: [
            name,
            input_name,
            output_name,
            axis,
            start_index=0,
            end_index=-1,
            stride=1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: str,
                description: 'axis along which input is sliced.
allowed values: ‘channel’, ‘height’, ‘width’',
            },
            start_index: {
                type: int,
                description: 'must be non-negative.',
            },
            end_index: {
                type: int,
                description: 'negative indexing is supported.',
            },
            stride: {
                type: int,
                description: 'must be positive.',
            },
        },
        description: 'Add a slice layer.',
    },
    add_slice_by_size: {
        params: [
            name,
            input_names,
            output_name,
            axis,
            size,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int,
                description: 'axis along which input is sliced.',
            },
            size: {
                type: int,
                description: 'The size of which input will be taken',
            },
        },
        description: 'Add a slice layer.',
    },
    add_slice_dynamic: {
        params: [
            name,
            input_names,
            output_name,
            end_ids=None,
            strides=None,
            begin_masks=None,
            end_masks=None,
            squeeze_masks=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            end_ids: {
                type: list of int or tuple of int, optional,
                description: 'End offsets for slice layer, default: [1].',
            },
            strides: {
                type: list of int or tuple of int, optional,
                description: 'Strides for slice layer, default: [1].',
            },
            begin_masks: {
                type: list of bool, optional,
                description: 'Boolean masks for begin offsets, default: [false].',
            },
            end_masks: {
                type: list of bool, optional,
                description: 'Boolean masks for end offsets, default: [false].',
            },
            squeeze_masks: {
                type: list of bool, optional,
                description: 'Boolean masks for squeezing axis, default: [false].',
            },
        },
        description: 'Add a slice_dynamic layer to the model that extracts a slice of size (end - begin) / stride from the given input tensor.',
    },
    add_slice_static: {
        params: [
            name,
            input_name,
            output_name,
            begin_ids,
            end_ids,
            strides,
            begin_masks,
            end_masks,
            squeeze_masks=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            begin_ids: {
                type: list of int or tuple of int,
                description: 'Begin offsets for slice layer.',
            },
            end_ids: {
                type: list of int or tuple of int,
                description: 'End offsets for slice layer.',
            },
            strides: {
                type: list of int or tuple of int,
                description: 'Strides for slice layer.',
            },
            begin_masks: {
                type: list of bool,
                description: 'Boolean masks for begin offsets.',
            },
            end_masks: {
                type: list of bool,
                description: 'Boolean masks for end offsets.',
            },
            squeeze_masks: {
                type: list of bool,
                description: 'Boolean masks for squeezing axis.',
            },
        },
        description: 'Add a slice_static layer to the model that extracts a slice of size (end - begin) / stride from the given input tensor.',
    },
    add_sliding_windows: {
        params: [
            name,
            input_name,
            output_name,
            axis,
            window_size,
            step=1,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The of input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int,
                description: 'Axis to perform the operation.',
            },
            window_size: {
                type: int,
                description: 'Number of elements in the sliding window.',
            },
            step: {
                type: int, optional,
                description: 'The stride of the input elements in the sliding window, default: 1.',
            },
        },
        description: 'Add a sliding_windows layer to the model that returns a tensor containing all windows of size window_size * separated by step along the dimension axis .',
    },
    add_softmax: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a softmax layer to the model.',
    },
    add_softmax_nd: {
        params: [
            name,
            input_name,
            output_name,
            axis,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int,
                description: 'Axis to perform the softmax operation on.',
            },
        },
        description: 'Add a softmax_nd layer to the model that performs softmax operation along the given axis.',
    },
    add_split: {
        params: [
            name,
            input_name,
            output_names,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_names: {
                type: list of str,
                description: 'List of output blob names of this layer.',
            },
        },
        description: 'Add a split layer that uniformly splits the input along the channel dimension to produce multiple outputs.',
    },
    add_split_nd: {
        params: [
            name,
            input_name,
            output_names,
            axis,
            num_splits=2,
            split_sizes=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_names: {
                type: list of str,
                description: 'The output blob names of this layer.',
            },
            axis: {
                type: int,
                description: 'Axis to perform split on.',
            },
            num_splits: {
                type: int, optional,
                description: 'Number of splits, default: 2.',
            },
            split_sizes: {
                type: list of int or tuple of int, optional,
                description: 'List of size to split, default [] or None .',
            },
        },
        description: 'Add a split layer to the model that splits the input tensor into multiple output tensors.',
    },
    add_squeeze: {
        params: [
            name,
            input_name,
            output_name,
            axes=None,
            squeeze_all=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axes: {
                type: list of int or tuple of int, optional,
                description: 'Dimensions to perform the operation, default: None (squeeze_all).',
            },
            squeeze_all: {
                type: bool, optional,
                description: 'If true, all dimensions that are 1 are squeezed, default: false.',
            },
        },
        description: 'Add a squeeze layer to the model that decrease the rank of the input tensor by removing unit dimensions.',
    },
    add_stack: {
        params: [
            name,
            input_names,
            output_name,
            axis=0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            axis: {
                type: int, optional,
                description: 'The axis to perform stack operation, default: 0.',
            },
        },
        description: 'Add a stack layer to the model that performs stack operation on a list of tensors into one rank+1 tensor on the given axis.',
    },
    add_subtract_broadcastable: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a subtract_broadcastable layer to the model that performs element-wise subtraction operation with broadcast support.',
    },
    add_tan: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a tan layer to the model that computes element-wise tangent for the input tensor.',
    },
    add_tanh: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a tanh layer to the model that computes element-wise hyperbolic tangent for the input tensor.',
    },
    add_tile: {
        params: [
            name,
            input_name,
            output_name,
            reps=[],
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str or list[str],
                description: 'The input blob name of this layer.
If second input is provided, reps parameter is ignored.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            reps: {
                type: list of int or tuple of int,
                description: 'Number of times to replicate.
If input_name provides two inputs, second input is used as
reps and this parameter is ignored.',
            },
        },
        description: 'Add a tile layer to the model that construct a tensor by repeating the input tensor multiple number of times.',
    },
    add_topk: {
        params: [
            name,
            input_names,
            output_names,
            k=0,
            axis=0,
            use_bottom_k=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer. It must be of length 1 or 2.
The optional second input corresponds to value of K.',
            },
            output_names: {
                type: list of str,
                description: 'The output blob names of this layer. First and second correspond to
values and indices, respectively.',
            },
            k: {
                type: int, optional,
                description: 'number of values/indices to be computed along the axis.
Need not be given of there are two inputs, default: 0.',
            },
            axis: {
                type: int, optional,
                description: 'axis along which the topk values/indices are computed.
negative indexing is supported, default: 0',
            },
            use_bottom_k: {
                type: bool, optional,
                description: 'if true, bottom k values are computed instead, default: false.',
            },
        },
        description: 'Add a topk layer to the model that returns top or bottom k values and the corresponding indices of the input tensor along a given axis.',
    },
    add_transpose: {
        params: [
            name,
            axes,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            axes: {
                type: list of int or tuple of int,
                description: 'The list containing a permutation of “[0,1,2,…,N-1]” where N is the rank of input/output tensor.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a N-D transpose layer with axes as a parameter.',
    },
    add_unary: {
        params: [
            name,
            input_name,
            output_name,
            mode,
            alpha=1.0,
            shift=0,
            scale=1.0,
            epsilon=None,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            mode: {
                type: str,
                description: 'Unary function.
Allowed values: ‘sqrt’, ‘rsqrt’, ‘inverse’, ‘power’, ‘exp’, ‘log’, ‘abs’, threshold’.',
            },
            alpha: {
                type: float,
                description: 'constant used in with modes ‘power’ and ‘threshold’.',
            },
            shift, scale: {
                type: float,
                description: 'input is modified by scale and shift prior to the application of the unary function.',
            },
            epsilon: {
                type: float,
                description: 'small bias to prevent division by zero.',
            },
        },
        description: 'Add a Unary layer.',
    },
    add_unilstm: {
        params: [
            name,
            W_h,
            W_x,
            b,
            hidden_size,
            input_size,
            input_names,
            output_names,
            inner_activation='SIGMOID',
            cell_state_update_activation='TANH',
            output_activation='TANH',
            peep=None,
            output_all=False,
            forget_bias=False,
            coupled_input_forget_gate=False,
            cell_clip_threshold=50000.0,
            reverse_input=False,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            W_h: {
                type: [numpy.array],
                description: 'List of recursion weight matrices. The ordering is [R_i, R_f, R_o, R_z],
where R_i, R_f, R_o, R_z are weight matrices at input gate, forget gate, output gate and cell gate.
The shapes of these matrices are (hidden_size, hidden_size).',
            },
            W_x: {
                type: [numpy.array],
                description: 'List of input weight matrices. The ordering is [W_i, W_f, W_o, W_z],
where W_i, W_f, W_o, W_z are weight matrices at input gate, forget gate, output gate and cell gate.
The shapes of these matrices are (hidden_size, input_size).',
            },
            b: {
                type: [numpy.array] or None,
                description: 'List of biases. The ordering is [b_i, b_f, b_o, b_z],
where b_i, b_f, b_o, b_z are biases at input gate, forget gate, output gate and cell gate.
If None , biases are ignored. Otherwise the shapes of the biases are (hidden_size, ).',
            },
            hidden_size: {
                type: int,
                description: 'Number of hidden units. This is equal to the number of channels of output shape.',
            },
            input_size: {
                type: int,
                description: 'Number of the number of channels of input shape.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names list of this layer, in the order of [x, h_input, c_input].',
            },
            output_names: {
                type: list of str,
                description: 'The output blob names list of this layer, in the order of [y, h_output, c_output].',
            },
            inner_activation: {
                type: str,
                description: 'Inner activation function used at input and forget gate. Can be one of the following option:
[‘RELU’, ‘TANH’, ‘SIGMOID’, ‘SCALED_TANH’, ‘SIGMOID_HARD’, ‘LINEAR’].',
            },
            cell_state_update_activation: {
                type: str,
                description: 'Cell state update activation function used at the cell state update gate.
[‘RELU’, ‘TANH’, ‘SIGMOID’, ‘SCALED_TANH’, ‘SIGMOID_HARD’, ‘LINEAR’].',
            },
            output_activation: {
                type: str,
                description: 'Activation function used at the output gate. Can be one of the following option:
[‘RELU’, ‘TANH’, ‘SIGMOID’, ‘SCALED_TANH’, ‘SIGMOID_HARD’, ‘LINEAR’].',
            },
            peep: {
                type: [numpy.array] or None,
                description: 'List of peephole vectors. The ordering is [p_i, p_f, p_o],
where p_i, p_f, and p_o are peephole vectors at input gate, forget gate, output gate.
The shapes of the peephole vectors are (hidden_size,).',
            },
            output_all: {
                type: boolean,
                description: 'Whether the LSTM layer should output at every time step. If False, the output is the result after the final state update. If True, the output is a sequence, containing outputs at all time steps.',
            },
            forget_bias: {
                type: boolean,
                description: 'If True, a vector of 1s is added to forget gate bias.',
            },
            coupled_input_forget_gate: {
                type: boolean,
                description: 'If True, the input gate and forget gate is coupled. i.e. forget gate is not used.',
            },
            cell_clip_threshold: {
                type: float,
                description: 'The limit on the maximum and minimum values on the cell state.
If not provided, it is defaulted to 50.0.',
            },
            reverse_input: {
                type: boolean,
                description: 'Whether the LSTM layer should process the input sequence in the reverse order. If False, the input sequence order is not reversed. If True, the input sequence order is reversed.',
            },
        },
        description: 'Add a Uni-directional LSTM layer to the model.',
    },
    add_upper_triangular: {
        params: [
            name,
            input_name,
            output_name,
            k=0,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The of input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
            k: {
                type: int, optional,
                description: 'Diagonal above which to zero elements, default: 0 (main diagonal),
k < 0 is lower it and k > 0 is upper.',
            },
        },
        description: 'Add a upper_triangular layer to the model that copies a tensor setting everything outside upper triangular to zero.',
    },
    add_upsample: {
        params: [
            name,
            scaling_factor_h,
            scaling_factor_w,
            input_name,
            output_name,
            mode='NN',
            linear_upsample_mode='DEFAULT',
        ],
        param_docs: {
        },
        description: 'Add an upsample layer to the model.',
    },
    add_where_broadcastable: {
        params: [
            name,
            input_names,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_names: {
                type: list of str,
                description: 'The input blob names of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a where_broadcastable layer to the model that returns the elements either from tensor x or tensor y, depending on the value in the condition tensor.',
    },
    add_where_nonzero: {
        params: [
            name,
            input_name,
            output_name,
        ],
        param_docs: {
            name: {
                type: str,
                description: 'The name of this layer.',
            },
            input_name: {
                type: str,
                description: 'The input blob name of this layer.',
            },
            output_name: {
                type: str,
                description: 'The output blob name of this layer.',
            },
        },
        description: 'Add a where_nonzero layer to the model that returns a tensor containing the indices of all non-zero elements of input tensor.',
    },
}
