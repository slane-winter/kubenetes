#--------------------------------
# Load: Standard Python Libraries
#--------------------------------

import torch.nn as nn

#--------------------------------
# Create: Layers, Decoder Block
#--------------------------------

def decode_block(in_size, out_size, k_size = 3, stride = 2, padd = 1, out_padd = 1, last_block = 0):

    """
    Purpose:
        - Create decoder blocks in a VGG network format
    Arguments:
        - in_size (int): layer input channels
        - out_size (int): layer output channels
        - k_size (int): convolution filter size
        - stride (int): convolution traversal rate
        - padd (int): convolution filter zero-padding 
        - last_layer (int): flag, indentifies last block 
        - out_padd (int): addtional padding on conovlution output
    Returns:
        - Decoder block (list)
    """

    block = []
    
    if(not(last_block)):

        block.append(nn.ConvTranspose2d(in_size, out_size, k_size, stride, padd, out_padd))
        block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())

    else:

        block.append(nn.ConvTranspose2d(in_size, out_size, k_size, stride, padd, out_padd))
    
    return block

#--------------------------------
# Create: Layers , Encoder Block
#--------------------------------


def encode_block(in_size, out_size, k_size = 3, stride = 1, padd = 1, last_block = 0):
 
    """
    Purpose:
        - Create encoder blocks in a VGG network format
    Arguments:
        - Create: Layers, Encoder Block
        - in_size (int): layer input channels
        - out_size (int): layer output channels
        - k_size (int): convolution filter size
        - stride (int): convolution traversal rate
        - padd (int): convolution filter zero-padding 
        - last_layer (int): flag, indentifies last block 
    Returns:
        - Encoder block (list)
    """
 
    block = []

    block.append(nn.Conv2d(in_size, out_size, k_size, stride, padd))
    block.append(nn.BatchNorm2d(out_size))
    block.append(nn.ReLU())

    block.append(nn.Conv2d(out_size, out_size, k_size, stride, padd))
    block.append(nn.BatchNorm2d(out_size))
    block.append(nn.ReLU())

    # stride = 2 
    #block.append(nn.Conv2d(out_size, out_size, k_size, stride, padd))

    block.append(nn.MaxPool2d(2, 2))
    block.append(nn.Dropout(0.2))

    return block

#--------------------------------
# Create: Architecture, Decoder
#--------------------------------

def create_decoder(data_size, in_size, using_standard = 0):

    """
    Purpose:
        - Create decoder architecture
    Arguments:
        - data_size (tuple): data sample dimensions
        - in_size (int): initial input feature size
    Returns:
        - Decoder architecture (nn.Sequential)
    """

    channels, max_size, _, = data_size

    # Calculate: Number Decoder Blocks
    # - Assuming a construction from [N, C, 1, 1] shape

    f_size = 1
    num_blocks = 0
    while(f_size <= max_size):
        f_size = f_size * 2
        num_blocks = num_blocks + 1     

    num_blocks = num_blocks - 1 
    
    # Construct: Network Decoder

    out_size = in_size

    layers = []

    last_one = 0

    for i in range(num_blocks):

        if(i != 0):

            in_size = in_size // 2

        out_size = out_size // 2

        if(i == (num_blocks - 1)):

            out_size = channels

            # ***** EXPERIMENTAL ******

            # out_size = channels * 2

            # *************************        

            last_one = 1

        [layers.append(ele) for ele in decode_block(in_size, out_size, last_block = last_one)]

    if(using_standard):
        layers.append(nn.Tanh())
    else:
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)  

#--------------------------------
# Create: Architecture, Encoder
#--------------------------------

def create_encoder(data_size, out_size = 32, min_size = 5, max_size = 512):

    """
    Purpose:
        - Create encoder architecture
    Arguments:
        - data_size (tuple): data sample dimensions
        - out_size (int): initial output channel size
        - min_size (int): threshold for feature reduction
        - max_size (int): threshold for feature expansion
    Returns:
        - Encoder architecture (nn.Sequential)
    """

    in_size, p_k_size, _ = data_size

    # Calculate: Pooling Size & Number Encoder Blocks
    # - Pooling is invoked as global average pooling

    num_blocks = 1
    while(p_k_size >= min_size):
        p_k_size = p_k_size // 2
        num_blocks = num_blocks + 1     

    num_blocks = num_blocks + 1

    # Construct: Network Encoder

    layers = []
    
    last_one = 0

    for i in range(num_blocks):

        if(i != 0):

            in_size = out_size
            
        out_size = out_size * 2

        if(out_size >= max_size):

            out_size = max_size

        if(i == (num_blocks - 1)):

            layers.append(nn.AvgPool2d(p_k_size))
        
        elif(i == (num_blocks - 2)):

            last_one = 1

        else:

            [layers.append(ele) for ele in encode_block(in_size, out_size, last_block = last_one)]

    return nn.Sequential(*layers)


