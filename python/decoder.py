###########################################
# 
#   Create and Implement  by John Feng
#   
#   Data :   
#       2019/04/17:       
#       Implement the CTC Greedy Decoder 
#

import numpy as np

def softmax_layer(input_tensor) :
    assert len(input_tensor.shape) == 3, "The dim of input tensor is not 3."
    # if (len(input_tensor.shape) != 3):
    #     print ("The dim of input tensor is not 3")
        
    _exp = np.exp(input_tensor)
    exp_sum = np.sum(_exp, 2).reshape((input_tensor.shape[0], input_tensor.shape[1], 1))
   
    return _exp / exp_sum

def CTCGreedyDecoder(input_tensor, text_label, blank) :
    text = ''
    pred = softmax_layer(input_tensor)

    pos = np.argmax(pred, 2).squeeze() # (25, 1, 1) -> (25)   
    last_char = len(pos)

    prev_blank = True
    for i, v in np.ndenumerate(pos) :
        char = text_label[v]
        if (char != blank):
            if (len(text) == 0 or prev_blank or char != text[-1]):
                prev_blank = False;
                text += char    
        else:
            prev_blank = True

    return text
  
 
