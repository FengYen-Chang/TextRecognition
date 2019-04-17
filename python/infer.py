import sys, os, time
import numpy as np
import argparse

from openvino.inference_engine import IENetwork, IEPlugin

import decoder
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def parsing():
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument('-m', '--model', default='', type=str)
    parser.add_argument('-i', '--input', default='', type=str)
    parser.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)

    return parser

def main() :
    words = ['a', 'b', 'c', 'd', 'e', 'f', 
             'g', 'h', 'i', 'j', 'k', 'l', 
             'm', 'n', 'o', 'p', 'q', 'r', 
             's', 't', 'u', 'v', 'w', 'x', 
             'y', 'z', '0', '1', '2', '3', 
             '4', '5', '6', '7', '8', '9', 
             '-']    

    args = parsing().parse_args()

    model_graph = args.model
    model_weight = args.model[:-3] + 'bin'

    net = IENetwork(model = model_graph, 
                    weights = model_weight)

    iter_inputs = iter(net.inputs)
    iter_outputs = iter(net.outputs)
   
    # inputs_num = len(net.inputs)
    # print (inputs_num)

    '''
    input_blob = []
    for _inputs in iter_inputs:
        input_blob.append(_inputs)

    output_blob = []
    for _outputs in iter_outputs:
        output_blob.append(_outputs)
    '''

    '''
    input_l = []
    for i in input_blob:
        print (net.inputs[i].shape)
        input_l.append(np.ones(shape=net.inputs[i].shape, dtype=np.float32))

    inputs = dict()
    for i in range (inputs_num):
        inputs[input_blob[i]] = input_l[i]
    '''

    input_blob = next(iter_inputs)
    output_blob = next(iter_outputs)

    if args.input == '':
        input = np.ones(shape=net.inputs[input_blob].shape, dtype = np.float32)
    else :
        b, c, h, w = net.inputs[input_blob].shape
        print (b, c, h, w)
        import cv2
        input = cv2.imread(args.input)
        print (input.shape)
        input = cv2.resize(input, (w, h))
        input = input.transpose((2, 0, 1)).reshape(1, c, h, w) 

    plugin = IEPlugin(device = 'CPU')
    exec_net = plugin.load(network = net)
    # if args.cpu_extension :
    #    plugin.add_cpu_extension(args.cpu_extension)
    # res = plugin.impl.CTCGreedyDecoder();

    

    inputs = {input_blob: np.concatenate(input, 0)}
    out = exec_net.infer(inputs)

    # print (out)
    
    print (decoder.CTCGreedyDecoder(out[output_blob], words, words[-1]))
    ''' 
    print (out[output_blob].shape)
    
    sum_ = np.sum(np.exp(out[output_blob]), 2)
    # print (sum_.shape)
    softmax_ = np.exp(out[output_blob])/sum_.reshape(25, 1, 1)
    # print (softmax_)
    # print (softmax_.shape)    

    # print (np.argmax(out[output_blob], axis=2))
    label = np.argmax(softmax_, axis=2)
    label_d = np.max(softmax_, axis=2)
    
    for i in range(25):
        print (label[i], label_d[i])

    label = label.squeeze() 
    
    output_str = ''
    output_sheets = []    
    for i in range(label.shape[0]) :
        if i > 0:
            if post != label[i]:
                output_sheets.append(post)
        post = label[i]
        if i == label.shape[0] - 1:
            output_sheets.append(post)
        
    for i in output_sheets:
        if i != 36 :
            output_str += words[i]

    print (output_str)
    print (out[output_blob].shape)
    '''
    
if "__main__" :
    main()
