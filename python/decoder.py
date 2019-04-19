################################################
# 
#   Create and Implement  by John Feng
#   
#   Data :   
#       2019/04/17:       
#       Implement the CTC Greedy Decoder 
#       2019/04/18:
#       Implement the CTC Beam Search Decoder
#
################################################

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
    
    # print (pred.squeeze())    

    pos = np.argmax(pred, 2).squeeze() # (25, 1, 1) -> (25)   
    last_char = len(pos)

    # print (pos)

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
  
 
def __bestBeam(beam, bandwidth) :
    sorted_beam = sorted(enumerate(beam), reverse=True, key=lambda x: x[1])
    
    return (sorted_beam[:bandwidth])

def CTCBeamSearchDecoder(input_tensor, text_label, blank, bandwidth) :
    pred = softmax_layer(input_tensor)
    pred = pred.squeeze() # (t, 1, l) -> (t, l)
        
    t_step = pred.shape[0]
    idx_b = text_label.index(blank)

    _pB = {}
    _pNB = {}
    _pT = {}
    
    _init = () # init state, to make sure the first index is not blank ****

    for __t in ['c', 'l'] :
        _pB[__t] = {}
        _pNB[__t] = {}
        _pT[__t] = {}

    _pB['l'][_init] = 1
    _pNB['l'][_init] = 0
    _pT['l'][_init] = 1

    for _t in range(t_step):
        _pB['c'] = {}
        _pNB['c'] = {}
        _pT['c'] = {}
        
        if _t == 10000 :
            for _sorted  in __bestBeam(pred[_t], bandwidth):
                print ([_sorted[1]])
                _pNB['l'][((_sorted[0],),)] = _sorted[1]
                _pB['l'][((_sorted[0],),)] = 0
                _pT['l'][((_sorted[0],),)] = _sorted[1]
        else :
            for _candidate in _pNB['l']:
                _TpNB = 0
                if _candidate != _init:
                    # print (_candidate, _candidate[-1])
                    _TpNB = _pNB['l'][_candidate] * pred[_t][_candidate[-1]]
                _TpB = _pT['l'][_candidate] * pred[_t][idx_b]
                # print (_candidate, _TpNB +  _TpB)
                if _candidate in _pNB['c'] :
                    _pNB['c'][_candidate] += _TpNB
                else :
                    _pNB['c'][_candidate] = _TpNB
                _pB['c'][_candidate] = _TpB
                _pT['c'][_candidate] = _pNB['c'][_candidate] + _pB['c'][_candidate]
                # print (_pT['c'][_candidate])

                for i, v in np.ndenumerate(pred[_t]) :
                    if i < (idx_b,) :
                        extand_t = _candidate + (i,)
                        if len(_candidate) > 0 and _candidate[-1] == i:
                            _TpNB = v * _pB['l'][_candidate]
    
                        else :
                            _TpNB = v * _pT['l'][_candidate]
           
                        if extand_t in _pT['c'] :
                            _pT['c'][extand_t] += _TpNB
                            _pNB['c'][extand_t] += _TpNB
                        else :
                            _pB['c'][extand_t] = 0
                            _pT['c'][extand_t] = _TpNB
                            _pNB['c'][extand_t] = _TpNB
            
            sorted_c = sorted(_pT['c'].items(), reverse=True, key=lambda item:item[1])
            _pB['l'] = {}
            _pNB['l'] = {}
            _pT['l'] = {}
            for _sent in sorted_c[:bandwidth] :
                # print (_sent)
                _pB['l'][_sent[0]] = _pB['c'][_sent[0]]
                _pNB['l'][_sent[0]] = _pNB['c'][_sent[0]]
                _pT['l'][_sent[0]] = _pT['c'][_sent[0]]
            # print ("ddddddddss")

    res = sorted(_pT['l'].items(), reverse=True, key=lambda item:item[1])[0]
    # print (res[0])       
 
    text = ''
    for idx, _r in enumerate(res[0]) :        
        text += text_label[_r[0]]
    
    return text
