/*******************************************************/
/*                                                     */
/*  Create and implement by Feng                       */
/*                                                     */
/*  Date:                                              */
/*      2019/04/22                                     */
/*          Implement the decoder functions            */
/*                                                     */
/*******************************************************/

#include "decoder.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <climits>

void softmax_layer(std::vector<float> data, std::vector<float> &prob, int shape)
{
    float sum = 0;
    std::vector<float> exp = std::vector<float>(shape, 0.f);

    for (int i = 0; i < shape; i++)
    {
        exp[i] = std::exp(data[i]);
        sum += exp[i];
    }

    for (int i = 0; i < shape; i++)
    {
        prob[i] = exp[i] / sum;
    }
}

//std::string 
std::string CTCGreedyDecoder(const std::vector<float> &data, const std::string& _words, char _blank, int shape)
{
    std::string text = "";    

    const int label_size = _words.length();

    std::vector<float> prob = std::vector<float>(label_size, 0.f);    

    bool prev_blank = true;

    for(int __t = 0; __t < shape; __t++)
    {
        std::vector<float> process_data(data.begin() + (__t * label_size), data.begin() + ((__t + 1) * label_size));

        softmax_layer(process_data, prob, label_size);
        int argmax = std::distance(prob.begin(), std::max_element(prob.begin(), prob.end()));
        auto __char = _words[argmax];
        
        if (__char != _blank)
        {
            if (text.empty() || prev_blank || __char != text.back())
            {
                text += __char;
                prev_blank = false;
            }
        }
        else
        {
            prev_blank = true;
        }
    }
    
    return text;
}

class beam
{
public:
    beam(std::string _text, float _pB, float _pNB, float _pT)
    {
        sentance = _text;
        pB = _pB;
        pNB = _pNB;
        pT = _pT;
    };
public :
    std::string sentance;
    float pB;
    float pNB;
    float pT;
};

std::string CTCBeamSearchDecoder(const std::vector<float> &data, const std::string& _words, char _blank, int shape, int bandwidth)
{
    std::string text = "";
    
    const int label_size = _words.length();
    std::vector<float> prob = std::vector<float>(label_size, 0.f);

    std::vector<beam> curr;
    std::vector<beam> last;

    beam init("", 1.f, 0.f, 1.f);
    
    last.push_back(init);

    for (int __t = 0; __t < shape; __t++)
    {
        curr.clear();
        std::vector<float> process_data(data.begin() + (__t * label_size), data.begin() + ((__t + 1) * label_size));
        softmax_layer(process_data, prob, label_size);
        
        for (int _candidate = 0; _candidate < last.size(); _candidate++)
        {
            float _pNB = 0.f;
            auto __can = last[_candidate];
            std::string __can_sentance = __can.sentance;
            if (__can_sentance != "")
            {   
                int n = (int)(__can_sentance.back());
                _pNB = __can.pNB * prob[n];
            }
            float _pB = __can.pT * prob[(label_size - 1)];
    
            auto check_res = std::find_if(curr.begin(), curr.end(), [__can_sentance](beam const& n)
            {
                return n.sentance == __can_sentance;
            });
            if (check_res == std::end(curr))
            {
                curr.push_back(beam(__can.sentance, _pB, _pNB, _pB + _pNB));
            }
            else
            {
                auto __i = std::distance(curr.begin(), check_res);
                curr[__i].pNB += _pNB;
                curr[__i].pB = _pB;
                curr[__i].pT = curr[__i].pB + curr[__i].pNB;
            }

            for (int i = 0; i < label_size - 1; i++)
            {
                auto extand_t = __can_sentance + (char)i;
                if (__can_sentance.length() > 0 && __can.sentance.back() == (char)i)
                {
                    _pNB = prob[i] * __can.pB;
                }
                else
                {
                    _pNB = prob[i] * __can.pT;
                }
                
                auto check_res = std::find_if(curr.begin(), curr.end(), [extand_t](beam const& n)
                {
                    return n.sentance == extand_t;
                });

                if (check_res == std::end(curr))
                {
                    curr.push_back(beam(extand_t, 0.f, _pNB, _pNB));
                }
                else
                {
                    auto __i = std::distance(curr.begin(), check_res);
                    curr[__i].pNB += _pNB;
                    curr[__i].pT += _pNB;
                }
            }            
        }

        sort(curr.begin(), curr.end(), [](const beam &a, const beam &b) -> bool
        {
            return a.pT > b.pT;
        });

        last.clear();
        for (int _b = 0; _b < bandwidth; _b++)
        {
            last.push_back(curr[_b]);
        }
    }

    auto idx = last[0].sentance;
    for (int _idx = 0; _idx < idx.length(); _idx++)
    {
        text += _words[(int)(idx[_idx])];
    }

    return text;
}





