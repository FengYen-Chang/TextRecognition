/*******************************************************/
/*                                                     */
/*  Create and implement by Feng                       */
/*                                                     */
/*  Date:                                              */
/*      2019/04/22                                     */
/*          Define decoder functions                   */
/*                                                     */
/*******************************************************/


#pragma once

#include <string>
#include <vector>

std::string CTCGreedyDecoder(const std::vector<float> &data, 
                             const std::string& _words, 
                             char _blank, 
                             int shape);

std::string CTCBeamSearchDecoder(const std::vector<float> &data, 
                                 const std::string& _words, 
                                 char _blank, 
                                 int shape, 
                                 int bandwidth);

