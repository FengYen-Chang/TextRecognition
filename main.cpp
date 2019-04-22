/*******************************************************/
/*                                                     */
/*  Create and implement by Feng                       */
/*                                                     */
/*  Date:                                              */
/*      2019/04/22                                     */
/*          Implement the main function                */
/*                                                     */
/*******************************************************/


#include <iomanip>
#include <memory>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <ext_list.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"

#include "src/argparse.h"
// #include "src/decoder.h"

using namespace InferenceEngine;
using namespace std;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_ni < 1) {
        throw std::logic_error("Parameter -ni should be greater than zero (default 1)");
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

int main(int argc, char *argv[])
{
    // ------------------------------ Parsing and validation of input args ---------------------------------
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }

    auto engine_Ptr = PluginDispatcher({"./lib", ""}).getPluginByDevice(FLAGS_d);

    InferencePlugin plugin(engine_Ptr);

    if (FLAGS_d.find("CPU") != std::string::npos) {
        /**
         * cpu_extensions library is compiled from "extension" folder containing
         * custom MKLDNNPlugin layer implementations. These layers are not supported
         * by mkldnn, but they can be useful for inferring custom topologies.
        **/
        plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    }

    const char _blank = '-';
    std::string _words = std::string("abcdefghijklmnopqrstuvwxyz1234567890") + _blank;

    CNNNetReader network_reader;

    slog::info << "device = " << FLAGS_d << slog::endl;
    
    network_reader.ReadNetwork(FLAGS_m);
    std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
    network_reader.ReadWeights(binFileName);
    
    network_reader.getNetwork().setBatchSize(1);
    CNNNetwork network = network_reader.getNetwork();

    auto input_info = network.getInputsInfo().begin()->second;
    auto input_name = network.getInputsInfo().begin()->first;
    input_info->setInputPrecision(Precision::U8);

    size_t inputH = input_info->getDims()[0];
    size_t inputW = input_info->getDims()[1];

    auto output_info = network.getOutputsInfo().begin()->second;
    auto output_name = network.getOutputsInfo().begin()->first;
    output_info->setPrecision(Precision::FP32);

    auto executable_network = plugin.LoadNetwork(network, {});
    auto infer_request = executable_network.CreateInferRequest();

    auto input = infer_request.GetBlob(input_name);
    
    cv::Mat image = cv::imread(FLAGS_i);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(inputW, inputW));
    
    auto input_data = input->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
    
    size_t channels_number = input->dims()[2];
    size_t image_size = inputW * inputH;
    
    for(size_t pid = 0; pid < image_size; pid++)
    {
        for (size_t ch = 0; ch < channels_number; ch++)
        {
            input_data[ch * image_size + pid] = image.at<cv::Vec3b>(pid)[ch];
        }        
    }

    infer_request.Infer();

    auto output = infer_request.GetBlob(output_name);
    auto output_data = output->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
   
    std::cout << output_data << std::endl;
    
    std::cout << "Decode by CTC Greedy Decoder: \n";
    // auto res_greedy =  CTCGreedyDecoder(output_data, _words, _blank, &conf);
    // std::cout << res_greedy << "\n";
 
    return 0;
}

