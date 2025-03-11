#include <iostream>
#include <fstream>
#include <vector>
#include "hdf5.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/error_reporter.h"
#include "./bands_int8_model.h"


//g++ -std=c++17 inference.cpp -o inference -ltensorflowlite -Itensorflow -Ltensorflow/bazel-bin/tensorflow/lite -rpath @executable_path/tensorflow/bazel-bin/tensorflow/lite -I/opt/homebrew/include -lhdf5



float inference_from_tflite(std::string model_path, std::vector<float> test_sample, bool quantized) {
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

    if (!model) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    // Create interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to build interpreter!" << std::endl;
        return -1;
    }

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!" << std::endl;
        return -1;
    }

    // Get the input and output details
    int input_tensor_index = interpreter->inputs()[0];
    int output_tensor_index = interpreter->outputs()[0];

    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_index);
    TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index);
    int input_height = input_tensor->dims->data[1];
    int input_width = input_tensor->dims->data[2];

    float input_scale = input_tensor->params.scale;
    int input_zero_point = input_tensor->params.zero_point;

    std::vector<int8_t> int_test_sample;

    if (quantized){
        for (int i=0; i<test_sample.size(); i++){
            float input_value = test_sample[i];
            int quantized_value = static_cast<int>(round((input_value / input_scale) + input_zero_point));
            quantized_value = std::min(std::max(quantized_value, -128), 127);
            int_test_sample.push_back(static_cast<int8_t>(quantized_value));
        }
        std::memcpy(input_tensor->data.f, int_test_sample.data(), int_test_sample.size() * sizeof(int8_t));
    }
    else{
        std::memcpy(input_tensor->data.f, test_sample.data(), test_sample.size() * sizeof(float));
    }

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke the interpreter!" << std::endl;
        return -1;
    }

    // Get the output
    float output_score; 
    if (quantized){
        std::vector<int8_t> output_data(output_tensor->bytes / sizeof(int8_t));
        std::memcpy(output_data.data(), output_tensor->data.int8, output_data.size() * sizeof(int8_t));

        int8_t output_quan_val = output_data[0];

        float output_scale = output_tensor->params.scale;
        int output_zero_point = output_tensor->params.zero_point;

        output_score = static_cast<float>(static_cast<int>(output_quan_val) - output_zero_point) * output_scale;
    }
    else{
        float* output = interpreter->typed_output_tensor<float>(0);
        output_score = * output; 
    }

    return output_score;
}


float inference_from_ccp(std::vector<float> test_sample, bool quantized) {
    
    auto model = tflite::GetModel(model_tflite);
    tflite::MutableOpResolver resolver;
    RegisterSelectedOps(&resolver);
    tflite::InterpreterBuilder builder(model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    if (!interpreter) {
        std::cerr << "Failed to build interpreter!" << std::endl;
        return -1;
    }

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!" << std::endl;
        return -1;
    }

    // Get the input and output details
    int input_tensor_index = interpreter->inputs()[0];
    int output_tensor_index = interpreter->outputs()[0];

    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_index);
    TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index);
    int input_height = input_tensor->dims->data[1];
    int input_width = input_tensor->dims->data[2];

    float input_scale = input_tensor->params.scale;
    int input_zero_point = input_tensor->params.zero_point;

    std::vector<int8_t> int_test_sample;

    if (quantized){
        for (int i=0; i<test_sample.size(); i++){
            float input_value = test_sample[i];
            int quantized_value = static_cast<int>(round((input_value / input_scale) + input_zero_point));
            quantized_value = std::min(std::max(quantized_value, -128), 127);
            int_test_sample.push_back(static_cast<int8_t>(quantized_value));
        }
        std::memcpy(input_tensor->data.f, int_test_sample.data(), int_test_sample.size() * sizeof(int8_t));
    }
    else{
        std::memcpy(input_tensor->data.f, test_sample.data(), test_sample.size() * sizeof(float));
    }

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke the interpreter!" << std::endl;
        return -1;
    }

    // Get the output
    float output_score; 
    if (quantized){
        std::vector<int8_t> output_data(output_tensor->bytes / sizeof(int8_t));
        std::memcpy(output_data.data(), output_tensor->data.int8, output_data.size() * sizeof(int8_t));

        int8_t output_quan_val = output_data[0];

        float output_scale = output_tensor->params.scale;
        int output_zero_point = output_tensor->params.zero_point;

        output_score = static_cast<float>(static_cast<int>(output_quan_val) - output_zero_point) * output_scale;
    }
    else{
        float* output = interpreter->typed_output_tensor<float>(0);
        output_score = * output; 
    }

    return output_score;
}


void get_data(const char* FILE_NAME, std::vector<float> &inference_data, std::vector<size_t>& data_dims) {
    const char* DATASET_NAME = "test";
    //Open the file
    hid_t file_id = H5Fopen(FILE_NAME, H5F_ACC_RDONLY, H5P_DEFAULT);

    if (file_id < 0) {
        std::cerr << "Error opening file: " << FILE_NAME << std::endl;
    }

    // Open the dataset using the C API function H5Dopen
    hid_t dataset_id = H5Dopen(file_id, DATASET_NAME, H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Unable to open dataset: " << DATASET_NAME << std::endl;
        H5Fclose(file_id);
    }

    hid_t dataspace_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(dataspace_id);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

    data_dims = std::vector<size_t>(dims, dims + ndims);

    // Calculate the number of elements in the dataset
    size_t num_elements = 1;
    for (int i = 0; i < ndims; ++i) {
        num_elements *= dims[i];
    }

    // Allocate memory for the dataset (assuming the data type is int)
    std::vector<float> data(num_elements);

    // Read the dataset into the buffer
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    if (status < 0) {
        std::cerr << "Unable to read dataset." << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
    }

    // Get one sample as a vector
    inference_data = data; 

    // Cleanup
    H5Dclose(dataset_id);
    H5Fclose(file_id);
}

void get_sample(std::vector<float> data, int sample_index, int dim1, int dim2, std::vector<float>& one_sample){
    std::vector<float> indexed_data(&data[sample_index*dim1*dim2],&data[(sample_index+1)*dim1*dim2]);
    one_sample = indexed_data; 
}

void saveToCSV(const std::vector<float>& vec, const std::string& filename) {
    std::ofstream outFile(filename);

    // Check if file opened successfully
    if (!outFile) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }
    outFile << "scores" << std::endl;
    for (size_t i = 0; i < vec.size(); ++i) {
        outFile << vec[i] << std::endl; 
    }
    outFile.close();
}

int main(int argc, char* argv[]) {
    std::string name = "bands_neg"; //argv[1];
    std::cout << "H5 data file: " << name << std::endl;
    const std::string file_name = name+".h5f"; std::string outfilename = name+"_cc_scores.csv"; 
    const char* FILE_NAME = file_name.c_str();
    const std::string model_path = "bands_int8_model.tflite";

    std::vector<float> inference_data; std::vector<size_t> data_dims; 

    get_data(FILE_NAME, inference_data, data_dims); 

    std::cout << "Data: " << inference_data.size() << std::endl;
    //Print dataset dimensions
    std::cout << "Dataset dimensions: ";
    size_t num_elements = 1;
    for (int i = 0; i < data_dims.size(); i++) {
        std::cout << data_dims[i] << " ";
        num_elements*=data_dims[i]; 
    }
    std::cout << "= "<< num_elements << std::endl;

    std::vector<float> test_sample; 
    std::vector<float> scores_test; 
    for (int i=0; i<data_dims[0]; i++){
        get_sample(inference_data, i, data_dims[1], data_dims[2], test_sample); 
        float output = inference_from_ccp(test_sample, true); //inference_from_tflite(model_path,test_sample, true);
        scores_test.push_back(output);
    }
    saveToCSV(scores_test, outfilename); 
    std::cout << "Inference complete! " << scores_test.size() << std::endl;
    return 0;
}