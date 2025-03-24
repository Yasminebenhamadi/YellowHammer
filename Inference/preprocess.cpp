#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include "hdf5.h"
#include "./bands_preprocess.cpp"
//#include "./inference.cpp"
//#include "./butterworth.c"

//g++ -std=c++17 preprocess.cpp -o preprocess -I/opt/homebrew/include -lhdf5
//g++ -std=c++17 preprocess.cpp -o preprocess -ltensorflowlite -Itensorflow -Ltensorflow/bazel-bin/tensorflow/lite -rpath @executable_path/tensorflow/bazel-bin/tensorflow/lite -I/opt/homebrew/include -lhdf5


void filter_audio(std::vector<float> a, std::vector<float> b, std::vector<float> audio, std::vector<float> &filtered_audio){
    BW_filter_t filter_buffer;
    Butterworth_initialise(&filter_buffer);
    for(auto sample:audio){
        float y_filter = Butterworth_applyBandPassFilter(a,b,sample, &filter_buffer);
        filtered_audio.push_back(y_filter);
    }
}

void get_audio(const char* FILE_NAME, std::vector<float> &audio_data) {
    const char* DATASET_NAME = "audio";
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
    audio_data = data; 

    // Cleanup
    H5Dclose(dataset_id);
    H5Fclose(file_id);
}

void saveToCSV(const std::vector<float>& vec, const std::string& filename) {
    std::ofstream outFile(filename);

    // Check if file opened successfully
    if (!outFile) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }
    outFile << "audio" << std::endl;
    for (size_t i = 0; i < vec.size(); ++i) {
        outFile << vec[i] << std::endl; 
    }
    outFile.close();
}

int main(){
    const char* FILE_NAME = "clip_1_20240726_forest_6_5m_0_9_test1_YH_4_D_7.h5f";
    std::vector<float> audio_data; std::vector<float> filtered_audio; std::vector<float> filtered_audio_2;

    get_audio(FILE_NAME, audio_data);

    std::cout << "here... " << std::endl;

    std::string outfilename = "band1_envelope.csv"; 

    std::string outfilename_2 = "band1.csv"; 
    
    std::vector<float> band1; std::vector<float> band1_envelope;
    lfilt_envelope(a_band1, b, audio_data, band1, band1_envelope);
    saveToCSV(band1_envelope, outfilename);
    saveToCSV(band1, outfilename_2);    
    return 0;
}