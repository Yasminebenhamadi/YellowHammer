#include <vector>
#include <iostream>
#include <fstream>
#include "hdf5.h"
//#include "./butterworth.c"

//g++ -std=c++17 preprocess.cpp -o preprocess -I/opt/homebrew/include -lhdf5


typedef struct {
    float xv[11];
    float yv[11];
} BW_filter_t;

typedef struct {
    float gain;
    float yc[2];
} BW_filterCoefficients_t;

void Butterworth_initialise(BW_filter_t *filter) {
    for (uint32_t i = 0; i < 11; i += 1) {
        filter->xv[i] = 0.0f;
        filter->yv[i] = 0.0f;
    }
}

const std::vector<float> a = {0.24748712,  -1.01459797,   3.25044176,  -6.60480357,
         11.17397924, -13.96354968,  14.75151685, -11.52428018,
          7.49511809,  -3.10514426,   1.};

const std::vector<float> b = {-0.00023953,  0.        ,  0.00119764,  0.        , -0.00239528,
         0.        ,  0.00239528,  0.        , -0.00119764,  0.        ,
         0.00023953};



float Butterworth_applyBandPassFilter(float sample, BW_filter_t*filter){
    //Shift x buffer
    for (int i=0; i< b.size() - 1; i++){
        filter->xv[i]=filter->xv[i+1];
    }
    filter->xv[b.size() - 1]= sample;

    // sum(x[n-k]*b[k])
    float feed_fwd=0;
    for (int i=0; i< b.size(); i++){
        feed_fwd +=b[i]*filter->xv[i];
    }

    //Shift y buffer
    for (int i=0; i< a.size() - 1; i++){
        filter->yv[i]=filter->yv[i+1];
    }

    //sum(y[n-k]*a[k])
    float feed_bck=0;
    for (int i=0; i< a.size() - 1; i++){
        feed_bck +=a[i]*filter->yv[i];
    }

    filter->yv[a.size() - 1] = feed_fwd - feed_bck; 
    return filter->yv[a.size() - 1]; 
}

void filter_audio(std::vector<float> audio, std::vector<float> &filtered_audio){
    BW_filter_t filter_buffer;
    Butterworth_initialise(&filter_buffer);
    for(auto sample:audio){
        float y_filter = Butterworth_applyBandPassFilter(sample, &filter_buffer);
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
    const char* FILE_NAME = "audio_1.h5";
    std::vector<float> audio_data; std::vector<float> filtered_audio; std::vector<float> filtered_audio_2;

    get_audio(FILE_NAME, audio_data);

    std::cout << "here... " << std::endl;

    filter_audio(audio_data, filtered_audio);

    filter_audio(filtered_audio, filtered_audio_2);

    //for (auto y:filtered_audio)
        //std::cout << y << " " << std::endl;

    std::string outfilename = "filt_audio_cpp.csv"; 
    saveToCSV(filtered_audio, outfilename);

    std::string outfilename_2 = "filt_audio_2_cpp.csv"; 
    saveToCSV(filtered_audio_2, outfilename_2);
    
    return 0;
}