#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>

typedef struct {
    float xv[11];
    float yv[11];
} BW_filter_t;

void Butterworth_initialise(BW_filter_t *filter) {
    for (uint32_t i = 0; i < 11; i += 1) {
        filter->xv[i] = 0.0f;
        filter->yv[i] = 0.0f;
    }
}

const std::vector<float> a_band1 = {0.15321141, -0.5377237 ,  1.8085212 , -3.52529355,  6.34467972,
        -7.95807998,  9.18825376, -7.42000549,  5.52502756, -2.42000723,
         1.        };

const std::vector<float> a_band2 = {0.15321141, 0.26168753, 1.22782599, 1.51382384, 3.70608462,
        3.28072692, 5.35124787, 3.1793689 , 3.7219365 , 1.17771582,
        1.        };

const std::vector<float> a_band3 = {0.15321141,  1.23019342,  5.03014405, 13.32784841, 25.13192442,
        35.0384834 , 36.5651516 , 28.23264406, 15.52834643,  5.5364437 ,
         1.        };

//IDK why but all b coefficient are the same for the 3 filters
const std::vector<float> b = {-0.00084414,  0.        ,  0.00422071,  0.        , -0.00844141,
         0.        ,  0.00844141,  0.        , -0.00422071,  0.        ,
         0.00084414};


float Butterworth_applyBandPassFilter(std::vector<float> a, std::vector<float> b, float sample, BW_filter_t*filter){
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
void lfilt_envelope(std::vector<float> a, std::vector<float> b, std::vector<float> audio, std::vector<float> &filtered_audio, std::vector<float> &envelope){
    const int frame_length=1048; 
    const int hop_length=128; 
    BW_filter_t filter_buffer; 
    Butterworth_initialise(&filter_buffer);

    audio.insert(audio.begin(), int(frame_length/2), 0);
    audio.insert(audio.end(), int(frame_length/2), 0);

    int count=0; int count_frame=0;
    for(auto sample:audio){
        float y_filter = Butterworth_applyBandPassFilter(a,b, sample, &filter_buffer);
        filtered_audio.push_back(y_filter);

        if (count % hop_length == 0 && count+frame_length < audio.size()){
            envelope.push_back(y_filter*y_filter);
        }
        
        for (int j=count_frame; j< div(count, hop_length).quot; j++){
            envelope[j]+=y_filter*y_filter;
        }
        
        if (count - count_frame * hop_length == frame_length){
            envelope[count_frame] =  sqrt(envelope[count_frame]/frame_length);
            count_frame++;
        }
        count ++;
    }

    //Normalize
    float min = *std::min_element(envelope.begin(), envelope.end());
    float max = *std::max_element(envelope.begin(), envelope.end());
    for (int j=0; j < envelope.size(); j++){
        envelope[j] =  (envelope[j] - min) / (max-min);
    }
}

void process_clip(std::vector<float> audio_data, std::vector<float>& processed_data){

    std::vector<float> envelope_1, envelope_2, envelope_3; 
    std::vector<float> filtered_audio_1, filtered_audio_2, filtered_audio_3; //TODO remove this

    lfilt_envelope(a_band1, b, audio_data, filtered_audio_1, envelope_1);
    lfilt_envelope(a_band2, b, audio_data, filtered_audio_2, envelope_2);
    lfilt_envelope(a_band3, b, audio_data, filtered_audio_3, envelope_3);

    for (int i=0; i<envelope_1.size(); i++){
        processed_data.push_back(envelope_1[i]);
        processed_data.push_back(envelope_2[i]);
        processed_data.push_back(envelope_3[i]);
    }
}