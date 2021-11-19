#ifndef ABSTRACT_DATA_STRUC_CUDA_H
#define ABSTRACT_DATA_STRUC_CUDA_H

#include <vector>
#include <cstdint>

#include <thrust/device_vector.h>

#include "types.h"
#include "../common/pvector.h"
#include "abstract_data_struc.h"

class dataStrucCuda : public dataStruc {
public:        
    thrust::device_vector<float> property_c;
    dataStrucCuda(bool _weighted, bool _directed):
        dataStruc(_weighted, _directed) {}

    virtual ~dataStrucCuda(){}      
};
#endif