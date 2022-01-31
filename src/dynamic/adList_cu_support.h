#ifndef ADLIST_CU_SUPPORT_H_
#define ADLIST_CU_SUPPORT_H_

#include "abstract_data_struc.h"
#include "print.h"
#include "types.h"

__global__ void initProperties(int* property, int numNodes)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numNodes; i+=stride)
        property[i] = -1;
}

template <typename T>
void resizeAndCopyToCudaMemory(T* ds)
{
    if(ds->sizeOfNodesArrayOnCuda < ds->num_nodes)
    {
        int newSizeOfNodesArrayOnCuda = (ds->sizeOfNodesArrayOnCuda == 0 ? ds->num_nodes : ds->sizeOfNodesArrayOnCuda) * 2;
        int NEIGHBORS_POINTERS_SIZE = newSizeOfNodesArrayOnCuda * sizeof(NodeID*);
        int NEIGHBORS_SIZE = newSizeOfNodesArrayOnCuda * sizeof(int);
        // Modified verision using https://stackoverflow.com/questions/54297756/declare-and-initialize-array-of-arrays-in-cuda
        // create intermediate host array for storage of device row-pointers
        NodeID** h_array = (NodeID**)malloc(NEIGHBORS_POINTERS_SIZE);
        int* h_NeighborSizes = (int*)malloc(NEIGHBORS_SIZE);

        // create top-level device array pointer
        NodeID** d_array;
        cudaMalloc((void**)&d_array, NEIGHBORS_POINTERS_SIZE);

        int* d_TempNeighborSizes;
        cudaMalloc((void**)&d_TempNeighborSizes, NEIGHBORS_SIZE);

        ds->numberOfNeighborsOnCuda = 0;
        // allocate each device row-pointer, then copy host data to it
        for(size_t i = 0 ; i < ds->num_nodes ; i++){
            h_array[i];
            gpuErrchk(cudaMalloc((void**)&h_array[i], (ds->out_neighbors[i]).size() * sizeof(NodeID)));
            gpuErrchk(cudaMemcpy(h_array[i], &(((ds->out_neighbors)[i])[0]), (ds->out_neighbors[i]).size() * sizeof(NodeID), cudaMemcpyHostToDevice));
            h_NeighborSizes[i] = (ds->out_neighbors[i]).size();
            ds->numberOfNeighborsOnCuda += (ds->out_neighbors[i]).size();
        }

        // fixup top level device array pointer to point to array of device row-pointers
        cudaMemcpy(d_array, h_array, NEIGHBORS_POINTERS_SIZE, cudaMemcpyHostToDevice);
        free(h_array);

        cudaMemcpy(d_TempNeighborSizes, h_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyHostToDevice);
        free(h_NeighborSizes);
        ///////

        int* d_TempProperty;
        int PROPERTY_SIZE = newSizeOfNodesArrayOnCuda * sizeof(*ds->property_c);
        gpuErrchk(cudaMalloc(&d_TempProperty, PROPERTY_SIZE));
        
        const int BLK_SIZE = 512;
        dim3 blkSize(BLK_SIZE);
        dim3 gridSize((newSizeOfNodesArrayOnCuda + BLK_SIZE - 1) / BLK_SIZE);
        initProperties<<<gridSize, blkSize>>>(d_TempProperty, newSizeOfNodesArrayOnCuda);

        if(ds->sizeOfNodesArrayOnCuda > 0)
        {
            int OLD_NEIGHBORS_POINTERS_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(NodeID*);
            NodeID** h_array = (NodeID**)malloc(OLD_NEIGHBORS_POINTERS_SIZE);
            cudaMemcpy(h_array, ds->d_NeighborsArrays, OLD_NEIGHBORS_POINTERS_SIZE, cudaMemcpyDeviceToHost);

            for(int i = 0 ; i < ds->numberOfNodesOnCuda ; i++){
                gpuErrchkMod(cudaFree(h_array[i]), i);
            }
            cudaFree(ds->d_NeighborsArrays);
            free(h_array);

            cudaFree(ds->d_NeighborSizes);

            gpuErrchk(cudaMemcpy(d_TempProperty, ds->property_c, ds->sizeOfNodesArrayOnCuda * sizeof(*(ds->property_c)), cudaMemcpyDeviceToDevice));
            cudaFree(ds->property_c);
        }
        ds->d_NeighborsArrays = d_array;
        ds->d_NeighborSizes = d_TempNeighborSizes;
        ds->property_c = d_TempProperty;
        ds->sizeOfNodesArrayOnCuda = newSizeOfNodesArrayOnCuda;
        ds->numberOfNodesOnCuda = ds->num_nodes;

        for(NodeID i = 0; i < ds->num_nodes; i++){
            if(ds->affected[i])
            {
                ds->affectedNodes.push_back(i);
            }
        }
    }
}

template <typename T>
void copyToCudaMemory(T* ds)
{
    if(ds->numberOfNodesOnCuda < ds->num_nodes)
    {
        int numNewNodes = ds->num_nodes - ds->numberOfNodesOnCuda;
        int NEIGHBORS_POINTERS_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(NodeID*);
        int NEIGHBORS_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(int);
        NodeID** h_array = (NodeID**)malloc(NEIGHBORS_POINTERS_SIZE);
        int* h_NeighborSizes = (int*)malloc(NEIGHBORS_SIZE);

        cudaMemcpy(h_array, ds->d_NeighborsArrays, NEIGHBORS_POINTERS_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_NeighborSizes, ds->d_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyDeviceToHost);

        for(size_t i = ds->numberOfNodesOnCuda ; i < ds->num_nodes ; i++){
            cudaMalloc(&h_array[i], (ds->out_neighbors[i]).size() * sizeof(NodeID));
            gpuErrchk(cudaMemcpy(h_array[i], &((ds->out_neighbors[i])[0]), (ds->out_neighbors[i]).size() * sizeof(NodeID), cudaMemcpyHostToDevice));
            h_NeighborSizes[i] = (ds->out_neighbors[i]).size();
        }
        cudaMemcpy(ds->d_NeighborsArrays, h_array, NEIGHBORS_POINTERS_SIZE, cudaMemcpyHostToDevice);
        free(h_array);

        cudaMemcpy(ds->d_NeighborSizes, h_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyHostToDevice);
        free(h_NeighborSizes);

        ds->numberOfNodesOnCuda = ds->num_nodes;

        ds->numberOfNeighborsOnCuda = 0;
        for(NodeID i = 0; i < ds->num_nodes; i++){
            ds->numberOfNeighborsOnCuda += (ds->out_neighbors[i]).size();
            if(ds->affected[i])
            {
                ds->affectedNodes.push_back(i);
            }
        }
    }
}

template <typename T>
void updateNeighbors(T* ds)
{
    if(ds->numberOfNeighborsOnCuda < ds->num_edges)
    {
        int NEIGHBORS_POINTERS_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(NodeID*);
        int NEIGHBORS_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(int);
        NodeID** h_array = (NodeID**)malloc(NEIGHBORS_POINTERS_SIZE);
        int* h_NeighborSizes = (int*)malloc(NEIGHBORS_SIZE);

        cudaMemcpy(h_array, ds->d_NeighborsArrays, NEIGHBORS_POINTERS_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_NeighborSizes, ds->d_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyDeviceToHost);
        ds->numberOfNeighborsOnCuda = 0;
        for(size_t i = 0 ; i < ds->num_nodes ; i++){
            ds->numberOfNeighborsOnCuda += (ds->out_neighbors[i]).size();
            if(ds->affected[i])
            {
                ds->affectedNodes.push_back(i);
                free(h_array[i]);
                cudaMalloc(&h_array[i], (ds->out_neighbors[i]).size() * sizeof(NodeID));
                gpuErrchk(cudaMemcpy(h_array[i], &((ds->out_neighbors[i])[0]), (ds->out_neighbors[i]).size() * sizeof(NodeID), cudaMemcpyHostToDevice));
                h_NeighborSizes[i] = (ds->out_neighbors[i]).size();
            }
        }
        cudaMemcpy(ds->d_NeighborsArrays, h_array, NEIGHBORS_POINTERS_SIZE, cudaMemcpyHostToDevice);
        free(h_array);

        cudaMemcpy(ds->d_NeighborSizes, h_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyHostToDevice);
        free(h_NeighborSizes);
    }
}

#endif  // ADLIST_CU_SUPPORT_H_
