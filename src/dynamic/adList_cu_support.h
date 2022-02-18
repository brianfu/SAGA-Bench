#ifndef ADLIST_CU_SUPPORT_H_
#define ADLIST_CU_SUPPORT_H_

#include "abstract_data_struc.h"
#include "print.h"
#include "types.h"
#include <algorithm>

__global__ void initProperties(int* property, int numNodes)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    int curr;

    for(int i = idx; i < numNodes; i+=stride)
    {
        property[i] = -1;
        curr = property[i];
    }
}

__global__ void copyToCuda(NodeID* d_affectedNodes, int* d_copySize, bool* copyFullOrDelta, Node* d_coalesceNeighbors, int coalesceSize, int numAffectedNodes, Node** d_NeighborsArrays, int* d_NeighborSizes)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if(idx < numAffectedNodes)
    {
        NodeID node = d_affectedNodes[idx];
        int copyStart = d_copySize[idx];
        int copyEnd = (idx + 1) < numAffectedNodes ? d_copySize[idx+1] : coalesceSize;
        int offset = copyFullOrDelta[idx] ? 0 : d_NeighborSizes[node];
        memcpy(d_NeighborsArrays[node] + offset, d_coalesceNeighbors + copyStart, (copyEnd - copyStart) * sizeof(Node));
    }
}

template <typename T>
void coalesceEdgesAndCopyToCuda(T* ds, bool* copyFullOrDelta)
{
    std::vector<Node> coalesceNeighbors;
    std::vector<int> copySize;
    int numNodes = ds->affectedNodes.size();
    int start = 0;
    for(int i=0; i < numNodes; i++)
    {
        NodeID node = ds->affectedNodes[i];
        if (copyFullOrDelta[i])
        {
            coalesceNeighbors.insert(coalesceNeighbors.end(), ds->out_neighbors[node].begin(), ds->out_neighbors[node].end());
            copySize.push_back(start);
            start += ds->out_neighbors[node].size();
        }
        else
        {
            coalesceNeighbors.insert(coalesceNeighbors.end(), ds->out_neighborsDelta[node].begin(), ds->out_neighborsDelta[node].end());
            copySize.push_back(start);
            start += ds->out_neighborsDelta[node].size();
        }
    }

    int coalesceSize = coalesceNeighbors.size();
    Node* d_coalesceNeighbors;
    int* d_copySize;
    NodeID* d_affectedNodes;
    bool* d_copyFullOrDelta;

    cudaMalloc(&d_coalesceNeighbors, coalesceSize * sizeof(Node));
    cudaMemcpy(d_coalesceNeighbors, &(coalesceNeighbors[0]), coalesceSize * sizeof(Node), cudaMemcpyHostToDevice);
    cudaMalloc(&d_copySize, copySize.size() * sizeof(int));
    cudaMemcpy(d_copySize, &(copySize[0]), copySize.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_affectedNodes, ds->affectedNodes.size() * sizeof(NodeID));
    cudaMemcpy(d_affectedNodes, &(ds->affectedNodes[0]), ds->affectedNodes.size() * sizeof(NodeID), cudaMemcpyHostToDevice);
    cudaMalloc(&d_copyFullOrDelta, ds->affectedNodes.size() * sizeof(bool));
    cudaMemcpy(d_copyFullOrDelta, copyFullOrDelta, ds->affectedNodes.size() * sizeof(bool), cudaMemcpyHostToDevice);

    const int BLK_SIZE = 512;
    dim3 blkSize(BLK_SIZE);
    dim3 gridSize((ds->affectedNodes.size() + BLK_SIZE - 1) / BLK_SIZE);
    copyToCuda<<<gridSize, blkSize>>>(d_affectedNodes, d_copySize, d_copyFullOrDelta, d_coalesceNeighbors, coalesceSize, ds->affectedNodes.size(), ds->d_NeighborsArrays, ds->d_NeighborSizes);
    cudaDeviceSynchronize();
    cudaFree(d_coalesceNeighbors);
    cudaFree(d_copySize);
    cudaFree(d_affectedNodes);
    cudaFree(d_copyFullOrDelta);
}

template <typename T>
void resizeAndCopyToCudaMemory(T* ds)
{
    if(ds->sizeOfNodesArrayOnCuda < ds->num_nodes)
    {
        int newSizeOfNodesArrayOnCuda = (ds->sizeOfNodesArrayOnCuda == 0 ? ds->num_nodes : ds->sizeOfNodesArrayOnCuda) * 2;
        int NEIGHBORS_POINTERS_SIZE = newSizeOfNodesArrayOnCuda * sizeof(Node*);
        int NEIGHBORS_SIZE = newSizeOfNodesArrayOnCuda * sizeof(int);
        // Modified verision using https://stackoverflow.com/questions/54297756/declare-and-initialize-array-of-arrays-in-cuda
        // create intermediate host array for storage of device row-pointers
        Node** h_array = (Node**)malloc(NEIGHBORS_POINTERS_SIZE);
        int* h_NeighborSizes = (int*)malloc(NEIGHBORS_SIZE);
        memset(h_NeighborSizes, 0, NEIGHBORS_SIZE);
        int* h_NeighborCapacity = (int*)malloc(NEIGHBORS_SIZE);
        memset(h_NeighborCapacity, 0, NEIGHBORS_SIZE);

        // create top-level device array pointer
        Node** d_array;
        cudaMalloc((void**)&d_array, NEIGHBORS_POINTERS_SIZE);

        int* d_TempNeighborSizes;
        cudaMalloc((void**)&d_TempNeighborSizes, NEIGHBORS_SIZE);


        int* d_TempProperty;
        int PROPERTY_SIZE = newSizeOfNodesArrayOnCuda * sizeof(*ds->property_c);
        gpuErrchk(cudaMalloc(&d_TempProperty, PROPERTY_SIZE));
        
        const int BLK_SIZE = 512;
        dim3 blkSize(BLK_SIZE);
        dim3 gridSize((newSizeOfNodesArrayOnCuda + BLK_SIZE - 1) / BLK_SIZE);
        initProperties<<<gridSize, blkSize>>>(d_TempProperty, newSizeOfNodesArrayOnCuda);

        ds->numberOfNeighborsOnCuda = 0;
        if(ds->sizeOfNodesArrayOnCuda == 0)
        {
            // allocate each device row-pointer, then copy host data to it
            #pragma omp for schedule(dynamic, 16)
            for(size_t i = 0 ; i < ds->num_nodes ; i++){
                gpuErrchk(cudaMalloc((void**)&h_array[i], (ds->out_neighbors[i]).size() * sizeof(Node) * 2));
                gpuErrchk(cudaMemcpyAsync(h_array[i], &(((ds->out_neighbors)[i])[0]), (ds->out_neighbors[i]).size() * sizeof(Node), cudaMemcpyHostToDevice));
                h_NeighborSizes[i] = (ds->out_neighbors[i]).size();
                h_NeighborCapacity[i] = (ds->out_neighbors[i]).size() * 2;

                // ds->numberOfNeighborsOnCuda += (ds->out_neighbors[i]).size();
                // if(ds->affected[i])
                // {
                //     ds->affectedNodes.push_back(i);
                // }
            }
        }
        else
        {
            int OLD_NEIGHBORS_POINTERS_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(Node*);
            int OLD_NEIGHBORS_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(int);
            
            std::copy(ds->h_NeighborsArrays, ds->h_NeighborsArrays + ds->sizeOfNodesArrayOnCuda, h_array);
            std::copy(ds->h_NeighborSizes, ds->h_NeighborSizes + ds->sizeOfNodesArrayOnCuda, h_NeighborSizes);
            std::copy(ds->h_NeighborCapacity, ds->h_NeighborCapacity + ds->sizeOfNodesArrayOnCuda, h_NeighborCapacity);

            #pragma omp for schedule(dynamic, 16)
            for(NodeID i : ds->affectedNodesSet) {
            // for(size_t i = 0 ; i < ds->num_nodes ; i++){
            //     if(ds->affected[i])
            //     {
                    // ds->affectedNodes.push_back(i);
                    if(i < ds->numberOfNodesOnCuda)
                    {
                        cudaFree(h_array[i]);
                    }
                    if(h_NeighborCapacity[i] < (ds->out_neighbors[i]).size())
                    {
                        h_NeighborCapacity[i] = ((h_NeighborCapacity[i] * 2 < (ds->out_neighbors[i]).size()) ? (ds->out_neighbors[i]).size() : h_NeighborCapacity[i]) * 2;
                    }
                    cudaMalloc(&h_array[i], (h_NeighborCapacity[i] * sizeof(Node)));
                    gpuErrchk(cudaMemcpyAsync(h_array[i], &((ds->out_neighbors[i])[0]), (ds->out_neighbors[i]).size() * sizeof(Node), cudaMemcpyHostToDevice));
                    h_NeighborSizes[i] = (ds->out_neighbors[i]).size();
                // }
                // ds->numberOfNeighborsOnCuda += (ds->out_neighbors[i]).size();
            }
            cudaFree(ds->d_NeighborsArrays);
            cudaFree(ds->d_NeighborSizes);
            free(ds->h_NeighborSizes);
            free(ds->h_NeighborCapacity);
            free(ds->h_NeighborsArrays);

            gpuErrchk(cudaMemcpyAsync(d_TempProperty, ds->property_c, ds->sizeOfNodesArrayOnCuda * sizeof(*(ds->property_c)), cudaMemcpyDeviceToDevice));
            cudaFree(ds->property_c);
        }
        // fixup top level device array pointer to point to array of device row-pointers
        cudaMemcpy(d_array, h_array, NEIGHBORS_POINTERS_SIZE, cudaMemcpyHostToDevice);

        cudaMemcpy(d_TempNeighborSizes, h_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyHostToDevice);

        ds->h_NeighborsArrays = h_array;
        ds->h_NeighborSizes = h_NeighborSizes;
        ds->h_NeighborCapacity = h_NeighborCapacity;
        ds->d_NeighborsArrays = d_array;
        ds->d_NeighborSizes = d_TempNeighborSizes;
        ds->property_c = d_TempProperty;
        ds->sizeOfNodesArrayOnCuda = newSizeOfNodesArrayOnCuda;
        ds->numberOfNodesOnCuda = ds->num_nodes;
        ds->numberOfNeighborsOnCuda = ds->num_edges;
    }
}

template <typename T>
void copyToCudaMemory(T* ds)
{
    if(ds->numberOfNodesOnCuda < ds->num_nodes)
    {
        int NEIGHBORS_POINTERS_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(Node*);
        int NEIGHBORS_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(int);
        // Node** h_array = (Node**)malloc(NEIGHBORS_POINTERS_SIZE);
        // int* h_NeighborSizes = (int*)malloc(NEIGHBORS_SIZE);
        bool copyFullOrDelta[ds->affectedNodesSet.size()] = {false};
        int index = 0;

        // std::copy(h_array, ds->d_NeighborsArrays, NEIGHBORS_POINTERS_SIZE, cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_NeighborSizes, ds->d_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyDeviceToHost);

        // ds->numberOfNeighborsOnCuda = 0;
        // #pragma omp for schedule(dynamic, 16)
        for(NodeID i : ds->affectedNodesSet) {
        // for(size_t i = 0 ; i < ds->num_nodes ; i++){
            // if(ds->affected[i])
            // {
                // ds->affectedNodes.push_back(i);
                if(ds->h_NeighborCapacity[i] < (ds->out_neighbors[i]).size())
                {
                    if(i < ds->numberOfNodesOnCuda)
                        cudaFree(ds->h_NeighborsArrays[i]);
                    ds->h_NeighborCapacity[i] = ((ds->h_NeighborCapacity[i] * 2 < (ds->out_neighbors[i]).size()) ? (ds->out_neighbors[i]).size() : ds->h_NeighborCapacity[i]) * 2;
                    cudaMalloc(&ds->h_NeighborsArrays[i], (ds->h_NeighborCapacity[i] * sizeof(Node)));
                    copyFullOrDelta[index] = true;
                    // gpuErrchk(cudaMemcpy(ds->h_NeighborsArrays[i], &((ds->out_neighbors[i])[0]), (ds->out_neighbors[i]).size() * sizeof(Node), cudaMemcpyHostToDevice));
                }
                index++;
                // else
                // {
                //     gpuErrchk(cudaMemcpy(ds->h_NeighborsArrays[i] + ds->h_NeighborSizes[i], &((ds->out_neighbors[i])[0]) + ds->h_NeighborSizes[i], ((ds->out_neighbors[i]).size() - ds->h_NeighborSizes[i]) * sizeof(Node), cudaMemcpyHostToDevice));
                // }
                ds->h_NeighborSizes[i] = (ds->out_neighbors[i]).size();
                ds->affectedNodes.push_back(i);
            // }
            // ds->numberOfNeighborsOnCuda += (ds->out_neighbors[i]).size();
        }
        cudaMemcpy(ds->d_NeighborsArrays, ds->h_NeighborsArrays, NEIGHBORS_POINTERS_SIZE, cudaMemcpyHostToDevice);
        coalesceEdgesAndCopyToCuda(ds, copyFullOrDelta);
        cudaMemcpy(ds->d_NeighborSizes, ds->h_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyHostToDevice);

        ds->numberOfNodesOnCuda = ds->num_nodes;
        ds->numberOfNeighborsOnCuda = ds->num_edges;
    }
}

template <typename T>
void updateNeighbors(T* ds)
{
    if(ds->numberOfNeighborsOnCuda < ds->num_edges)
    {
        int NEIGHBORS_POINTERS_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(Node*);
        int NEIGHBORS_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(int);
        // Node** h_array = (Node**)malloc(NEIGHBORS_POINTERS_SIZE);
        // int* h_NeighborSizes = (int*)malloc(NEIGHBORS_SIZE);

        // cudaMemcpy(h_array, ds->d_NeighborsArrays, NEIGHBORS_POINTERS_SIZE, cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_NeighborSizes, ds->d_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyDeviceToHost);
        // ds->numberOfNeighborsOnCuda = 0;
        bool flag = false;
        bool copyFullOrDelta[ds->affectedNodesSet.size()] {false};

        int index = 0;
        // #pragma omp for schedule(dynamic, 16)
        for(NodeID i : ds->affectedNodesSet) {
        // for(size_t i = 0 ; i < ds->num_nodes ; i++){
            // ds->numberOfNeighborsOnCuda += (ds->out_neighbors[i]).size();
            // if(ds->affected[i])
            // {
                if(ds->h_NeighborCapacity[i] < (ds->out_neighbors[i]).size())
                {
                    // Node* d_tempNeighbors;
                    // int temp_Capacity = ((ds->h_NeighborCapacity[i] * 2 < (ds->out_neighbors[i]).size()) ? (ds->out_neighbors[i]).size() : ds->h_NeighborCapacity[i]) * 2;
                    // cudaMalloc(&d_tempNeighbors, (temp_Capacity * sizeof(Node)));
                    // gpuErrchk(cudaMemcpyAsync(d_tempNeighbors, ds->h_NeighborsArrays[i], (ds->h_NeighborCapacity[i] * sizeof(Node)), cudaMemcpyDeviceToDevice));
                    // cudaFree(ds->h_NeighborsArrays[i]);
                    // ds->h_NeighborsArrays[i] = d_tempNeighbors;
                    // ds->h_NeighborCapacity[i] = temp_Capacity;
                    ds->h_NeighborCapacity[i] = ((ds->h_NeighborCapacity[i] * 2 < (ds->out_neighbors[i]).size()) ? (ds->out_neighbors[i]).size() : ds->h_NeighborCapacity[i]) * 2;
                    cudaFree(ds->h_NeighborsArrays[i]);
                    cudaMalloc(&ds->h_NeighborsArrays[i], (ds->h_NeighborCapacity[i] * sizeof(Node)));
                    flag = true;
                    copyFullOrDelta[index] = true;
                }
                index++;
                ds->affectedNodes.push_back(i);
                // cudaFree(h_array[i]);
                // cudaMalloc(&h_array[i], (ds->out_neighbors[i]).size() * sizeof(Node));
                ds->h_NeighborSizes[i] = (ds->out_neighbors[i]).size();
            // }
        }

        if(flag)
        {
            cudaMemcpy(ds->d_NeighborsArrays, ds->h_NeighborsArrays, NEIGHBORS_POINTERS_SIZE, cudaMemcpyHostToDevice);
        }
        coalesceEdgesAndCopyToCuda(ds, copyFullOrDelta);
        cudaMemcpy(ds->d_NeighborSizes, ds->h_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyHostToDevice);
        ds->numberOfNeighborsOnCuda = ds->num_edges;
    }
}


#endif  // ADLIST_CU_SUPPORT_H_
