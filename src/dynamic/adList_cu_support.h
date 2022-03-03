#ifndef ADLIST_CU_SUPPORT_H_
#define ADLIST_CU_SUPPORT_H_

#include "abstract_data_struc.h"
#include "print.h"
#include "types.h"
#include <algorithm>
#include <atomic>
#include <array>

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

__global__ void copyToCuda(NodeID* d_affectedNodes, bool* copyFullOrDelta, Node* d_coalesceNeighbors, int coalesceSize, int numAffectedNodes, Node** d_NeighborsArrays, int* d_NeighborSizes, int* d_startPosition, int* d_copySize)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if(idx < numAffectedNodes)
    {
        NodeID node = d_affectedNodes[idx];
        int copyStart = d_startPosition[idx];
        int numNodes = d_copySize[idx];
        int offset = copyFullOrDelta[idx] ? 0 : d_NeighborSizes[node];
        memcpy(d_NeighborsArrays[node] + offset, d_coalesceNeighbors + copyStart, numNodes * sizeof(Node));
    }
}

template <typename T>
void coalesceEdgesAndCopyToCuda(T* ds, bool* copyFullOrDelta, int* startPosition, int* copySize, int totalSize)
{
    Node* coalesceNeighbors;
    cudaMallocHost((void**) &coalesceNeighbors, totalSize * sizeof(Node));
    int numNodes = ds->affectedNodes.size();
    #pragma omp for schedule(dynamic, 16)
    for(int i=0; i < numNodes; i++)
    {
        NodeID node = ds->affectedNodes[i];
        int start = startPosition[i];
        if (copyFullOrDelta[i])
        {
            std::copy(ds->out_neighbors[node].begin(), ds->out_neighbors[node].end(), coalesceNeighbors + start);
        }
        else
        {
            std::copy(ds->out_neighborsDelta[node].begin(), ds->out_neighborsDelta[node].end(), coalesceNeighbors + start);
        }
    }

    int coalesceSize = totalSize;
    Node* d_coalesceNeighbors;
    int* d_copySize;
    int* d_startPosition;
    NodeID* d_affectedNodes;
    bool* d_copyFullOrDelta;

    cudaMalloc(&d_coalesceNeighbors, coalesceSize * sizeof(Node));
    cudaMemcpy(d_coalesceNeighbors, coalesceNeighbors, totalSize * sizeof(Node), cudaMemcpyHostToDevice);
    cudaMalloc(&d_affectedNodes, ds->affectedNodes.size() * sizeof(NodeID));
    cudaMemcpy(d_affectedNodes, &(ds->affectedNodes[0]), ds->affectedNodes.size() * sizeof(NodeID), cudaMemcpyHostToDevice);
    cudaMalloc(&d_copyFullOrDelta, ds->affectedNodes.size() * sizeof(bool));
    cudaMemcpy(d_copyFullOrDelta, copyFullOrDelta, ds->affectedNodes.size() * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMalloc(&d_startPosition, ds->affectedNodes.size() * sizeof(int));
    cudaMemcpy(d_startPosition, startPosition, ds->affectedNodes.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_copySize, ds->affectedNodes.size() * sizeof(int));
    cudaMemcpy(d_copySize, copySize, ds->affectedNodes.size() * sizeof(int), cudaMemcpyHostToDevice);

    const int BLK_SIZE = 512;
    dim3 blkSize(BLK_SIZE);
    dim3 gridSize((ds->affectedNodes.size() + BLK_SIZE - 1) / BLK_SIZE);
    copyToCuda<<<gridSize, blkSize>>>(d_affectedNodes, d_copyFullOrDelta, d_coalesceNeighbors, coalesceSize, ds->affectedNodes.size(), ds->d_NeighborsArrays, ds->d_NeighborSizes, d_startPosition, d_copySize);
    cudaDeviceSynchronize();
    ds->stale_neighbors.push_back(d_coalesceNeighbors);
    cudaFree(d_copySize);
    cudaFree(d_startPosition);
    cudaFree(d_affectedNodes);
    cudaFree(d_copyFullOrDelta);
    cudaFreeHost(coalesceNeighbors);
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
                        Node* tempNeighbors = h_array[i];
                        ds->stale_neighbors.push_back(tempNeighbors);
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

        int copySize[ds->affectedNodesSet.size()] = {0};
        int startPosition[ds->affectedNodesSet.size()] = {0};
        std::atomic<int> index(0);
        std::atomic<int> totalNodesToCopy(0);

        // std::copy(h_array, ds->d_NeighborsArrays, NEIGHBORS_POINTERS_SIZE, cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_NeighborSizes, ds->d_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyDeviceToHost);

        // ds->numberOfNeighborsOnCuda = 0;
        ds->affectedNodes.resize(ds->affectedNodesSet.size());
        #pragma omp for schedule(dynamic, 16)
        for(NodeID i : ds->affectedNodesSet) {
        // for(size_t i = 0 ; i < ds->num_nodes ; i++){
            // if(ds->affected[i])
            // {
                // ds->affectedNodes.push_back(i);
                int currIndex = std::atomic_fetch_add(&index, 1);
                if(ds->h_NeighborCapacity[i] < (ds->out_neighbors[i]).size())
                {
                    if(i < ds->numberOfNodesOnCuda)
                    {
                        Node* tempNeighbors = ds->h_NeighborsArrays[i];
                        ds->stale_neighbors.push_back(tempNeighbors);
                    }
                    copyFullOrDelta[currIndex] = true;
                    startPosition[currIndex] = std::atomic_fetch_add(&totalNodesToCopy, ds->out_neighbors[i].size());
                    copySize[currIndex] = ds->out_neighbors[i].size();
                    ds->h_NeighborCapacity[i] = ((ds->h_NeighborCapacity[i] * 2 < (ds->out_neighbors[i]).size()) ? (ds->out_neighbors[i]).size() : ds->h_NeighborCapacity[i]) * 2;
                    cudaMalloc(&ds->h_NeighborsArrays[i], (ds->h_NeighborCapacity[i] * sizeof(Node)));
                    // gpuErrchk(cudaMemcpy(ds->h_NeighborsArrays[i], &((ds->out_neighbors[i])[0]), (ds->out_neighbors[i]).size() * sizeof(Node), cudaMemcpyHostToDevice));
                }
                else
                {
                    startPosition[currIndex] = std::atomic_fetch_add(&totalNodesToCopy, ds->out_neighborsDelta[i].size());
                    copySize[currIndex] = ds->out_neighborsDelta[i].size();
                }
                ds->h_NeighborSizes[i] = (ds->out_neighbors[i]).size();
                ds->affectedNodes[currIndex] = i;
            // }
            // ds->numberOfNeighborsOnCuda += (ds->out_neighbors[i]).size();
        }
        cudaMemcpy(ds->d_NeighborsArrays, ds->h_NeighborsArrays, NEIGHBORS_POINTERS_SIZE, cudaMemcpyHostToDevice);
        coalesceEdgesAndCopyToCuda(ds, copyFullOrDelta, startPosition, copySize, totalNodesToCopy.load());
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
        int copySize[ds->affectedNodesSet.size()] = {0};
        int startPosition[ds->affectedNodesSet.size()] = {0};

        std::atomic<int> index(0);
        std::atomic<int> totalNodesToCopy(0);
        ds->affectedNodes.resize(ds->affectedNodesSet.size());
        #pragma omp for schedule(dynamic, 16)
        for(NodeID i : ds->affectedNodesSet) {
        // for(size_t i = 0 ; i < ds->num_nodes ; i++){
            // ds->numberOfNeighborsOnCuda += (ds->out_neighbors[i]).size();
            // if(ds->affected[i])
            // {
                int currIndex = std::atomic_fetch_add(&index, 1);
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
                    Node* tempNeighbors = ds->h_NeighborsArrays[i];
                    ds->stale_neighbors.push_back(tempNeighbors);
                    cudaMalloc(&ds->h_NeighborsArrays[i], (ds->h_NeighborCapacity[i] * sizeof(Node)));
                    flag = true;
                    copyFullOrDelta[currIndex] = true;
                    startPosition[currIndex] = std::atomic_fetch_add(&totalNodesToCopy, ds->out_neighbors[i].size());
                    copySize[currIndex] = ds->out_neighbors[i].size();
                }
                else
                {
                    startPosition[currIndex] = std::atomic_fetch_add(&totalNodesToCopy, ds->out_neighborsDelta[i].size());
                    copySize[currIndex] = ds->out_neighborsDelta[i].size();
                }
                ds->affectedNodes[currIndex] = i;
                // cudaFree(h_array[i]);
                // cudaMalloc(&h_array[i], (ds->out_neighbors[i]).size() * sizeof(Node));
                ds->h_NeighborSizes[i] = (ds->out_neighbors[i]).size();
            // }
        }

        if(flag)
        {
            cudaMemcpy(ds->d_NeighborsArrays, ds->h_NeighborsArrays, NEIGHBORS_POINTERS_SIZE, cudaMemcpyHostToDevice);
        }
        coalesceEdgesAndCopyToCuda(ds, copyFullOrDelta, startPosition, copySize, totalNodesToCopy.load());
        cudaMemcpy(ds->d_NeighborSizes, ds->h_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyHostToDevice);
        ds->numberOfNeighborsOnCuda = ds->num_edges;
    }
}


#endif  // ADLIST_CU_SUPPORT_H_
