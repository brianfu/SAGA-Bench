#ifndef DYN_PR_H_
#define DYN_PR_H_

#include "traversal.h"
#include "../common/timer.h"
#include "sliding_queue_dynamic.h"
#include "../common/pvector.h"
#include <cmath>
#include <iostream>
#include <numeric>

#include "adList_cu.h"
#include "adList_cu_support.h"

/* Algorithm: Incremental PageRank and PageRank starting from scratch */

__device__ __host__ const float kDamp = 0.85;
__device__ __host__ const float PRThreshold = 0.0000001;  

__global__ void PRIter0_cuda(NodeID* d_affectedNodes, int* d_affectedNum, int* d_frontierNum, NodeID* d_frontierNodes, NodeID* d_newFrontierNodes, bool* affected, Rank* property, Node** d_NeighborsArrays, int* d_NeighborSizes,
                    bool* visited, int64_t numNodes, int64_t numEdges,
                    bool* frontierArr, bool* frontierExists, Rank* outgoing_contrib, Rank base_score) {
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if (idx < *d_affectedNum)
    {
        NodeID node = d_affectedNodes[idx];
        Rank old_rank = property[node];
        Rank incoming_total = 0;
        // Not using in-neghibors because graph being tested is not directed
        // int iEnd = (idx + 1) < numNodes ? d_InNodes[idx+1] : numEdges;
        // for(int i = d_InNodes[idx]; i < iEnd; i++)
        int iEnd = d_NeighborSizes[node];
        for(int i = 0; i < iEnd; i++)
        {
            NodeID v = d_NeighborsArrays[node][i].node;
            incoming_total += outgoing_contrib[v];
        }
        Rank new_rank = base_score + kDamp * incoming_total;
        property[node] = new_rank;

        bool trigger = fabsf(new_rank - old_rank) > PRThreshold;

        if(trigger){
            if(!(*frontierExists))
            {
                atomic_CAS(frontierExists, false, true);
            }
            for(int j = 0; j < iEnd; j++)
            {
                NodeID v = d_NeighborsArrays[node][j].node; 
                bool curr_val = visited[v];
                if(!curr_val){
                    atomic_CAS(&visited[v], curr_val, true);
                }
            }
        }
    }
}

__global__ void dynPr_kerenel(int* d_frontierNum, NodeID* d_frontierNodes, NodeID* d_newFrontierNodes, int* d_newFrontierNum, Rank* property, Rank* outgoing_contrib, Node** d_NeighborsArrays, int* d_NeighborSizes,
                    bool* visited_c, int64_t num_nodes, int64_t num_edges,
                    bool* frontierExists, Rank base_score){
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if(idx < *d_frontierNum){
        NodeID node = d_frontierNodes[idx];
        Rank old_rank = property[node];
        Rank incoming_total = 0;
        int iEnd = d_NeighborSizes[node];
        for(int i = 0; i < iEnd; i++)
        {
            NodeID v = d_NeighborsArrays[node][i].node; 
            incoming_total += outgoing_contrib[v];
        }
        Rank new_rank = base_score + kDamp * incoming_total;
        property[node] = new_rank;
        bool trigger = fabsf(new_rank - old_rank) > PRThreshold;
        if (trigger)
        {
            if(!(*frontierExists))
            {
                atomic_CAS(frontierExists, false, true);
            }
            for(int i = 0; i < iEnd; i++)
            {
                NodeID v = d_NeighborsArrays[node][i].node;
                bool curr_val = visited_c[v];
                if(!curr_val){
                    atomic_CAS(&visited_c[v], curr_val, true);
                }
            }
        }
    }
}

__global__ void initPropertiesDyn(Rank* property, int numNodes, Rank value)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numNodes; i+=stride)
    {
        if (property[i] == -1)
            property[i] = value;
    }
}

__global__ void updateOutgoingContribDyn(Rank* outgoing_contrib, Rank* property, int64_t numNodes, int* d_NeighborSizes)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numNodes; i+=stride)
    {
        outgoing_contrib[i] = property[i] / d_NeighborSizes[i];
    }
}

template<typename T>
void dynPRAlg(T* ds)
{     
    std::cout << "Running dynamic PR" << std::endl;  

    Timer t_cuda;
    t_cuda.Start();
    {
        std::scoped_lock lock(ds->cudaNeighborsMutex);

        if(ds->sizeOfNodesArrayOnCuda < ds->num_nodes)
        {
            resizeAndCopyToCudaMemory(ds);
        }
        else if(ds->numberOfNodesOnCuda < ds->num_nodes)
        {
            copyToCudaMemory(ds);
        }
        else
        {
            updateNeighbors(ds);
        }
        cudaDeviceSynchronize();

        bool *d_frontierExists = nullptr;
        gpuErrchk(cudaMallocAsync((void**)&d_frontierExists, sizeof(bool), ds->adListStream));
        bool h_frontierExists = false;
        cudaMemsetAsync(d_frontierExists, 0, sizeof(bool), ds->adListStream);

        const Rank base_score = (1.0f - kDamp)/(ds->num_nodes);
        int max_iters = 10;

        ds->affectedNodes.assign(ds->affectedNodesSet.begin(), ds->affectedNodesSet.end());

        int FRONTIER_SIZE = ds->num_nodes * sizeof(*ds->frontierArr_c);
        
        bool* visited_c;
        gpuErrchk(cudaMallocAsync((void**)&visited_c, FRONTIER_SIZE, ds->adListStream));
        cudaMemsetAsync(visited_c, 0, FRONTIER_SIZE, ds->adListStream);

        bool* h_visited;
        gpuErrchk(cudaMallocHost((void**)&h_visited, FRONTIER_SIZE));
        memset(h_visited, 0, FRONTIER_SIZE);

        int NODES_SIZE = ds->num_nodes * sizeof(NodeID);
        int affectedNum = ds->affectedNodes.size();
        int* d_affectedNum;
        gpuErrchk(cudaMallocAsync(&(d_affectedNum), sizeof(int), ds->adListStream));
        gpuErrchk(cudaMemcpyAsync(d_affectedNum, &(affectedNum), sizeof(int), cudaMemcpyHostToDevice, ds->adListStream));

        int AFFECTED_SIZE = affectedNum * sizeof(NodeID);
        NodeID* d_affectedNodes;
        gpuErrchk(cudaMallocAsync(&(d_affectedNodes), AFFECTED_SIZE, ds->adListStream));
        gpuErrchk(cudaMemcpyAsync(d_affectedNodes, &(ds->affectedNodes[0]), AFFECTED_SIZE, cudaMemcpyHostToDevice, ds->adListStream));

        int* d_frontierNum;
        gpuErrchk(cudaMallocAsync(&(d_frontierNum), sizeof(int), ds->adListStream));
        cudaMemsetAsync(d_frontierNum, 0, sizeof(int), ds->adListStream);

        NodeID* d_frontierNodes;
        gpuErrchk(cudaMallocAsync(&d_frontierNodes, NODES_SIZE, ds->adListStream));
        cudaMemsetAsync(d_frontierNodes, 0, NODES_SIZE, ds->adListStream);

        NodeID* d_newFrontierNodes;
        gpuErrchk(cudaMallocAsync(&d_newFrontierNodes, NODES_SIZE, ds->adListStream));
        cudaMemsetAsync(d_newFrontierNodes, 0, NODES_SIZE, ds->adListStream);

        const int BLK_SIZE = 512;
        dim3 blkSize(BLK_SIZE);
        dim3 gridSize((ds->num_nodes + BLK_SIZE - 1) / BLK_SIZE);

        initPropertiesDyn<<<gridSize, blkSize, 0, ds->adListStream>>>(ds->property_c, ds->num_nodes, 1.0f / (ds->num_nodes));

        Rank* outgoing_contrib;
        int PROPERTY_SIZE = ds->num_nodes * sizeof(Rank);
        gpuErrchk(cudaMallocAsync(&outgoing_contrib, PROPERTY_SIZE, ds->adListStream));
        updateOutgoingContribDyn<<<gridSize, blkSize, 0, ds->adListStream>>>(outgoing_contrib, ds->property_c, ds->num_nodes, ds->d_NeighborSizes);

        gridSize = ((affectedNum + BLK_SIZE - 1) / BLK_SIZE);
        PRIter0_cuda<<<gridSize, blkSize, 0, ds->adListStream>>>(d_affectedNodes, d_affectedNum, d_frontierNum, d_frontierNodes,
                    d_newFrontierNodes, ds->affected_c, ds->property_c, ds->d_NeighborsArrays, ds->d_NeighborSizes,
                    visited_c, ds->num_nodes, ds->num_edges,
                    ds->frontierArr_c, d_frontierExists, outgoing_contrib, base_score);

        gpuErrchk(cudaMemcpyAsync(h_visited, visited_c, FRONTIER_SIZE, cudaMemcpyDeviceToHost, ds->adListStream));

        int frontierNum = 0;
        gpuErrchk(cudaMemcpyAsync(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost, ds->adListStream));

        cudaDeviceSynchronize();

        getFrontier(h_visited, ds->num_nodes, &frontierNum, d_frontierNodes, ds->adListStream);
        gpuErrchk(cudaMemcpyAsync(d_frontierNum, &frontierNum, sizeof(int), cudaMemcpyHostToDevice, ds->adListStream));

        cudaFreeAsync(d_affectedNodes, ds->adListStream);
        cudaFreeAsync(d_affectedNum, ds->adListStream);

        int* d_newFrontierNum;
        gpuErrchk(cudaMallocAsync(&(d_newFrontierNum), sizeof(int), ds->adListStream));
        cudaMemsetAsync(d_newFrontierNum, 0, sizeof(int), ds->adListStream);
        
        int iter = 0;
        while(h_frontierExists && iter < max_iters){        
            h_frontierExists = false;     
            
            cudaMemsetAsync(visited_c, 0, FRONTIER_SIZE, ds->adListStream);
            cudaMemsetAsync(d_frontierExists, 0, sizeof(bool), ds->adListStream);

            cudaStreamSynchronize(ds->adListStream);
            gridSize = ((ds->num_nodes + BLK_SIZE - 1) / BLK_SIZE);
            updateOutgoingContribDyn<<<gridSize, blkSize, 0, ds->adListStream>>>(outgoing_contrib, ds->property_c, ds->num_nodes, ds->d_NeighborSizes);
            
            cudaStreamSynchronize(ds->adListStream);
            gridSize = (frontierNum + BLK_SIZE - 1) / BLK_SIZE;
            dynPr_kerenel<<<gridSize, blkSize, 0, ds->adListStream>>>(d_frontierNum, d_frontierNodes, d_newFrontierNodes, d_newFrontierNum, ds->property_c, outgoing_contrib,  ds->d_NeighborsArrays, ds->d_NeighborSizes,
                        visited_c, ds->num_nodes, ds->num_edges, d_frontierExists, base_score);

            gpuErrchk(cudaMemcpyAsync(h_visited, visited_c, FRONTIER_SIZE, cudaMemcpyDeviceToHost, ds->adListStream));
            cudaDeviceSynchronize();
            getFrontier(h_visited, ds->num_nodes, &frontierNum, d_frontierNodes, ds->adListStream);
            gpuErrchk(cudaMemcpyAsync(d_frontierNum, &frontierNum, sizeof(int), cudaMemcpyHostToDevice, ds->adListStream));
            gpuErrchk(cudaMemcpy(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost));

            iter++;
        }

        cudaStreamSynchronize(ds->adListStream);
        gpuErrchk(cudaMemcpyAsync(&(ds->property[0]), ds->property_c, PROPERTY_SIZE, cudaMemcpyDeviceToHost, ds->adListStream));

        cudaFreeAsync(visited_c, ds->adListStream);
        cudaFreeAsync(d_frontierExists, ds->adListStream);
        cudaFreeAsync(outgoing_contrib, ds->adListStream);

        cudaFreeAsync(d_frontierNum, ds->adListStream);
        cudaFreeAsync(d_frontierNodes, ds->adListStream);
        cudaFreeAsync(d_newFrontierNodes, ds->adListStream);
        cudaFreeAsync(d_newFrontierNum, ds->adListStream);
        cudaFreeHost(h_visited);

        #pragma omp for schedule(dynamic, 16)
        for(NodeID i = 0; i < ds->num_nodes; i++){
            ds->affected[i] = false;  
        }

        #pragma omp for schedule(dynamic, 16)
        for(NodeID i : ds->affectedNodesSet){
            ds->out_neighborsDelta[i].clear();  
        }
        (ds->affectedNodes).clear();
        ds->affectedNodesSet.clear();

        gpuErrchk(cudaDeviceSynchronize());
    }
    ds->cudaNeighborsConditional.notify_all();

    t_cuda.Stop();
    ofstream out("Alg.csv", std::ios_base::app);   
    out << t_cuda.Seconds() << std::endl;    
    out.close();
}

__global__ void initProperties(Rank* property, int numNodes, Rank value)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numNodes; i+=stride)
    {
        property[i] = value;
    }
}

__global__ void updateOutgoingContrib(Rank* outgoing_contrib, NodeID *nodes, Rank* property, int64_t numNodes, int64_t numEdges)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if (idx < numNodes)
    {
        int numNeighbors = (idx + 1) < numNodes ? nodes[idx+1] - nodes[idx]: numEdges - nodes[idx];
        outgoing_contrib[idx] = property[idx] / numNeighbors;
    }
}

__global__ void updatePageRank(Rank* outgoing_contrib, NodeID *nodes, NodeID* neighbors, Rank* property, double* error, Rank baseScore, int64_t numNodes, int64_t numEdges)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if (idx < numNodes)
    {
        int iStart = nodes[idx];
        int iEnd = (idx + 1) < numNodes ? nodes[idx+1] : numEdges;
        Rank incoming_total = 0;
        for(int i = iStart; i < iEnd; i++)
        {
            incoming_total += outgoing_contrib[neighbors[i]];
        }
        Rank old_rank = property[idx];
        property[idx] = baseScore + kDamp * incoming_total;
        Rank newRank = property[idx];
        
        error[idx] = fabsf(newRank - old_rank);
    }
}

template<typename T>
void PRStartFromScratch(T* ds)
{ 
    std::cout << "Running PR from scratch" << std::endl;

    Timer t;
    t.Start();

    const Rank base_score = (1.0f - kDamp)/(ds->num_nodes);
    int max_iters = 10;
    double epsilon = 0.0001;
    // Reset ALL property values 
    int PROPERTY_SIZE = ds->num_nodes * sizeof(*ds->propertyF_c);
    gpuErrchk(cudaMallocAsync(&ds->propertyF_c, PROPERTY_SIZE, ds->adListStream));
    
    int NODES_SIZE = ds->num_nodes * sizeof(NodeID);
    int NEIGHBOURS_SIZE = ds->num_edges * sizeof(NodeID);

    cudaMallocHost((void**)&ds->h_nodes, NODES_SIZE);
    cudaMallocHost((void**)&ds->h_out_neighbors, NEIGHBOURS_SIZE);
    
    int outNeighborPosition = 0;
    int currentNode = 0;
    for(auto outNeighbor = ds->out_neighbors.begin(); outNeighbor != ds->out_neighbors.end(); outNeighbor++)
    {
        ds->h_nodes[currentNode] = outNeighborPosition;
        currentNode++;
        
        for(auto node = (*outNeighbor).begin(); node != (*outNeighbor).end(); node++)
        {
            ds->h_out_neighbors[outNeighborPosition] = (*node).getNodeID();
            outNeighborPosition++;
        }
    }

    gpuErrchk(cudaMallocAsync(&(ds->d_nodes), NODES_SIZE, ds->adListStream));
    gpuErrchk(cudaMemcpyAsync(ds->d_nodes, ds->h_nodes, NODES_SIZE, cudaMemcpyHostToDevice, ds->adListStream));

    gpuErrchk(cudaMallocAsync(&(ds->d_out_neighbors), NEIGHBOURS_SIZE, ds->adListStream));
    gpuErrchk(cudaMemcpyAsync(ds->d_out_neighbors, ds->h_out_neighbors, NEIGHBOURS_SIZE, cudaMemcpyHostToDevice, ds->adListStream));

    const int BLK_SIZE = 512;
    dim3 blkSize(BLK_SIZE);
    dim3 gridSize((ds->num_nodes + BLK_SIZE - 1) / BLK_SIZE);
    initProperties<<<gridSize, blkSize, 0, ds->adListStream>>>(ds->propertyF_c, ds->num_nodes, 1.0f / (ds->num_nodes));

    Rank* outgoing_contrib;
    gpuErrchk(cudaMallocAsync(&outgoing_contrib, PROPERTY_SIZE, ds->adListStream));

    double* error;
    double* h_error;
    gpuErrchk(cudaMallocAsync(&error, ds->num_nodes * sizeof(double), ds->adListStream));
    cudaMallocHost((void**)&h_error, ds->num_nodes * sizeof(double));
    cudaDeviceSynchronize();

    double errorSum = 0;
    int iter;
    for (iter = 0; iter < max_iters; iter++) {
        errorSum = 0.0;
        cudaMemsetAsync(error, 0, ds->num_nodes * sizeof(double), ds->adListStream);
        updateOutgoingContrib<<<gridSize, blkSize, 0, ds->adListStream>>>(outgoing_contrib, ds->d_nodes, ds->propertyF_c, ds->num_nodes, ds->num_edges);
        updatePageRank<<<gridSize, blkSize, 0, ds->adListStream>>>(outgoing_contrib, ds->d_nodes, ds->d_out_neighbors, ds->propertyF_c, error, base_score, ds->num_nodes, ds->num_edges);
        gpuErrchk(cudaMemcpyAsync(h_error, error, ds->num_nodes * sizeof(double), cudaMemcpyDeviceToHost, ds->adListStream));
        cudaDeviceSynchronize();
        errorSum = std::accumulate(h_error, h_error + ds->num_nodes, 0.0);
        if(errorSum < epsilon)
            break;
    }

    gpuErrchk(cudaMemcpyAsync(&(ds->property[0]), ds->propertyF_c, PROPERTY_SIZE, cudaMemcpyDeviceToHost, ds->adListStream));

    cudaFreeAsync(ds->propertyF_c, ds->adListStream);
    cudaFreeAsync(error, ds->adListStream);
    cudaFreeAsync(outgoing_contrib, ds->adListStream);
    cudaFreeAsync(ds->d_nodes, ds->adListStream);
    cudaFreeAsync(ds->d_out_neighbors, ds->adListStream);
    cudaFreeAsync(error, ds->adListStream);

    cudaDeviceSynchronize();
    cudaFreeHost(ds->h_out_neighbors);
    cudaFreeHost(ds->h_nodes);
    cudaFreeHost(h_error);

    t.Stop();    
    ofstream out("Alg.csv", std::ios_base::app);   
    out << t.Seconds() << std::endl;    
    out.close();
}

#endif // DYN_PR_H