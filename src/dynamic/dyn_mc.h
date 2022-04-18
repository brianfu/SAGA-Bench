#ifndef DYN_MC_H_
#define DYN_MC_H_

#include <algorithm>

#include "traversal.h"
#include "../common/timer.h"
#include "sliding_queue_dynamic.h"
#include "../common/pvector.h"

#include "adList_cu.h"
#include "adList_cu_support.h"

/* Algorithm: Incremental Max computation and Max Computation starting from scratch */

template<typename T>
void MCIter0(T* ds, SlidingQueue<NodeID>& queue){
    pvector<bool> visited(ds->num_nodes, false);
    
    #pragma omp parallel     
    {
        QueueBuffer<NodeID> lqueue(queue);
        #pragma omp for schedule(dynamic, 64)
        for(NodeID n=0; n < ds->num_nodes; n++){
            if(ds->affected[n]){
                float old_val = ds->property[n];
                float new_val = old_val;

                // calculate new value
                for(auto v: in_neigh(n, ds)){
                    new_val = std::max(new_val, ds->property[v]);
                }
                
                assert(new_val >= old_val);

                ds->property[n] = new_val;                                
                bool trigger = (
                    (ds->property[n] > old_val)
                    || (old_val == n)
                ); 

                if(trigger){                   
                    //put the out-neighbors into active list 
                    for(auto v: out_neigh(n, ds)){
                        bool curr_val = visited[v];
                        if(!curr_val){
                            if(compare_and_swap(visited[v], curr_val, true)) 
                                 lqueue.push_back(v);
                        }             
                    }                                                                    
                }        
            }
        }
        lqueue.flush();
    }
}

__global__ void MCIter0_cuda(NodeID* d_affectedNodes, int* d_affectedNum, int* d_frontierNum, NodeID* d_frontierNodes, NodeID* d_newFrontierNodes, bool* affected, float* property, Node** d_NeighborsArrays, int* d_NeighborSizes,
                    bool* visited, int64_t numNodes, int64_t numEdges,
                    bool* frontierArr, bool* frontierExists) {
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if (idx < *d_affectedNum)
    {
        NodeID node = d_affectedNodes[idx];
        float old_val = property[node];
        float new_val = old_val;
        // Not using in-neghibors because graph being tested is not directed
        // int iEnd = (idx + 1) < numNodes ? d_InNodes[idx+1] : numEdges;
        // for(int i = d_InNodes[idx]; i < iEnd; i++)
        int iEnd = d_NeighborSizes[node];
        for(int i = 0; i < iEnd; i++)
        {
            NodeID v = d_NeighborsArrays[node][i].node;
            float tempVal = property[v];
            new_val = tempVal > new_val ? tempVal : new_val;
        }

        assert(new_val >= old_val);

        property[node] = new_val;

        bool trigger = (new_val > old_val) || (old_val == node);

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

__global__ void dynMc_kerenel(int* d_frontierNum, NodeID* d_frontierNodes,
                    NodeID* d_newFrontierNodes, int* d_newFrontierNum,
                    float* property, Node** d_NeighborsArrays,
                    int* d_NeighborSizes, bool* visited_c, int64_t num_nodes,
                    int64_t num_edges, bool* frontierExists){
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if(idx < *d_frontierNum){
        NodeID node = d_frontierNodes[idx];
        float old_val = property[node];
        float new_val = old_val;
        int iEnd = d_NeighborSizes[node];
        for(int i = 0; i < iEnd; i++)
        {
            NodeID v = d_NeighborsArrays[node][i].node; 
            float tempVal = property[v];
            new_val = tempVal > new_val ? tempVal : new_val;
        }

        assert(new_val >= old_val);

        property[node] = new_val;

        bool trigger = (new_val > old_val) || (old_val == node);
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


__global__ void initPropertiesMcDyn(float* property, int numNodes)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numNodes; i+=stride)
    {
        if (property[i] == -1)
            property[i] = i;
    }
}

template<typename T>
void dynMCAlg(T* ds){
    //std::cout << "Number of nodes: "<< ds->num_nodes << std::endl;   
    std::cout << "Running dynamic MC" << std::endl;
    Timer t;
    t.Start();   
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

        initPropertiesMcDyn<<<gridSize, blkSize, 0, ds->adListStream>>>(ds->property_c, ds->num_nodes);

        int PROPERTY_SIZE = ds->num_nodes * sizeof(*ds->property_c);

        gridSize = ((affectedNum + BLK_SIZE - 1) / BLK_SIZE);
        MCIter0_cuda<<<gridSize, blkSize, 0, ds->adListStream>>>(d_affectedNodes, d_affectedNum, d_frontierNum, d_frontierNodes,
                    d_newFrontierNodes, ds->affected_c, ds->property_c, ds->d_NeighborsArrays, ds->d_NeighborSizes,
                    visited_c, ds->num_nodes, ds->num_edges,
                    ds->frontierArr_c, d_frontierExists);

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

        while(h_frontierExists){
            h_frontierExists = false;

            cudaMemsetAsync(visited_c, 0, FRONTIER_SIZE, ds->adListStream);
            cudaMemsetAsync(d_frontierExists, 0, sizeof(bool), ds->adListStream);

            cudaStreamSynchronize(ds->adListStream);
            gridSize = (frontierNum + BLK_SIZE - 1) / BLK_SIZE;
            dynMc_kerenel<<<gridSize, blkSize, 0, ds->adListStream>>>(d_frontierNum, d_frontierNodes, d_newFrontierNodes, d_newFrontierNum, ds->property_c,  ds->d_NeighborsArrays, ds->d_NeighborSizes,
                        visited_c, ds->num_nodes, ds->num_edges, d_frontierExists);

            gpuErrchk(cudaMemcpyAsync(h_visited, visited_c, FRONTIER_SIZE, cudaMemcpyDeviceToHost, ds->adListStream));
            cudaDeviceSynchronize();
            getFrontier(h_visited, ds->num_nodes, &frontierNum, d_frontierNodes, ds->adListStream);
            gpuErrchk(cudaMemcpyAsync(d_frontierNum, &frontierNum, sizeof(int), cudaMemcpyHostToDevice, ds->adListStream));
            gpuErrchk(cudaMemcpy(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost));
        }

        cudaStreamSynchronize(ds->adListStream);
        gpuErrchk(cudaMemcpyAsync(&(ds->property[0]), ds->property_c, PROPERTY_SIZE, cudaMemcpyDeviceToHost, ds->adListStream));

        cudaFreeAsync(visited_c, ds->adListStream);
        cudaFreeAsync(d_frontierExists, ds->adListStream);

        cudaFreeAsync(d_frontierNum, ds->adListStream);
        cudaFreeAsync(d_frontierNodes, ds->adListStream);
        cudaFreeAsync(d_newFrontierNodes, ds->adListStream);
        cudaFreeAsync(d_newFrontierNum, ds->adListStream);
        cudaFreeHost(h_visited);

        #pragma omp for schedule(dynamic, 16)
        for(NodeID i : ds->affectedNodesSet){
            ds->out_neighborsDelta[i].clear();  
        }
        (ds->affectedNodes).clear();
        ds->affectedNodesSet.clear();

        gpuErrchk(cudaDeviceSynchronize());
    }
    ds->cudaNeighborsConditional.notify_all();

    t.Stop();    
    ofstream out("Alg.csv", std::ios_base::app);   
    out << t.Seconds() << std::endl;    
    out.close();
}

__global__ void initMcProperties(float* property, int numNodes)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < numNodes; i+=stride)
    {
        property[i] = i;
    }
}

__global__ void maxComputation(NodeID *nodes, NodeID *d_out_neighbors, bool *frontierArr, bool *newFrontierArr, bool isInitial, float *property, bool* frontierExists, int64_t numNodes, int64_t numEdges)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if (idx < numNodes)
    {
        if(frontierArr[idx] || isInitial)
        {
            float old_val = property[idx];
            float new_val = old_val;

            int iEnd = (idx + 1) < numNodes ? nodes[idx+1] : numEdges;
            int iStart = nodes[idx];
            for(int i = iStart; i < iEnd; i++)
            {
                float neighborVal = property[d_out_neighbors[i]];
                new_val = new_val > neighborVal ? new_val : neighborVal;
            }

            assert(new_val >= old_val);

            property[idx] = new_val;

            if(new_val != old_val)
            {
                for(int i = iStart; i < iEnd; i++)
                {
                    bool curr_val = newFrontierArr[d_out_neighbors[i]];

                    if(!curr_val)
                    {
                        atomic_CAS(&newFrontierArr[d_out_neighbors[i]], false, true);
                    }

                    if(!(*frontierExists))
                    {
                        atomic_CAS(frontierExists, false, true);
                    }
                }
            }
            
            frontierArr[idx] = false;
        }
    }
}

template<typename T>
void MCStartFromScratch(T* ds){ 
    std::cout << "Running MC from scratch" << std::endl;

    Timer t;
    t.Start();

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
    initMcProperties<<<gridSize, blkSize, 0, ds->adListStream>>>(ds->propertyF_c, ds->num_nodes);

    bool *d_frontierExists;
    gpuErrchk(cudaMalloc(&d_frontierExists, sizeof(bool)));
    bool h_frontierExists = false;
    gpuErrchk(cudaMemcpy(d_frontierExists, &h_frontierExists, sizeof(bool), cudaMemcpyHostToDevice));

    int FRONTIER_SIZE = ds->num_nodes * sizeof(*ds->frontierArr_c);
    gpuErrchk(cudaMallocAsync((void**)&ds->frontierArr_c, FRONTIER_SIZE, ds->adListStream));
    cudaMemsetAsync(ds->frontierArr_c, 0, FRONTIER_SIZE, ds->adListStream);
    bool* newFrontierArr_c;
    gpuErrchk(cudaMallocAsync((void**)&newFrontierArr_c, FRONTIER_SIZE, ds->adListStream));
    cudaMemsetAsync(newFrontierArr_c, 0, FRONTIER_SIZE, ds->adListStream);

    maxComputation<<<gridSize, blkSize, 0, ds->adListStream>>>(ds->d_nodes, ds->d_out_neighbors, ds->frontierArr_c, newFrontierArr_c, true, ds->propertyF_c, d_frontierExists, ds->num_nodes, ds->num_edges);
    cudaDeviceSynchronize();
    swap(ds->frontierArr_c, newFrontierArr_c);
    cudaMemsetAsync(newFrontierArr_c, 0, FRONTIER_SIZE, ds->adListStream);
    gpuErrchk(cudaMemcpy(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost));

    while(h_frontierExists){
        h_frontierExists = false;
        gpuErrchk(cudaMemcpy(d_frontierExists, &h_frontierExists, sizeof(bool), cudaMemcpyHostToDevice));
        maxComputation<<<gridSize, blkSize, 0, ds->adListStream>>>(ds->d_nodes, ds->d_out_neighbors, ds->frontierArr_c, newFrontierArr_c, false, ds->propertyF_c, d_frontierExists, ds->num_nodes, ds->num_edges);
        cudaDeviceSynchronize();
        swap(ds->frontierArr_c, newFrontierArr_c);
        cudaMemsetAsync(newFrontierArr_c, 0, FRONTIER_SIZE, ds->adListStream);
        gpuErrchk(cudaMemcpy(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost));
    }

    gpuErrchk(cudaMemcpyAsync(&(ds->property[0]), ds->propertyF_c, PROPERTY_SIZE, cudaMemcpyDeviceToHost, ds->adListStream));

    cudaFreeAsync(ds->propertyF_c, ds->adListStream);
    cudaFreeAsync(ds->d_nodes, ds->adListStream);
    cudaFreeAsync(ds->d_out_neighbors, ds->adListStream);
    cudaFreeAsync(ds->frontierArr_c, ds->adListStream);
    cudaFreeAsync(newFrontierArr_c, ds->adListStream);

    cudaDeviceSynchronize();
    cudaFreeHost(ds->h_out_neighbors);
    cudaFreeHost(ds->h_nodes);

    t.Stop();    
    ofstream out("Alg.csv", std::ios_base::app);   
    out << t.Seconds() << std::endl;    
    out.close();
}

#endif  // DYN_MC_H_    
