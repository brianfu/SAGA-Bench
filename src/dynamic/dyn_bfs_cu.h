#ifndef DYN_BFS_H_
#define DYN_BFS_H_

#include <algorithm>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#if 0
#include "traversal.h"
#endif

#include "../common/timer.h"
#include "sliding_queue_dynamic.h"
#include "../common/pvector.h"

#include "adList_cu.h"
#include "adList_cu_support.h"

#include <stdio.h>

/* Algorithm: Incremental BFS and BFS starting from scratch */
__global__ void BFSIter0_cuda(NodeID* d_affectedNodes, int* d_affectedNum, int* d_frontierNum, NodeID* d_frontierNodes, NodeID* d_newFrontierNodes, bool* affected, float* property, Node** d_NeighborsArrays, int* d_NeighborSizes,
                    bool* visited, int64_t numNodes, int64_t numEdges,
                    bool* frontierArr, bool* frontierExists) {
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    const float MAX = 214748;
    if (idx < *d_affectedNum)
    {
        NodeID node = d_affectedNodes[idx];
        // if(affected[idx]){
            float old_depth = property[node];
            int newDepth = MAX;
            // Not using in-neghibors because graph being tested is not directed
            // int iEnd = (idx + 1) < numNodes ? d_InNodes[idx+1] : numEdges;
            // for(int i = d_InNodes[idx]; i < iEnd; i++)
            int iEnd = d_NeighborSizes[node];
            for(int i = 0; i < iEnd; i++)
            {
                NodeID v = d_NeighborsArrays[node][i].node;
                float neighborDepth = property[v];
                if (neighborDepth != -1)
                {
                    newDepth = newDepth <  (neighborDepth + 1) ? newDepth : (neighborDepth + 1);
                }
            }

            bool trigger = (
                ((newDepth < old_depth) || (old_depth == -1)) 
                && (newDepth != MAX));

            if(trigger){
                if(!(*frontierExists))
                {
                    atomic_CAS(frontierExists, false, true);
                }
                property[node] = newDepth;
                int iOutEnd = d_NeighborSizes[node];
                for(int j = 0; j < iOutEnd; j++)
                {
                    NodeID v = d_NeighborsArrays[node][j].node; 
                    float curr_depth = property[v];
                    float updated_depth = newDepth + 1;
                    if((updated_depth < curr_depth) || (curr_depth == -1)){
                        bool curr_val = visited[v];
                        int currPos = *d_frontierNum;
                        if(!curr_val){
                            if(curr_val == atomic_CAS(&visited[v], curr_val, true))
                            {
                                while(!(currPos == atomicCAS(d_frontierNum, currPos, currPos+1))){
                                currPos = *d_frontierNum;
                                }
                                d_frontierNodes[currPos] = v;
                            }
                        }
                        // float diff = updated_depth - curr_depth;
                        while(!(curr_depth == atomicCAS_f32(&property[v], curr_depth, updated_depth))){
                            curr_depth = property[v];
                            // diff = updated_depth - curr_depth;
                            if(curr_depth <= updated_depth){
                                break;
                            }
                        }
                    }
                }
            }
        // }
    }
}

// template<typename T> 
// void BFSIter0(T* ds, SlidingQueue<NodeID>& queue){  
//     pvector<bool> visited(ds->num_nodes, false);     
  
//     #pragma omp parallel     
//     {
//         QueueBuffer<NodeID> lqueue(queue);
//         #pragma omp for schedule(dynamic, 64)
//         for(NodeID n=0; n < ds->num_nodes; n++){
//             if(ds->affected[n]){
//                 float old_depth = ds->property[n];
//                 float new_depth = std::numeric_limits<float>::max();

//                 // pull new depth from incoming neighbors
//                 for(auto v: in_neigh(n, ds)){
//                     if (ds->property[v] != -1) {
//                         new_depth = std::min(new_depth, ds->property[v] + 1);
//                     }
//                 }                                         
                
//                 // trigger happens if it is:
//                 // 1) brand new vertex with old_prop = -1 and we found a new valid min depth 
//                 // 2) already existing vertex and we found a new depth smaller than old depth 
//                 bool trigger = (
//                 ((new_depth < old_depth) || (old_depth == -1)) 
//                 && (new_depth != std::numeric_limits<float>::max())                 
//                 );               

//                 /*if(trigger){                                                 
//                     ds->property[n] = new_depth; 
//                     for(auto v: out_neigh(n, dataStruc, ds, directed)){
//                         float curr_depth = ds->property[v];
//                         float updated_depth = ds->property[n] + 1;                        
//                         if((updated_depth < curr_depth) || (curr_depth == -1)){   
//                             if(compare_and_swap(ds->property[v], curr_depth, updated_depth)){                                                              
//                                 lqueue.push_back(v); 
//                             }
//                         }
//                     }
//                 }*/

//                 // Note: above is commented and included this new thing. 
//                 // Above was leading to vertices being queued redundantly
//                 // Above assumes updated_depth < curr_depth only once. 
//                 // This is not true in dynamic case because we start from affected vertices
//                 // whose depths are not all necessary the same.
//                 // In static version, the above works because static version starts from the source 
//                 // and we know that updated_depth < curr_depth only once. 

//                 if(trigger){
//                     ds->property[n] = new_depth; 
//                     for(auto v: out_neigh(n, ds)){
//                         float curr_depth = ds->property[v];
//                         float updated_depth = ds->property[n] + 1;
//                         if((updated_depth < curr_depth) || (curr_depth == -1)){
//                             bool curr_val = visited[v];
//                             if(!curr_val){
//                                 if(compare_and_swap(visited[v], curr_val, true))
//                                     lqueue.push_back(v);
//                             }
//                             while(!compare_and_swap(ds->property[v], curr_depth, updated_depth)){
//                                 curr_depth = ds->property[v];
//                                 if(curr_depth <= updated_depth){
//                                     break;
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//         lqueue.flush();
//     }   
// }

__global__ void dynBfs_kerenel(int* d_frontierNum, NodeID* d_frontierNodes, NodeID* d_newFrontierNodes, int* d_newFrontierNum, float* property, Node** d_NeighborsArrays, int* d_NeighborSizes,
                    bool* visited_c, int64_t num_nodes, int64_t num_edges,
                    bool* frontierExists){
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if(idx < *d_frontierNum){
        NodeID node = d_frontierNodes[idx];
        int iEnd = d_NeighborSizes[node];
        for(int i = 0; i < iEnd; i++)
        {
            NodeID v = d_NeighborsArrays[node][i].node; 
            float curr_depth = property[v];
            float updated_depth = property[node] + 1;
            if((updated_depth < curr_depth) || (curr_depth == -1)){
                bool curr_val = visited_c[v];
                if(!(*frontierExists))
                {
                    atomic_CAS(frontierExists, false, true);
                }
                int currPos = *d_newFrontierNum;
                if(!curr_val){
                    if(curr_val == atomic_CAS(&visited_c[v], curr_val, true))
                    {
                        while(!(currPos == atomicCAS(d_newFrontierNum, currPos, currPos+1))){
                            currPos = *d_newFrontierNum;
                        }
                        d_newFrontierNodes[currPos] = v;
                    }
                }

                // float diff = updated_depth - curr_depth; 
                while(!(curr_depth == atomicCAS_f32(&property[v], curr_depth, updated_depth))){
                    curr_depth = property[v];
                    // diff = updated_depth - curr_depth;
                    if(curr_depth <= updated_depth){
                        break;
                    }
                }
            }
        }
    }
}

template<typename T>
void dynBFSAlg(T* ds, NodeID source){
        std::cout <<"Running dynamic BFS " << std::endl;
        Timer t;
        Timer t_cuda;
        t_cuda.Start();
    {
        std::scoped_lock lock(ds->cudaNeighborsMutex);

        // Timer t1_cuda;
        // Timer t2_cuda;
        // Timer t3_cuda;

        // double t1Time = 0;
        // double t2Time = 0;
        // double t3Time = 0;

        if(ds->sizeOfNodesArrayOnCuda < ds->num_nodes)
        {
            // t1_cuda.Start();
            resizeAndCopyToCudaMemory(ds);
            // t1_cuda.Stop();
            // t1Time = t1_cuda.Seconds();
        }
        else if(ds->numberOfNodesOnCuda < ds->num_nodes)
        {
            // t2_cuda.Start();
            copyToCudaMemory(ds);
            // t2_cuda.Stop();
            // t2Time = t2_cuda.Seconds();
        }
        else
        {
            // t3_cuda.Start();
            updateNeighbors(ds);
            // t3_cuda.Stop();
            // t3Time = t3_cuda.Seconds();
        }

        // t_cuda.Start();
        ds->affectedNodes.assign(ds->affectedNodesSet.begin(), ds->affectedNodesSet.end());
        
        bool *d_frontierExists = nullptr;
        gpuErrchk(cudaMallocAsync((void**)&d_frontierExists, sizeof(bool), ds->adListStream));
        bool h_frontierExists = false;
        cudaMemsetAsync(d_frontierExists, 0, sizeof(bool), ds->adListStream);

        cudaStreamSynchronize(ds->adListStream);
        int PROPERTY_SIZE = ds->sizeOfNodesArrayOnCuda * sizeof(*ds->property_c);
        float *property_h;
        property_h = (float *)malloc(PROPERTY_SIZE);
        if(ds->property[source] == -1)
        {
            gpuErrchk(cudaMemcpy(property_h, ds->property_c, PROPERTY_SIZE, cudaMemcpyDeviceToHost));
            property_h[source] = 0;
            gpuErrchk(cudaMemcpy(ds->property_c, property_h, PROPERTY_SIZE, cudaMemcpyHostToDevice));
        }


        int FRONTIER_SIZE = ds->num_nodes * sizeof(*ds->frontierArr_c);
        
        bool* visited_c;
        gpuErrchk(cudaMallocAsync((void**)&visited_c, FRONTIER_SIZE, ds->adListStream));
        cudaMemsetAsync(visited_c, 0, FRONTIER_SIZE, ds->adListStream);

        // gpuErrchk(cudaMalloc((void**)&ds->affected_c, FRONTIER_SIZE));
        // gpuErrchk(cudaMemcpy(ds->affected_c, ds->affected.begin(), FRONTIER_SIZE, cudaMemcpyHostToDevice));
        
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
        // t_cuda.Stop();

        // t.Start();
        const int BLK_SIZE = 512;
        dim3 blkSize(BLK_SIZE);
        dim3 gridSize((affectedNum + BLK_SIZE - 1) / BLK_SIZE);

        BFSIter0_cuda<<<gridSize, blkSize, 0, ds->adListStream>>>(d_affectedNodes, d_affectedNum, d_frontierNum, d_frontierNodes,
                    d_newFrontierNodes, ds->affected_c, ds->property_c, ds->d_NeighborsArrays, ds->d_NeighborSizes,
                    visited_c, ds->num_nodes, ds->num_edges,
                    ds->frontierArr_c, d_frontierExists);

        cudaDeviceSynchronize();
        cudaMemsetAsync(visited_c, 0, FRONTIER_SIZE, ds->adListStream);
        
        // cudaFree(ds->d_InNodes);
        // cudaFree(ds->d_in_neighbors);
        // cudaFree(ds->affected_c);
        cudaFreeAsync(d_affectedNodes, ds->adListStream);
        cudaFreeAsync(d_affectedNum, ds->adListStream);
        
        int frontierNum = 0;
        gpuErrchk(cudaMemcpyAsync(&frontierNum, d_frontierNum, sizeof(int), cudaMemcpyDeviceToHost, ds->adListStream));
        gpuErrchk(cudaMemcpyAsync(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost, ds->adListStream));
        int* d_newFrontierNum;
        gpuErrchk(cudaMallocAsync(&(d_newFrontierNum), sizeof(int), ds->adListStream));
        cudaMemsetAsync(d_newFrontierNum, 0, sizeof(int), ds->adListStream);
        
        while(h_frontierExists){        
            h_frontierExists = false;     
            //std::cout << "Queue not empty, Queue size: " << queue.size() << std::endl;
            cudaMemsetAsync(visited_c, 0, FRONTIER_SIZE, ds->adListStream);
            cudaMemsetAsync(d_frontierExists, 0, sizeof(bool), ds->adListStream);
            cudaStreamSynchronize(ds->adListStream);
            gridSize = (frontierNum + BLK_SIZE - 1) / BLK_SIZE;
            dynBfs_kerenel<<<gridSize, blkSize, 0, ds->adListStream>>>(d_frontierNum, d_frontierNodes, d_newFrontierNodes, d_newFrontierNum, ds->property_c,  ds->d_NeighborsArrays, ds->d_NeighborSizes,
                        visited_c, ds->num_nodes, ds->num_edges,
                        d_frontierExists);

            cudaDeviceSynchronize();
            gpuErrchk(cudaMemcpy(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost));

            swap(d_frontierNum, d_newFrontierNum);
            gpuErrchk(cudaMemcpyAsync(&frontierNum, d_frontierNum, sizeof(int), cudaMemcpyDeviceToHost, ds->adListStream));
            cudaMemsetAsync(d_newFrontierNum, 0, sizeof(int), ds->adListStream);

            swap(d_frontierNodes, d_newFrontierNodes);
            cudaMemsetAsync(d_newFrontierNodes, 0, NODES_SIZE, ds->adListStream);
        }    

        cudaStreamSynchronize(ds->adListStream);
        gpuErrchk(cudaMemcpy(&(ds->property[0]), ds->property_c, ds->num_nodes * sizeof(*ds->property_c), cudaMemcpyDeviceToHost));
        // std::copy(property_h, property_h + ds->num_nodes, ds->property.begin());
        free(property_h);
        
        cudaFreeAsync(visited_c, ds->adListStream);
        cudaFreeAsync(d_frontierExists, ds->adListStream);

        cudaFreeAsync(d_frontierNum, ds->adListStream);
        cudaFreeAsync(d_frontierNodes, ds->adListStream);
        cudaFreeAsync(d_newFrontierNodes, ds->adListStream);
        cudaFreeAsync(d_newFrontierNum, ds->adListStream);

        #pragma omp for schedule(dynamic, 16)
        for(NodeID i = 0; i < ds->num_nodes; i++){
            ds->affected[i] = false;  
        }

        // t.Stop();  

        // Timer t4_cuda;
        // t4_cuda.Start();
        #pragma omp for schedule(dynamic, 16)
        for(NodeID i : ds->affectedNodesSet){
            ds->out_neighborsDelta[i].clear();  
        }
        (ds->affectedNodes).clear();
        ds->affectedNodesSet.clear();
        // t4_cuda.Stop();

        // ofstream out("JustAlg.csv", std::ios_base::app);   
        // out << t.Seconds() << std::endl;    
        // out.close();

        // ofstream cuda_out("InitFrontierMisc.csv", std::ios_base::app);   
        // cuda_out << t_cuda.Seconds() << std::endl;    
        // cuda_out.close();

        // ofstream cuda1_out("resizeAndCopyToCudaMemory.csv", std::ios_base::app);   
        // cuda1_out << t1Time << std::endl;    
        // cuda1_out.close();

        // ofstream cuda2_out("copyToCudaMemory.csv", std::ios_base::app);   
        // cuda2_out << t2Time << std::endl;    
        // cuda2_out.close();

        // ofstream cuda3_out("updateNeighbors.csv", std::ios_base::app);   
        // cuda3_out << t3Time << std::endl;    
        // cuda3_out.close();

        // ofstream cuda4_out("clearAffected.csv", std::ios_base::app);   
        // cuda4_out << t4_cuda.Seconds() << std::endl;    
        // cuda4_out.close();

        // std::cout << "Exiting!" << std::endl;
        gpuErrchk(cudaDeviceSynchronize());
    }
    ds->cudaNeighborsConditional.notify_all();

    t_cuda.Stop();
    ofstream out("Alg.csv", std::ios_base::app);   
    out << t_cuda.Seconds() << std::endl;    
    out.close();
} 


void swap(bool* &a, bool* &b){
  bool *temp = a;
  a = b;
  b = temp;
}

__global__ void bfs_kerenel(NodeID *nodes, NodeID *d_out_neighbors, bool *frontierArr, bool *newFrontierArr, float *property, bool* frontierExists, int64_t numNodes, int64_t numEdges, int level)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if (idx < numNodes)
    {
        if(frontierArr[idx])
        {
            int iEnd = (idx + 1) < numNodes ? nodes[idx+1] : numEdges;
            for(int i = nodes[idx]; i < iEnd; i++)
            {
                if (property[d_out_neighbors[i]] == -1)
                {
                    atomicAdd(&property[d_out_neighbors[i]], (float)(level + 1));
                    atomic_CAS(&newFrontierArr[d_out_neighbors[i]], false, true);
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
void BFSStartFromScratch(T* ds, NodeID source){  
    //std::cout << "Source " << source << std::endl;
    std::cout << "Running BFS from scratch" << std::endl;

    Timer t;
    t.Start(); 

    bool *d_frontierExists;
    gpuErrchk(cudaMalloc(&d_frontierExists, sizeof(bool)));
    bool h_frontierExists = true;
    gpuErrchk(cudaMemcpy(d_frontierExists, &h_frontierExists, sizeof(bool), cudaMemcpyHostToDevice));

    int PROPERTY_SIZE = ds->num_nodes * sizeof(*ds->property_c);
    gpuErrchk(cudaMalloc(&ds->property_c, PROPERTY_SIZE));
    float *property_h;
    property_h = (float *)malloc(PROPERTY_SIZE);
    std::fill(property_h, property_h + ds->num_nodes, -1);
    property_h[source] = 0;
    gpuErrchk(cudaMemcpy(ds->property_c, property_h, PROPERTY_SIZE, cudaMemcpyHostToDevice));

    int FRONTIER_SIZE = ds->num_nodes * sizeof(*ds->frontierArr_c);
    gpuErrchk(cudaMalloc((void**)&ds->frontierArr_c, FRONTIER_SIZE));
    bool *frontierArr_h;
    frontierArr_h = (bool *)malloc(FRONTIER_SIZE);
    std::fill(frontierArr_h, frontierArr_h + ds->num_nodes, false);
    frontierArr_h[source] = true;
    gpuErrchk(cudaMemcpy(ds->frontierArr_c, frontierArr_h, FRONTIER_SIZE, cudaMemcpyHostToDevice));
    frontierArr_h[source] = false;

    bool* newFrontierArr_c;
    gpuErrchk(cudaMalloc((void**)&newFrontierArr_c, FRONTIER_SIZE));
    gpuErrchk(cudaMemcpy(newFrontierArr_c, frontierArr_h, FRONTIER_SIZE, cudaMemcpyHostToDevice));

    int NODES_SIZE = ds->num_nodes * sizeof(NodeID);
    int NEIGHBOURS_SIZE = ds->num_edges * sizeof(NodeID);

    ds->h_nodes = (NodeID *)malloc(NODES_SIZE);
    ds->h_out_neighbors = (NodeID *)malloc(NEIGHBOURS_SIZE);
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

    // ofstream myfile;
    // myfile.open("/home/eurocom/dataset/v2_cuda_neighbors.csv");
    // if(ds->num_nodes == 334863)
    // {
    //     for(int i = 0; i < ds->num_edges; i++)
    //     {
    //         myfile << i << ", " << ds->h_out_neighbors[i] << "\n";
    //     }
    // }
    // myfile.close();


    gpuErrchk(cudaMalloc(&(ds->d_nodes), NODES_SIZE));
    gpuErrchk(cudaMemcpy(ds->d_nodes, ds->h_nodes, NODES_SIZE, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&(ds->d_out_neighbors), NEIGHBOURS_SIZE));
    gpuErrchk(cudaMemcpy(ds->d_out_neighbors, ds->h_out_neighbors, NEIGHBOURS_SIZE, cudaMemcpyHostToDevice));

    const int BLK_SIZE = 512;
    dim3 blkSize(BLK_SIZE);
    dim3 gridSize((ds->num_nodes + BLK_SIZE - 1) / BLK_SIZE);
    // NodeID *d_nodes =  thrust::raw_pointer_cast(&ds->d_nodes[0]);
    // NodeID *d_out_neighbors =  thrust::raw_pointer_cast(&ds->d_out_neighbors[0]);
    // bool *d_frontierArr =  thrust::raw_pointer_cast(&ds->frontierArr_c[0]);
    // float *d_property = thrust::raw_pointer_cast(&ds->property_c[0]);
    int level = 1;
    while(h_frontierExists){       
        //std::cout << "Queue not empty, Queue size: " << queue.size() << std::endl;
        h_frontierExists = false;
        gpuErrchk(cudaMemcpy(d_frontierExists, &h_frontierExists, sizeof(bool), cudaMemcpyHostToDevice));
        bfs_kerenel<<<gridSize, blkSize>>>(ds->d_nodes, ds->d_out_neighbors, ds->frontierArr_c, newFrontierArr_c, ds->property_c, d_frontierExists, ds->num_nodes, ds->num_edges, level);
        cudaDeviceSynchronize();
        swap(ds->frontierArr_c, newFrontierArr_c);
        gpuErrchk(cudaMemcpy(newFrontierArr_c, frontierArr_h, FRONTIER_SIZE, cudaMemcpyHostToDevice));
        level++;
        gpuErrchk(cudaMemcpy(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost));
    }
    std::cout << "Exiting kernel" << std::endl;
    
    t.Stop();    
    ofstream out("Alg.csv", std::ios_base::app);   
    out << t.Seconds() << std::endl;    
    out.close();

    gpuErrchk(cudaMemcpy(&(ds->property[0]), ds->property_c, PROPERTY_SIZE, cudaMemcpyDeviceToHost));
    // std::copy(property_h, property_h + ds->num_nodes, ds->property.begin());
    
    cudaFree(d_frontierExists);
    cudaFree(ds->property_c);
    cudaFree(ds->frontierArr_c);
    cudaFree(newFrontierArr_c);
    cudaFree(ds->d_nodes);
    cudaFree(ds->d_out_neighbors);

    free(property_h);
    free(frontierArr_h);
    free(ds->h_nodes);
    free(ds->h_out_neighbors);
}
#endif  // DYN_BFS_H_    
