#ifndef DYN_BFS_H_
#define DYN_BFS_H_

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

#include <stdio.h>

// From https://stackoverflow.com/questions/62091548/atomiccas-for-bool-implementation
static __inline__ __device__ bool atomic_CAS(bool *address, bool compare, bool val)
{
    unsigned long long addr = (unsigned long long)address;
    unsigned pos = addr & 3;  // byte position within the int
    int *int_addr = (int *)(addr - pos);  // int-aligned address
    int old = *int_addr, assumed, ival;

    bool current_value;

    do
    {
        current_value = (bool)(old & ((0xFFU) << (8 * pos)));

        if(current_value != compare) // If we expected that bool to be different, then
            break; // stop trying to update it and just return it's current value

        assumed = old;
        if(val)
            ival = old | (1 << (8 * pos));
        else
            ival = old & (~((0xFFU) << (8 * pos)));
        old = atomicCAS(int_addr, assumed, ival);
    } while(assumed != old);

    return current_value;
}

/* Algorithm: Incremental BFS and BFS starting from scratch */
__global__ void BFSIter0_cuda(NodeID* d_affectedNodes, int* d_affectedNum, int* d_frontierNum, NodeID* d_frontierNodes, NodeID* d_newFrontierNodes, bool* affected, int* property, NodeID* d_InNodes, NodeID* d_in_neighbors,
                    NodeID* d_nodes, NodeID* d_out_neighbors, bool* visited, int64_t numNodes, int64_t numEdges,
                    bool* frontierArr, bool* frontierExists) {
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    const int MAX = 214748360;
    if (idx < *d_affectedNum)
    {
        NodeID node = d_affectedNodes[idx];
        // if(affected[idx]){
            int old_depth = property[node];
            int newDepth = MAX;
            // Not using in-neghibors because graph being tested is not directed
            // int iEnd = (idx + 1) < numNodes ? d_InNodes[idx+1] : numEdges;
            // for(int i = d_InNodes[idx]; i < iEnd; i++)
            int iEnd = (node + 1) < numNodes ? d_nodes[node+1] : numEdges;
            for(int i = d_nodes[node]; i < iEnd; i++)
            {
                NodeID v = d_out_neighbors[i];
                int neighborDepth = property[v];
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
                int iOutEnd = (node + 1) < numNodes ? d_nodes[node+1] : numEdges;
                for(int j = d_nodes[node]; j < iOutEnd; j++)
                {
                    NodeID v = d_out_neighbors[j]; 
                    int curr_depth = property[v];
                    int updated_depth = newDepth + 1;
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
                        while(!(curr_depth == atomicCAS(&property[v], curr_depth, updated_depth))){
                            curr_depth = property[v];
                            if(curr_depth <= updated_depth){
                                break;
                            }
                        }
                    }
                }
            }
        }
    // }
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

__global__ void dynBfs_kerenel(int* d_frontierNum, NodeID* d_frontierNodes, NodeID* d_newFrontierNodes, int* d_newFrontierNum, int* property, NodeID* d_nodes, NodeID* d_out_neighbors,
                    bool* visited_c, int64_t num_nodes, int64_t num_edges,
                    bool* frontierExists){
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if(idx < *d_frontierNum){
        NodeID node = d_frontierNodes[idx];
        int iEnd = (node + 1) < num_nodes ? d_nodes[node+1] : num_edges;
        for(int i = d_nodes[node]; i < iEnd; i++)
        {
            NodeID v = d_out_neighbors[i]; 
            int curr_depth = property[v];
            int updated_depth = property[node] + 1;
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
                while(!(curr_depth == atomicCAS(&property[v], curr_depth, updated_depth))){
                    curr_depth = property[v];
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
    
    Timer t;
    t.Start();
    
    bool *d_frontierExists = nullptr;
    gpuErrchk(cudaMalloc((void**)&d_frontierExists, sizeof(bool)));
    bool h_frontierExists = false;
    cudaMemset(d_frontierExists, 0, sizeof(bool));

    int PROPERTY_SIZE = ds->num_nodes * sizeof(*ds->property_c);
    gpuErrchk(cudaMalloc(&ds->property_c, PROPERTY_SIZE));
    int *property_h;
    property_h = (int *)malloc(PROPERTY_SIZE);
    std::copy(ds->property.begin(), ds->property.end(), property_h);
    if(property_h[source] == -1) property_h[source] = 0;
    gpuErrchk(cudaMemcpy(ds->property_c, property_h, PROPERTY_SIZE, cudaMemcpyHostToDevice));


    int FRONTIER_SIZE = ds->num_nodes * sizeof(*ds->frontierArr_c);
    
    bool* visited_c;
    gpuErrchk(cudaMalloc((void**)&visited_c, FRONTIER_SIZE));
    cudaMemset(visited_c, 0, FRONTIER_SIZE);

    gpuErrchk(cudaMalloc((void**)&ds->affected_c, FRONTIER_SIZE));
    gpuErrchk(cudaMemcpy(ds->affected_c, ds->affected.begin(), FRONTIER_SIZE, cudaMemcpyHostToDevice));
    
    int NODES_SIZE = ds->num_nodes * sizeof(NodeID);
    int NEIGHBOURS_SIZE = ds->num_edges * sizeof(NodeID);

    ds->h_nodes = (NodeID *)malloc(NODES_SIZE);
    ds->h_out_neighbors = (NodeID *)malloc(NEIGHBOURS_SIZE);
    int neighborPosition = 0;
    int currentNode = 0;
    for(auto outNeighbor = ds->out_neighbors.begin(); outNeighbor != ds->out_neighbors.end(); outNeighbor++)
    {
        ds->h_nodes[currentNode] = neighborPosition;
        currentNode++;
        
        for(auto node = (*outNeighbor).begin(); node != (*outNeighbor).end(); node++)
        {
            ds->h_out_neighbors[neighborPosition] = (*node).getNodeID();
            neighborPosition++;
        }
    }

    // neighborPosition = 0;
    // currentNode = 0;
    
    // for(auto inNeighbor = ds->in_neighbors.begin(); inNeighbor != ds->in_neighbors.end(); inNeighbor++)
    // {
    //     ds->h_InNodes[currentNode] = neighborPosition;
    //     currentNode++;
        
    //     for(auto node = (*inNeighbor).begin(); node != (*inNeighbor).end(); node++)
    //     {
    //         ds->h_in_neighbors[neighborPosition] = (*node).getNodeID();
    //         neighborPosition++;
    //     }
    // }

    // if(ds->property[source] == -1) ds->property[source] = 0;
    gpuErrchk(cudaMalloc(&(ds->d_nodes), NODES_SIZE));
    gpuErrchk(cudaMemcpy(ds->d_nodes, ds->h_nodes, NODES_SIZE, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&(ds->d_out_neighbors), NEIGHBOURS_SIZE));
    gpuErrchk(cudaMemcpy(ds->d_out_neighbors, ds->h_out_neighbors, NEIGHBOURS_SIZE, cudaMemcpyHostToDevice));

    // gpuErrchk(cudaMalloc(&(ds->d_InNodes), NODES_SIZE));
    // gpuErrchk(cudaMemcpy(ds->d_InNodes, ds->h_InNodes, NODES_SIZE, cudaMemcpyHostToDevice));

    // gpuErrchk(cudaMalloc(&(ds->d_in_neighbors), NEIGHBOURS_SIZE));
    // gpuErrchk(cudaMemcpy(ds->d_in_neighbors, ds->h_in_neighbors, NEIGHBOURS_SIZE, cudaMemcpyHostToDevice));

    std::vector<NodeID> affectedNodes;
    for(NodeID i = 0; i < ds->num_nodes; i++){
        if(ds->affected[i])
        {
            affectedNodes.push_back(i);
        }
    }

    int affectedNum = affectedNodes.size();
    int* d_affectedNum;
    gpuErrchk(cudaMalloc(&(d_affectedNum), sizeof(int)));
    gpuErrchk(cudaMemcpy(d_affectedNum, &(affectedNum), sizeof(int), cudaMemcpyHostToDevice));

    int AFFECTED_SIZE = affectedNum * sizeof(NodeID);
    NodeID* d_affectedNodes;
    gpuErrchk(cudaMalloc(&(d_affectedNodes), AFFECTED_SIZE));
    gpuErrchk(cudaMemcpy(d_affectedNodes, &(affectedNodes[0]), AFFECTED_SIZE, cudaMemcpyHostToDevice));
    std::cout <<"Running dynamic BFS " << std::endl;

    int* d_frontierNum;
    gpuErrchk(cudaMalloc(&(d_frontierNum), sizeof(int)));
    cudaMemset(d_frontierNum, 0, sizeof(int));

    NodeID* d_frontierNodes;
    gpuErrchk(cudaMalloc(&d_frontierNodes, NODES_SIZE));
    cudaMemset(d_frontierNodes, 0, NODES_SIZE);

    NodeID* d_newFrontierNodes;
    gpuErrchk(cudaMalloc(&d_newFrontierNodes, NODES_SIZE));
    cudaMemset(d_newFrontierNodes, 0, NODES_SIZE);

    const int BLK_SIZE = 512;
    dim3 blkSize(BLK_SIZE);
    dim3 gridSize((affectedNum + BLK_SIZE - 1) / BLK_SIZE);

    BFSIter0_cuda<<<gridSize, blkSize>>>(d_affectedNodes, d_affectedNum, d_frontierNum, d_frontierNodes, d_newFrontierNodes, ds->affected_c, ds->property_c, ds->d_InNodes, ds->d_in_neighbors,
                ds->d_nodes, ds->d_out_neighbors, visited_c, ds->num_nodes, ds->num_edges,
                ds->frontierArr_c, d_frontierExists);

    cudaDeviceSynchronize();
    cudaMemset(visited_c, 0, FRONTIER_SIZE);
    
    // cudaFree(ds->d_InNodes);
    // cudaFree(ds->d_in_neighbors);
    cudaFree(ds->affected_c);
    cudaFree(d_affectedNodes);
    cudaFree(d_affectedNum);
    
    int frontierNum = 0;
    int* d_newFrontierNum;
    gpuErrchk(cudaMalloc(&(d_newFrontierNum), sizeof(int)));
    cudaMemset(d_newFrontierNum, 0, sizeof(int));
    gpuErrchk(cudaMemcpy(&frontierNum, d_frontierNum, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost));
    
    while(h_frontierExists){        
        h_frontierExists = false;     
        //std::cout << "Queue not empty, Queue size: " << queue.size() << std::endl;
        cudaMemset(visited_c, 0, FRONTIER_SIZE);
        cudaMemset(d_frontierExists, 0, sizeof(bool));
        gridSize = (frontierNum + BLK_SIZE - 1) / BLK_SIZE;
        dynBfs_kerenel<<<gridSize, blkSize>>>(d_frontierNum, d_frontierNodes, d_newFrontierNodes, d_newFrontierNum, ds->property_c, ds->d_nodes, ds->d_out_neighbors,
                    visited_c, ds->num_nodes, ds->num_edges,
                    d_frontierExists);

        cudaDeviceSynchronize();
        gpuErrchk(cudaMemcpy(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost));

        swap(d_frontierNum, d_newFrontierNum);
        gpuErrchk(cudaMemcpy(&frontierNum, d_frontierNum, sizeof(int), cudaMemcpyDeviceToHost));
        cudaMemset(d_newFrontierNum, 0, sizeof(int));

        swap(d_frontierNodes, d_newFrontierNodes);
        cudaMemset(d_newFrontierNodes, 0, NODES_SIZE);
    }    

    gpuErrchk(cudaMemcpy(property_h, ds->property_c, PROPERTY_SIZE, cudaMemcpyDeviceToHost));
    std::copy(property_h, property_h + ds->num_nodes, ds->property.begin());
    
    cudaFree(visited_c);
    cudaFree(d_frontierExists);
    cudaFree(ds->property_c);
    cudaFree(ds->d_nodes);
    cudaFree(ds->d_out_neighbors);

    cudaFree(d_frontierNum);
    cudaFree(d_frontierNodes);
    cudaFree(d_newFrontierNodes);
    cudaFree(d_newFrontierNum);

    free(property_h);
    free(ds->h_nodes);
    free(ds->h_out_neighbors);
    free(ds->h_InNodes);
    free(ds->h_in_neighbors);

    t.Stop();  

    for(NodeID i = 0; i < ds->num_nodes; i++){
        ds->affected[i] = false;  
    }

    ofstream out("Alg.csv", std::ios_base::app);   
    out << t.Seconds() << std::endl;    
    out.close();
    std::cout << "Exiting!" << std::endl;
} 


void swap(bool* &a, bool* &b){
  bool *temp = a;
  a = b;
  b = temp;
}

__global__ void bfs_kerenel(NodeID *nodes, NodeID *d_out_neighbors, bool *frontierArr, bool *newFrontierArr, int *property, bool* frontierExists, int64_t numNodes, int64_t numEdges, int level)
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
                    atomicCAS(&property[d_out_neighbors[i]], -1, level);
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
    int *property_h;
    property_h = (int *)malloc(PROPERTY_SIZE);
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

    gpuErrchk(cudaMemcpy(property_h, ds->property_c, PROPERTY_SIZE, cudaMemcpyDeviceToHost));
    std::copy(property_h, property_h + ds->num_nodes, ds->property.begin());
    
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
