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

/* Algorithm: Incremental BFS and BFS starting from scratch */

#if 0
template<typename T> 
void BFSIter0(T* ds, SlidingQueue<NodeID>& queue){  
    pvector<bool> visited(ds->num_nodes, false);     
  
    #pragma omp parallel     
    {
        QueueBuffer<NodeID> lqueue(queue);
        #pragma omp for schedule(dynamic, 64)
        for(NodeID n=0; n < ds->num_nodes; n++){
            if(ds->affected[n]){
                float old_depth = ds->property[n];
                float new_depth = std::numeric_limits<float>::max();

                // pull new depth from incoming neighbors
                for(auto v: in_neigh(n, ds)){
                    if (ds->property[v] != -1) {
                        new_depth = std::min(new_depth, ds->property[v] + 1);
                    }
                }                                         
                
                // trigger happens if it is:
                // 1) brand new vertex with old_prop = -1 and we found a new valid min depth 
                // 2) already existing vertex and we found a new depth smaller than old depth 
                bool trigger = (
                ((new_depth < old_depth) || (old_depth == -1)) 
                && (new_depth != std::numeric_limits<float>::max())                 
                );               

                /*if(trigger){                                                 
                    ds->property[n] = new_depth; 
                    for(auto v: out_neigh(n, dataStruc, ds, directed)){
                        float curr_depth = ds->property[v];
                        float updated_depth = ds->property[n] + 1;                        
                        if((updated_depth < curr_depth) || (curr_depth == -1)){   
                            if(compare_and_swap(ds->property[v], curr_depth, updated_depth)){                                                              
                                lqueue.push_back(v); 
                            }
                        }
                    }
                }*/

                // Note: above is commented and included this new thing. 
                // Above was leading to vertices being queued redundantly
                // Above assumes updated_depth < curr_depth only once. 
                // This is not true in dynamic case because we start from affected vertices
                // whose depths are not all necessary the same.
                // In static version, the above works because static version starts from the source 
                // and we know that updated_depth < curr_depth only once. 

                if(trigger){
                    ds->property[n] = new_depth; 
                    for(auto v: out_neigh(n, ds)){
                        float curr_depth = ds->property[v];
                        float updated_depth = ds->property[n] + 1;
                        if((updated_depth < curr_depth) || (curr_depth == -1)){
                            bool curr_val = visited[v];
                            if(!curr_val){
                                if(compare_and_swap(visited[v], curr_val, true))
                                    lqueue.push_back(v);
                            }
                            while(!compare_and_swap(ds->property[v], curr_depth, updated_depth)){
                                curr_depth = ds->property[v];
                                if(curr_depth <= updated_depth){
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        lqueue.flush();
    }   
}

template<typename T>
void dynBFSAlg(T* ds, NodeID source){
    std::cout <<"Running dynamic BFS " << std::endl;
    
    Timer t;
    t.Start();
    
    SlidingQueue<NodeID> queue(ds->num_nodes);         
    if(ds->property[source] == -1) ds->property[source] = 0;
    
    BFSIter0(ds, queue);
    queue.slide_window();   
    
    while(!queue.empty()){             
        //std::cout << "Queue not empty, Queue size: " << queue.size() << std::endl;
        pvector<bool> visited(ds->num_nodes, false); 

        #pragma omp parallel
        {
            QueueBuffer<NodeID> lqueue(queue);
            #pragma omp for schedule(dynamic, 64)
            for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++){
                NodeID n = *q_iter;                        
                for(auto v: out_neigh(n, ds)){
                    float curr_depth = ds->property[v];
                    float new_depth = ds->property[n] + 1;
                    /*if((new_depth < curr_depth) || (curr_depth == -1)){
                        if(compare_and_swap(ds->property[v], curr_depth, new_depth)){                            
                            lqueue.push_back(v);
                        }
                    }*/

                    if((new_depth < curr_depth) || (curr_depth == -1)){
                        bool curr_val = visited[v];
                        if(!curr_val){
                            if(compare_and_swap(visited[v], curr_val, true))
                                    lqueue.push_back(v);
                        }

                        while(!compare_and_swap(ds->property[v], curr_depth, new_depth)){
                            curr_depth = ds->property[v];
                            if(curr_depth <= new_depth){
                                break;
                            }
                        }
                    }               
                }
            }
            lqueue.flush();
        }
        queue.slide_window();               
    }    

    // clear affected array to get ready for the next update round
    #pragma omp parallel for schedule(dynamic, 64)
    for(NodeID i = 0; i < ds->num_nodes; i++){
        ds->affected[i] = false;
    }

    t.Stop();    
    ofstream out("Alg.csv", std::ios_base::app);   
    out << t.Seconds() << std::endl;    
    out.close();
} 

#endif

__global__ void bfs_kerenel(NodeID *nodes, NodeID *d_out_neighbors, bool *frontierArr, int *property, bool* frontierExists, int64_t numNodes, int64_t numEdges)
{
    int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
    if (idx < numNodes)
    {
        if(frontierArr[idx])
        {
            frontierArr[idx] = false;
            int iEnd = (idx + 1) < numNodes ? nodes[idx+1] : numEdges;
            for(int i = nodes[idx]; i < iEnd; i++)
            {
                if (property[d_out_neighbors[i]] < 0)
                {
                    atomicCAS(&property[d_out_neighbors[i]], -1, property[idx] + 1);
                    frontierArr[d_out_neighbors[i]] = true;
                    if(!(*frontierExists))
                    {
                        atomicCAS((int *)frontierExists, (int) false, (int) true);
                    }
                }
            }
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
    cudaMemcpy(d_frontierExists, &h_frontierExists, sizeof(bool), cudaMemcpyHostToDevice);

    int PROPERTY_SIZE = ds->num_nodes * sizeof(*ds->property_c);
    gpuErrchk(cudaMalloc(&ds->property_c, PROPERTY_SIZE));
    int *property_h;
    property_h = (int *)malloc(PROPERTY_SIZE);
    std::fill(property_h, property_h + ds->num_nodes, -1);
    property_h[source] = 0;
    cudaMemcpy(ds->property_c, property_h, PROPERTY_SIZE, cudaMemcpyHostToDevice);

    int FRONTIER_SIZE = ds->num_nodes * sizeof(*ds->frontierArr_c);
    gpuErrchk(cudaMalloc((void**)&ds->frontierArr_c, FRONTIER_SIZE));
    bool *frontierArr_h;
    frontierArr_h = (bool *)malloc(FRONTIER_SIZE);
    std::fill(frontierArr_h, frontierArr_h + ds->num_nodes, false);
    frontierArr_h[source] = true;
    cudaMemcpy(ds->frontierArr_c, frontierArr_h, FRONTIER_SIZE, cudaMemcpyHostToDevice);

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
    cudaMemcpy(ds->d_nodes, ds->h_nodes, NODES_SIZE, cudaMemcpyHostToDevice);

    gpuErrchk(cudaMalloc(&(ds->d_out_neighbors), NEIGHBOURS_SIZE));
    cudaMemcpy(ds->d_out_neighbors, ds->h_out_neighbors, NEIGHBOURS_SIZE, cudaMemcpyHostToDevice);

    const int BLK_SIZE = 512;
    dim3 blkSize(BLK_SIZE);
    dim3 gridSize((ds->num_nodes + BLK_SIZE - 1) / BLK_SIZE);
    // NodeID *d_nodes =  thrust::raw_pointer_cast(&ds->d_nodes[0]);
    // NodeID *d_out_neighbors =  thrust::raw_pointer_cast(&ds->d_out_neighbors[0]);
    // bool *d_frontierArr =  thrust::raw_pointer_cast(&ds->frontierArr_c[0]);
    // float *d_property = thrust::raw_pointer_cast(&ds->property_c[0]);
    while(h_frontierExists){       
        //std::cout << "Queue not empty, Queue size: " << queue.size() << std::endl;
        h_frontierExists = false;
        cudaMemcpy(d_frontierExists, &h_frontierExists, sizeof(bool), cudaMemcpyHostToDevice);
        bfs_kerenel<<<gridSize, blkSize>>>(ds->d_nodes, ds->d_out_neighbors, ds->frontierArr_c, ds->property_c, d_frontierExists, ds->num_nodes, ds->num_edges);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_frontierExists, d_frontierExists, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    std::cout << "Exiting kernel" << std::endl;
    
    t.Stop();    
    ofstream out("Alg.csv", std::ios_base::app);   
    out << t.Seconds() << std::endl;    
    out.close();

    cudaMemcpy(property_h, ds->property_c, PROPERTY_SIZE, cudaMemcpyDeviceToHost);
    std::copy(property_h, property_h + ds->num_nodes, ds->property.begin());
    
    cudaFree(d_frontierExists);
    cudaFree(ds->property_c);
    cudaFree(ds->frontierArr_c);
    cudaFree(ds->d_nodes);
    cudaFree(ds->d_out_neighbors);

    free(property_h);
    free(frontierArr_h);
    free(ds->h_nodes);
    free(ds->h_out_neighbors);
}
#endif  // DYN_BFS_H_    
