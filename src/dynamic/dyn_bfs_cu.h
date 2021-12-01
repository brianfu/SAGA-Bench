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

__global__ void bfs_kerenel(NodeID *nodes, NodeID *d_out_neighbors, bool *frontierArr, float *property, bool* frontierExists, int64_t numNodes, int64_t numEdges)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
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
                    property[d_out_neighbors[i]] = property[idx] + 1;
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

    bool *frontierExists;
    gpuErrchk(cudaMallocManaged((void**)&frontierExists, sizeof(bool)));

    int PROPERTY_SIZE = ds->num_nodes * sizeof(float);
    gpuErrchk(cudaMallocManaged((void**)&ds->property_c, PROPERTY_SIZE));
    int FRONTIER_SIZE = ds->num_nodes * sizeof(bool);
    gpuErrchk(cudaMallocManaged((void**)&ds->frontierArr_c, FRONTIER_SIZE));
    // memset(ds->property_c, -1, PROPERTY_SIZE);
    // memset(ds->frontierArr_c, false, FRONTIER_SIZE);
    thrust::fill(ds->property_c, ds->property_c + ds->num_nodes, -1);
    thrust::fill(ds->frontierArr_c, ds->frontierArr_c + ds->num_nodes, false);
    ds->property_c[source] = 0;    
    ds->frontierArr_c[source] = true;
    *frontierExists = true;

    ds->h_nodes.resize(0);
    ds->h_out_neighbors.resize(0);
    for(auto outNeighbor = ds->out_neighbors.begin(); outNeighbor != ds->out_neighbors.end(); outNeighbor++)
    {
        if(ds->h_nodes.size() == 0)
        {
            ds->h_nodes.push_back(0);
        }
        else
        {
            int start = *(ds->h_nodes.end() - 1) + (*(outNeighbor-1)).size();
            ds->h_nodes.push_back(start);
        }
        for(auto node = (*outNeighbor).begin(); node != (*outNeighbor).end(); node++)
        {
            ds->h_out_neighbors.push_back((*node).getNodeID());
        }
    }
    int NODES_SIZE = ds->h_nodes.size() * sizeof(NodeID);
    gpuErrchk(cudaMallocManaged((void**)&ds->d_nodes, NODES_SIZE));
    std::copy(ds->h_nodes.begin(), ds->h_nodes.end(), ds->d_nodes);
    int NEIGHBOURS_SIZE = ds->h_out_neighbors.size() * sizeof(NodeID);
    gpuErrchk(cudaMallocManaged((void**)&ds->d_out_neighbors, NEIGHBOURS_SIZE));
    std::cout << "Neighbour size: " << ds->h_out_neighbors.size() << std::endl;
    std::copy(ds->h_out_neighbors.begin(), ds->h_out_neighbors.end(), ds->d_out_neighbors);

    dim3 BLK_SIZE(512);
    dim3 gridSize(ds->num_nodes / 512);
    // NodeID *d_nodes =  thrust::raw_pointer_cast(&ds->d_nodes[0]);
    // NodeID *d_out_neighbors =  thrust::raw_pointer_cast(&ds->d_out_neighbors[0]);
    // bool *d_frontierArr =  thrust::raw_pointer_cast(&ds->frontierArr_c[0]);
    // float *d_property = thrust::raw_pointer_cast(&ds->property_c[0]);
    while(*frontierExists){       
        //std::cout << "Queue not empty, Queue size: " << queue.size() << std::endl;
        *frontierExists = false;
        bfs_kerenel<<<gridSize, BLK_SIZE>>>(ds->d_nodes, ds->d_out_neighbors, ds->frontierArr_c, ds->property_c, frontierExists, ds->num_nodes, ds->num_edges);
        cudaDeviceSynchronize();
    }
    std::cout << "Exiting kernel" << std::endl;
    
    t.Stop();    
    ofstream out("Alg.csv", std::ios_base::app);   
    out << t.Seconds() << std::endl;    
    out.close();

    std::copy(ds->property_c, ds->property_c + ds->num_nodes, ds->property.begin());
    cudaFree(ds->property_c);
    cudaFree(ds->frontierArr_c);
}
#endif  // DYN_BFS_H_    
