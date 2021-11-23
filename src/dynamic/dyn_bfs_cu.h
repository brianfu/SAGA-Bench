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
    cudaMallocManaged((void**)&frontierExists, sizeof(bool));

    ds->property_c.resize(ds->num_nodes);
    ds->frontierArr_c.resize(ds->num_nodes);
    thrust::fill(ds->property_c.begin(), ds->property_c.end(), -1);
    thrust::fill(ds->frontierArr_c.begin(), ds->frontierArr_c.end(), false);
    ds->property_c[source] = 0;    
    ds->frontierArr_c[source] = true;
    *frontierExists = true;

    for(auto outNeighbor = ds->out_neighbors.begin(); outNeighbor != ds->out_neighbors.end(); outNeighbor++)
    {
        ds->h_nodes.push_back(ds->h_out_neighbors.size());
        for(auto node = (*outNeighbor).begin(); node != (*outNeighbor).end(); node++)
        {
            ds->h_out_neighbors.push_back((*node).getNodeID());
        }
    }
    ds->d_nodes.resize(ds->h_nodes.size());
    ds->d_out_neighbors.resize(ds->h_out_neighbors.size());
    ds->d_nodes = ds->h_nodes;
    ds->d_out_neighbors = ds->h_out_neighbors;

    dim3 BLK_SIZE(512);
    dim3 gridSize(ds->num_nodes / 512);
    NodeID *d_nodes =  thrust::raw_pointer_cast(&ds->d_nodes[0]);
    NodeID *d_out_neighbors =  thrust::raw_pointer_cast(&ds->d_out_neighbors[0]);
    bool *d_frontierArr =  thrust::raw_pointer_cast(&ds->frontierArr_c[0]);
    float *d_property = thrust::raw_pointer_cast(&ds->property_c[0]);
    while(*frontierExists){       
        //std::cout << "Queue not empty, Queue size: " << queue.size() << std::endl;
        *frontierExists = false;
        bfs_kerenel<<<gridSize, BLK_SIZE>>>(d_nodes, d_out_neighbors, d_frontierArr, d_property, frontierExists, ds->num_nodes, ds->num_edges);
        cudaDeviceSynchronize();
    }

    t.Stop();    
    ofstream out("Alg.csv", std::ios_base::app);   
    out << t.Seconds() << std::endl;    
    out.close();
}
#endif  // DYN_BFS_H_    
