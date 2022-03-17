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

/* Algorithm: Incremental PageRank and PageRank starting from scratch */

// typedef float Rank;
__device__ __host__ const float kDamp = 0.85;
__device__ __host__ const float PRThreshold = 0.0000001;  

template<typename T> 
void PRIter0(T* ds, SlidingQueue<NodeID>& queue, Rank base_score)
{   
    pvector<Rank> outgoing_contrib(ds->num_nodes, 0);
    pvector<bool> visited(ds->num_nodes, false);
#pragma omp parallel for schedule(dynamic, 64)
    for(NodeID n=0; n < ds->num_nodes; n++) {    
        outgoing_contrib[n] = ds->property[n] / (ds->out_degree(n));      
    }

#pragma omp parallel     
    {
        QueueBuffer<NodeID> lqueue(queue);
#pragma omp for schedule(dynamic, 64)
        for (NodeID n=0; n < ds->num_nodes; n++) {
            if (ds->affected[n]) {
                Rank old_rank = ds->property[n];
                Rank incoming_total = 0;
                for(auto v: in_neigh(n, ds)){
                    incoming_total += outgoing_contrib[v];
                }
                    
                ds->property[n] = base_score + kDamp * incoming_total;                      
                bool trigger = fabs(ds->property[n] - old_rank) > PRThreshold; 
                if (trigger) {
                    //put the out-neighbors into active list 
                    for (auto v: out_neigh(n, ds)) {                        
                        bool curr_val = visited[v];
                        if (!curr_val) {
                            if (compare_and_swap(visited[v], curr_val, true)) 
				lqueue.push_back(v);
                        }
                    }                                             
                }    
            }
        }
        lqueue.flush();
    }    
}

template<typename T>
void dynPRAlg(T* ds)
{     
//     std::cout << "Running dynamic PR" << std::endl;  

//     Timer t;
//     t.Start();

//     SlidingQueue<NodeID> queue(ds->num_nodes);       
//     const Rank base_score = (1.0f - kDamp)/(ds->num_nodes); 
//     // set all new vertices' rank to 1/num_nodes, otherwise reuse old values 
// #pragma omp parallel for schedule(dynamic, 64)
//     for (NodeID n = 0; n < ds->num_nodes; n++) {
//         if (ds->property[n] == -1) {
//             ds->property[n] = 1.0f/(ds->num_nodes);
//         }
//     } 

//     // Iteration 0 only on affected vertices    
//     PRIter0(ds, queue, base_score); 
//     //cout << "Done iter 0" << endl;
//     queue.slide_window();
//     /*ofstream out("queueSizeParallel.csv", std::ios_base::app);   
//     out << queue.size() << std::endl;
//     std::cout << "Queue Size: " << queue.size() << std::endl;
//     out.close();*/
//     // Iteration 1 onward, process vertices in the queue 
//     while (!queue.empty()) {         
//         //std::cout << "Not empty queue, Queue Size:" << queue.size() << std::endl;
//         pvector<Rank> outgoing_contrib(ds->num_nodes, 0);
//         pvector<bool> visited(ds->num_nodes, false); 
//         #pragma omp parallel for 
//         for (NodeID n=0; n < ds->num_nodes; n++) { 
//             outgoing_contrib[n] = ds->property[n]/(ds->out_degree(n));      
//         }     
//         #pragma omp parallel 
//         {
//             QueueBuffer<NodeID> lqueue(queue);   

//             #pragma omp for schedule(dynamic, 64) 
//             for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
//                 NodeID n = *q_iter;
//                 Rank old_rank = ds->property[n];
//                 Rank incoming_total = 0;
//                 for(auto v: in_neigh(n, ds))
//                     incoming_total += outgoing_contrib[v];
//                 ds->property[n] = base_score + kDamp * incoming_total;                      
//                 bool trigger = fabs(ds->property[n] - old_rank) > PRThreshold; 
//                 if (trigger) {
//                     //put the out-neighbors into active list 
//                     for (auto v: out_neigh(n, ds)) {
//                         bool curr_val = visited[v];
//                         if (!curr_val) {
//                             if (compare_and_swap(visited[v], curr_val, true)) 
//                                 lqueue.push_back(v);
//                         }     
//                     }     
//                 }
//             }
//             lqueue.flush();
//         }
//         queue.slide_window();               
//     }   
>>>>>>> Changes for Page Rank From Scratch
    
//     // clear affected array to get ready for the next update round
// #pragma omp parallel for schedule(dynamic, 64)
//     for (NodeID i = 0; i < ds->num_nodes; i++) {
//         ds->affected[i] = false;
//     }

//      t.Stop();    
//     ofstream out("Alg.csv", std::ios_base::app);   
//     out << t.Seconds() << std::endl;    
//     out.close();
//     //cout << "Done" << endl;    
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
        // Rank currError = *error;
        // while(!(currError == atomicAdd(error, fabsf(newRank - old_rank))))
        // {
        //     currError = *error;
        // }
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
    // float *property_h;
    // property_h = (float *)malloc(PROPERTY_SIZE);
    // std::fill(property_h, property_h + ds->num_nodes, -1);
    // gpuErrchk(cudaMemcpy(ds->property_c, property_h, PROPERTY_SIZE, cudaMemcpyHostToDevice));

    int NODES_SIZE = ds->num_nodes * sizeof(NodeID);
    int NEIGHBOURS_SIZE = ds->num_edges * sizeof(NodeID);

    cudaMallocHost((void**)&ds->h_nodes, NODES_SIZE);
    cudaMallocHost((void**)&ds->h_out_neighbors, NEIGHBOURS_SIZE);
    // ds->h_nodes = (NodeID *)malloc(NODES_SIZE);
    // ds->h_out_neighbors = (NodeID *)malloc(NEIGHBOURS_SIZE);
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

// #pragma omp parallel for
//     for (NodeID n = 0; n < ds->num_nodes; n++) {
//         ds->property[n] = 1.0f / (ds->num_nodes);        
//     }

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
        // if(h_error < epsilon)
        //     break;
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

//     pvector<Rank> outgoing_contrib(ds->num_nodes, 0);
//     for (int iter = 0; iter < max_iters; iter++) {
//         double error = 0;
// #pragma omp parallel for
//         for (NodeID n = 0; n < ds->num_nodes; n++) { 
//             outgoing_contrib[n] = ds->property[n]/(ds->out_degree(n));      
//         }
// #pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
//         for (NodeID u = 0; u < ds->num_nodes; u++) {
//             Rank incoming_total = 0;
//             for (NodeID v : in_neigh(u, ds))
// 		incoming_total += outgoing_contrib[v];
//             Rank old_rank = ds->property[u];
//             ds->property[u] = base_score + kDamp * incoming_total;
//             error += fabs(ds->property[u] - old_rank);
//         }
//         //std::cout << "Epsilon: "<< epsilon << std::endl;
//         //printf(" %2d    %lf\n", iter, error);
//         if (error < epsilon)
// 	    break;
//     } 

    t.Stop();    
    ofstream out("Alg.csv", std::ios_base::app);   
    out << t.Seconds() << std::endl;    
    out.close();
}

#endif // DYN_PR_H