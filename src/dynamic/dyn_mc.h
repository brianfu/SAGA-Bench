#ifndef DYN_MC_H_
#define DYN_MC_H_

#include <algorithm>

#include "traversal.h"
#include "../common/timer.h"
#include "sliding_queue_dynamic.h"
#include "../common/pvector.h"

#include "adList_cu.h"
// #include "adList_cu_support.h"

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

template<typename T>
void dynMCAlg(T* ds){
    //std::cout << "Number of nodes: "<< ds->num_nodes << std::endl;   
    std::cout << "Running dynamic MC" << std::endl;
    Timer t;
    t.Start();   

    SlidingQueue<NodeID> queue(ds->num_nodes);        
    
    // Assign value of newly added vertices
    #pragma omp parallel for schedule(dynamic, 64)
    for(NodeID n = 0; n < ds->num_nodes; n++){
        if(ds->property[n] == -1){
            ds->property[n] = n;
        }
    }        

    MCIter0(ds, queue);
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
                float old_val = ds->property[n];
                float new_val = old_val;

                // calculate new value
                for(auto v: in_neigh(n, ds)){
                    new_val = std::max(new_val, ds->property[v]);
                }

                assert(new_val >= old_val);

                ds->property[n] = new_val;                
                bool trigger = (ds->property[n] > old_val); 

                if(trigger){
                    for(auto v: out_neigh(n, ds)){  
                        bool curr_val = visited[v];
                        if(!curr_val){
                            if(compare_and_swap(visited[v], curr_val, true)) 
                                 lqueue.push_back(v);
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

/*template<typename T> 
void MCStartFromScratch(const string& datatype, T* partition, bool directed){ 
    std::cout << "Running MC from scratch" << std::endl;
    #pragma omp parallel for
    for (NodeID n=0; n < partition->num_nodes; n++)
       partition->property[n] = n;
    
    int num_iter = 0;
    bool change = true;
    while(change){
        change = false;
        num_iter++;
        #pragma omp parallel for
        for(NodeID n = 0; n < partition->num_nodes; n++){
            float old_val = partition->property[n];
            float new_val = old_val;

            for(auto v: in_neigh(n, datatype, partition, partition->directed)){
                new_val = std::max(new_val, partition->property[v]);
            }

            assert(new_val >= old_val);

            partition->property[n] = new_val; 
            if(partition->property[n] != old_val) change = true;
        }
    }

    std::cout << "MCFromScratch took " << num_iter << " iterations" << std::endl;      
}*/
#endif  // DYN_MC_H_    
