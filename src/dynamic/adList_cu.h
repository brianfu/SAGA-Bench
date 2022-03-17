#ifndef ADLIST_CU_H_
#define ADLIST_CU_H_

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuErrchkMod(ans, iV) { gpuAssert((ans), __FILE__, __LINE__, iV); }
inline void gpuAssert(cudaError_t code, const char *file, int line, int i, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d -- i: %d\n", cudaGetErrorString(code), file, line, i);
      if (abort) exit(code);
   }
}

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include "abstract_data_struc.h"
#include "print.h"
#include <set>
#include <mutex>
#include <condition_variable>
#include <thread>

typedef float Rank;

// T can be either node or nodeweight
template <typename T>
class adList_cu: public dataStruc {
    private:                
      bool vertexExists(const Edge& e, bool source);
      void updateForNewVertex(const Edge& e, bool source);
      void updateForExistingVertex(const Edge& e, bool source);      
      void initProperties(int* property, int numNodes);
      void freeStaleArrays();
      
    public: 
      
      int* property_c; 
      Rank* propertyF_c; 
      bool* frontierArr_c;
      bool* affected_c;
      NodeID* h_nodes;
      NodeID* h_InNodes;
      NodeID* d_nodes;
      NodeID* d_InNodes;
      NodeID* h_out_neighbors;
      NodeID* h_in_neighbors;
      NodeID* d_out_neighbors;
      NodeID* d_in_neighbors;
      std::vector<std::vector<NodeID>> out_neighborsNodeID;
      std::vector<std::vector<T>> out_neighbors;
      std::vector<std::vector<T>> in_neighbors;  
      std::vector<std::vector<T>> out_neighborsDelta;
      std::vector<NodeID> affectedNodes;
      std::set<NodeID> affectedNodesSet;
      adList_cu(bool w, bool d); 
      ~adList_cu();   
      void update(const EdgeList& el) override;
      void print() override;
      int64_t in_degree(NodeID n) override;
      int64_t out_degree(NodeID n) override;
      int numberOfNodesOnCuda;
      int numberOfNeighborsOnCuda;
      int sizeOfNodesArrayOnCuda;
      Node** d_NeighborsArrays;
      Node** h_NeighborsArrays;
      int* d_NeighborSizes;
      int* h_NeighborSizes;
      int* h_NeighborCapacity;
      std::vector<bool> copyFullOrDelta;
      std::vector<int> startPosition;
      std::vector<int> copySize;

      std::vector<T*> stale_neighbors;
      std::mutex cudaNeighborsMutex;
      std::condition_variable cudaNeighborsConditional;
      bool isAlive;

      cudaStream_t adListStream;
};

// template <typename T>
// __global__ void adList_cu<T>::initProperties(int* property, int numNodes)
// {
//     int idx = threadIdx.x+ (blockDim.x*blockIdx.x);
//     int stride = blockDim.x * gridDim.x;

//     for(int i = idx; i < numNodes; i+=stride)
//         property[i] = -1;
// }

template <typename T>
inline void adList_cu<T>::freeStaleArrays()
{
    
}

template <>
inline void adList_cu<Node>::freeStaleArrays()
{
    while(isAlive)
    { 
        {
            std::unique_lock<std::mutex> lock(cudaNeighborsMutex);
            // #pragma omp for schedule(dynamic, 16)
            for(int i = 0; i < stale_neighbors.size(); i++)
            {
                cudaFreeAsync(stale_neighbors[i], adListStream);
            }
            // cudaStreamSynchronize(adListStream);
            gpuErrchk(cudaDeviceSynchronize());
            stale_neighbors.clear();
            cudaNeighborsConditional.wait(lock);
        }
    }
}


template <typename T>
inline adList_cu<T>::adList_cu(bool w, bool d)
    : dataStruc(w, d), sizeOfNodesArrayOnCuda(0), isAlive(true) {
         /*std::cout << "Creating AdList" << std::endl;*/ }    

template <>
inline adList_cu<Node>::adList_cu(bool w, bool d)
    : dataStruc(w, d), sizeOfNodesArrayOnCuda(0), isAlive(true) {
        cudaStreamCreate(&adListStream); 
        std::thread cudaFreeWorker(&adList_cu<Node>::freeStaleArrays, this);
        cudaFreeWorker.detach();
         /*std::cout << "Creating AdList" << std::endl;*/ }    

template <typename T>
adList_cu<T>::~adList_cu()
{
    if(sizeOfNodesArrayOnCuda > 0)
    {
        int OLD_NEIGHBORS_POINTERS_SIZE = sizeOfNodesArrayOnCuda * sizeof(NodeID*);
        NodeID** h_array = (NodeID**)malloc(OLD_NEIGHBORS_POINTERS_SIZE);
        cudaMemcpy(h_array, d_NeighborsArrays, OLD_NEIGHBORS_POINTERS_SIZE, cudaMemcpyDeviceToHost);

        for(size_t i = 0 ; i < num_nodes ; i++){
            gpuErrchk(cudaFree(h_array[i]));
        }
        cudaFree(d_NeighborsArrays);
        free(h_array);

        cudaFree(d_NeighborSizes);
        cudaFree(property_c);

        isAlive = false;
        cudaNeighborsConditional.notify_all();
    }
    cudaStreamDestroy(adListStream);
    cudaDeviceReset();
}

template <typename T>
bool adList_cu<T>::vertexExists(const Edge& e, bool source)
{
    bool exists;
    if (source)
	exists = e.sourceExists;
    else
	exists = e.destExists;
    affectedNodesSet.insert(e.source);
    affectedNodesSet.insert(e.destination);
    if (exists) {        
        num_edges++;        
        // return true;
        if(source) affected[e.source] = 1;
        else affected[e.destination] = 1;
        return true;
    } else {
        num_nodes++;        
        num_edges++;
        // return false;
        affected.push_back(1);
        return false;
    }  
}

template <typename T>
void adList_cu<T>::updateForNewVertex(const Edge& e, bool source)
{
    property.push_back(-1);      
    if (source || (!source && !directed)) {
        // update out_neighbors with meaningful data
        std::vector<T> edge_data;
        T neighbor;
        if (source)
	    neighbor.setInfo(e.destination, e.weight);
        else
	    neighbor.setInfo(e.source, e.weight);   
        edge_data.push_back(neighbor);
        out_neighbors.push_back(edge_data);
        out_neighborsDelta.push_back(edge_data);
        // push some junk in in_neighbors
        if (directed) {
            std::vector<T> fake_edge_data;
            in_neighbors.push_back(fake_edge_data);
        }              
    } else if (!source && directed) {
        // update in_neighbors with meaningful data
        std::vector<T> edge_data;
        T neighbor; 
        neighbor.setInfo(e.source, e.weight);        
        edge_data.push_back(neighbor);
        in_neighbors.push_back(edge_data);
        // push some junk out_neighbors
        std::vector<T> fake_edge_data;            
        out_neighbors.push_back(fake_edge_data);        
    }
}

template <typename T>
void adList_cu<T>::updateForExistingVertex(const Edge& e, bool source)
{
    NodeID index;
    if(source)
	index = e.source;
    else
	index = e.destination;
    if (source || (!source && !directed)) {
        NodeID dest;
        if(source)
	    dest = e.destination;
        else
	    dest = e.source;
        //search for the edge first in out_neighbors 
        bool found = false;
        for (unsigned int i = 0; i < out_neighbors[index].size(); i++) {
            if (out_neighbors[index][i].getNodeID()  == dest) {
                //std::cout << "Found repeating edges: source: " << index << " dest: " << dest << " new weight: " 
                //<< e.weight << std::endl;
                //std::cout << "old weight " << out_neighbors[index][i].getWeight() << std::endl;
                out_neighbors[index][i].setInfo(dest, e.weight);                
                //std::cout << "new weight " << out_neighbors[index][i].getWeight() << std::endl;
                found = true; 
                break;
            } 
        }
        if (!found) {
            T neighbor;
            neighbor.setInfo(dest, e.weight);           
            out_neighbors[index].push_back(neighbor); 
            out_neighborsDelta[index].push_back(neighbor);
        }           
    } else if (!source && directed) {
        // in_neighbors    
        // search for the edge first in in_neighbors 
        bool found = false;
        for(unsigned int i = 0; i < in_neighbors[index].size(); i++){
            if(in_neighbors[index][i].getNodeID() == e.source){
                in_neighbors[index][i].setInfo(e.source, e.weight);                
                found = true; 
                break;
            }            
        }
        if (!found) {
            T neighbor;
            neighbor.setInfo(e.source, e.weight);
            in_neighbors[index].push_back(neighbor);   
        }            
    }
}

template <typename T>
void adList_cu<T>::update(const EdgeList& el)
{
    for(auto it=el.begin(); it!=el.end(); it++){
        // examine source vertex
        bool exists = vertexExists(*it, true); 
        if(!exists) updateForNewVertex(*it, true);
        else updateForExistingVertex(*it, true);
    
        // examine destination vertex 
        bool exists1 = vertexExists(*it, false); 
        if(!exists1) updateForNewVertex(*it, false);
        else updateForExistingVertex(*it, false); 
    }               
}

template <typename T>
int64_t adList_cu<T>::in_degree(NodeID n)
{
    if(directed)
	return in_neighbors[n].size();
    else
	return out_neighbors[n].size();
}

template <typename T>
int64_t adList_cu<T>::out_degree(NodeID n)
{
    return out_neighbors[n].size();    
}

template <typename T>
void adList_cu<T>::print()
{
    std::cout << " numNodes: " << num_nodes << 
            " numEdges: " << num_edges << 
            " weighted: " << weighted << 
            " directed: " << directed << 
	std::endl;

    /*cout << "Property: "; printVector(property);    
    cout << "out_neighbors: " << endl; printVecOfVecOfNodes(out_neighbors); 
    cout << "in_neighbors: " << endl; printVecOfVecOfNodes(in_neighbors);*/
}

#endif  // ADLIST_H_
