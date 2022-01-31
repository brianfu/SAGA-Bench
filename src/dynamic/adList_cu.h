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


// T can be either node or nodeweight
template <typename T>
class adList_cu: public dataStruc {
    private:                
      bool vertexExists(const Edge& e, bool source);
      void updateForNewVertex(const Edge& e, bool source);
      void updateForExistingVertex(const Edge& e, bool source);      
      void initProperties(int* property, int numNodes);
      
    public: 
      void resizeAndCopyToCudaMemory();
      void copyToCudaMemory();
      void updateNeighbors();

      int* property_c; 
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
      std::vector<std::vector<T>> out_neighbors;
      std::vector<std::vector<T>> in_neighbors;  
      std::vector<NodeID> affectedNodes;
      adList_cu(bool w, bool d); 
      ~adList_cu();   
      void update(const EdgeList& el) override;
      void print() override;
      int64_t in_degree(NodeID n) override;
      int64_t out_degree(NodeID n) override;
      int numberOfNodesOnCuda;
      int numberOfNeighborsOnCuda;
      int sizeOfNodesArrayOnCuda;
      NodeID** d_NeighborsArrays;
      int* d_NeighborSizes;
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
adList_cu<T>::adList_cu(bool w, bool d)
    : dataStruc(w, d), sizeOfNodesArrayOnCuda(0) { /*std::cout << "Creating AdList" << std::endl;*/ }    

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
    }
}

template <typename T>
bool adList_cu<T>::vertexExists(const Edge& e, bool source)
{
    bool exists;
    if (source)
	exists = e.sourceExists;
    else
	exists = e.destExists;
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

// template <typename T>
// void adList_cu<T>::resizeAndCopyToCudaMemory()
// {
//     if(sizeOfNodesArrayOnCuda < num_nodes)
//     {
//         int newSizeOfNodesArrayOnCuda = (sizeOfNodesArrayOnCuda == 0 ? num_nodes : sizeOfNodesArrayOnCuda) * 2;
//         int NEIGHBORS_POINTERS_SIZE = newSizeOfNodesArrayOnCuda * sizeof(NodeID*);
//         int NEIGHBORS_SIZE = newSizeOfNodesArrayOnCuda * sizeof(int);
//         // Modified verision using https://stackoverflow.com/questions/54297756/declare-and-initialize-array-of-arrays-in-cuda
//         // create intermediate host array for storage of device row-pointers
//         NodeID** h_array = (NodeID**)malloc(NEIGHBORS_POINTERS_SIZE);
//         NodeID* h_NeighborSizes = (NodeID*)malloc(NEIGHBORS_SIZE);

//         // create top-level device array pointer
//         NodeID** d_array;
//         cudaMalloc((void**)&d_array, NEIGHBORS_POINTERS_SIZE);

//         int* d_TempNeighborSizes;
//         cudaMalloc((void**)&d_TempNeighborSizes, NEIGHBORS_SIZE);

//         // allocate each device row-pointer, then copy host data to it
//         for(size_t i = 0 ; i < num_nodes ; i++){
//             cudaMalloc(&h_array[i], (out_neighbors[i]).size() * sizeof(NodeID));
//             gpuErrchk(cudaMemcpy(h_array[i], &((out_neighbors[i])[0]), (out_neighbors[i]).size() * sizeof(NodeID), cudaMemcpyHostToDevice));
//             h_NeighborSizes[i] = (out_neighbors[i]).size();
//             numberOfNeighborsOnCuda += (out_neighbors[i]).size();
//         }

//         // fixup top level device array pointer to point to array of device row-pointers
//         cudaMemcpy(d_array, h_array, NEIGHBORS_POINTERS_SIZE, cudaMemcpyHostToDevice);
//         free(h_array);

//         cudaMemcpy(d_TempNeighborSizes, h_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyHostToDevice);
//         free(h_NeighborSizes);
//         ///////

//         int* d_TempProperty;
//         int PROPERTY_SIZE = newSizeOfNodesArrayOnCuda * sizeof(*property_c);
//         gpuErrchk(cudaMalloc(&d_TempProperty, PROPERTY_SIZE));
        
//         const int BLK_SIZE = 512;
//         dim3 blkSize(BLK_SIZE);
//         dim3 gridSize((newSizeOfNodesArrayOnCuda + BLK_SIZE - 1) / BLK_SIZE);
//         initProperties<<<gridSize, blkSize>>>(d_TempProperty, newSizeOfNodesArrayOnCuda);

//         if(sizeOfNodesArrayOnCuda > 0)
//         {
//             int OLD_NEIGHBORS_POINTERS_SIZE = sizeOfNodesArrayOnCuda * sizeof(NodeID*);
//             h_array = (NodeID**)malloc(OLD_NEIGHBORS_POINTERS_SIZE);
//             cudaMemcpy(h_array, d_NeighborsArrays, OLD_NEIGHBORS_POINTERS_SIZE, cudaMemcpyDeviceToHost);

//             for(size_t i = 0 ; i < sizeOfNodesArrayOnCuda ; i++){
//                 gpuErrchk(cudaFree(h_array[i]));
//             }
//             cudaFree(d_NeighborsArrays);
//             free(h_array);

//             cudaFree(d_NeighborSizes);

//             gpuErrchk(cudaMemcpy(d_TempProperty, property_c, sizeOfNodesArrayOnCuda * sizeof(*property_c), cudaMemcpyDeviceToDevice));
//             cudaFree(property_c);
//         }
//         d_NeighborsArrays = d_array;
//         d_NeighborSizes = d_TempNeighborSizes;
//         property_c = d_TempProperty;
//         sizeOfNodesArrayOnCuda = newSizeOfNodesArrayOnCuda;
//         numberOfNodesOnCuda = num_nodes;

//         for(NodeID i = 0; i < num_nodes; i++){
//             if(affected[i])
//             {
//                 affectedNodes.push_back(i);
//             }
//         }
//     }
// }

template <typename T>
void adList_cu<T>::copyToCudaMemory()
{
    if(numberOfNodesOnCuda < num_nodes)
    {
        int numNewNodes = num_nodes - numberOfNodesOnCuda;
        int NEIGHBORS_POINTERS_SIZE = sizeOfNodesArrayOnCuda * sizeof(NodeID*);
        int NEIGHBORS_SIZE = sizeOfNodesArrayOnCuda * sizeof(int);
        NodeID** h_array = (NodeID**)malloc(NEIGHBORS_POINTERS_SIZE);
        int* h_NeighborSizes = (int*)malloc(NEIGHBORS_SIZE);

        cudaMemcpy(h_array, d_NeighborsArrays, NEIGHBORS_POINTERS_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_NeighborSizes, d_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyDeviceToHost);

        for(size_t i = numberOfNodesOnCuda ; i < num_nodes ; i++){
            cudaMalloc(&h_array[i], (out_neighbors[i]).size() * sizeof(NodeID));
            gpuErrchk(cudaMemcpy(h_array[i], &((out_neighbors[i])[0]), (out_neighbors[i]).size() * sizeof(NodeID), cudaMemcpyHostToDevice));
            h_NeighborSizes[i] = (out_neighbors[i]).size();
            numberOfNeighborsOnCuda += (out_neighbors[i]).size();
        }
        cudaMemcpy(d_NeighborsArrays, h_array, NEIGHBORS_POINTERS_SIZE, cudaMemcpyHostToDevice);
        free(h_array);

        cudaMemcpy(d_NeighborSizes, h_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyHostToDevice);
        free(h_NeighborSizes);

        numberOfNodesOnCuda = num_nodes;

        for(NodeID i = 0; i < num_nodes; i++){
            if(affected[i])
            {
                affectedNodes.push_back(i);
            }
        }
    }
}

template <typename T>
void adList_cu<T>::updateNeighbors()
{
    if(numberOfNeighborsOnCuda < num_edges/2)
    {
        int NEIGHBORS_POINTERS_SIZE = sizeOfNodesArrayOnCuda * sizeof(NodeID*);
        int NEIGHBORS_SIZE = sizeOfNodesArrayOnCuda * sizeof(int);
        NodeID** h_array = (NodeID**)malloc(NEIGHBORS_POINTERS_SIZE);
        int* h_NeighborSizes = (int*)malloc(NEIGHBORS_SIZE);

        cudaMemcpy(h_array, d_NeighborsArrays, NEIGHBORS_POINTERS_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_NeighborSizes, d_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyDeviceToHost);
        for(size_t i = 0 ; i < num_nodes ; i++){
            if(affected[i])
            {
                affectedNodes.push_back(i);
                free(h_array[i]);
                cudaMalloc(&h_array[i], (out_neighbors[i]).size() * sizeof(NodeID));
                gpuErrchk(cudaMemcpy(h_array[i], &((out_neighbors[i])[0]), (out_neighbors[i]).size() * sizeof(NodeID), cudaMemcpyHostToDevice));
                h_NeighborSizes[i] = (out_neighbors[i]).size();
            }
        }
        cudaMemcpy(d_NeighborsArrays, h_array, NEIGHBORS_POINTERS_SIZE, cudaMemcpyHostToDevice);
        free(h_array);

        cudaMemcpy(d_NeighborSizes, h_NeighborSizes, NEIGHBORS_SIZE, cudaMemcpyHostToDevice);
        free(h_NeighborSizes);
    }
}

#endif  // ADLIST_H_
