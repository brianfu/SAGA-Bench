using namespace std;

#include <unistd.h>
#include <fstream>
#include <cstring>
#include <mutex>
#include <thread>

#include "builder.h"
#include "fileReader.h"
#include "topDataStruc_cu.h"
#include "parser.h"

/* Main thread that launches everything else */

int main(int argc, char* argv[])
{    
    LOG_PRINT("Spawning thread 0");
    cmd_args opts = parse(argc, argv);
    ifstream file(opts.filename);
    if (!file.is_open()) {
        cout << "Couldn't open file " << opts.filename << endl;
	    exit(-1);
    }    

    std::mutex q_lock;
    EdgeBatchQueue queue;
    MapTable VMAP;

    bool still_reading = true;  
    dataStruc* struc = createDataStruc(opts.type, opts.weighted, opts.directed, opts.num_nodes, opts.num_threads);
    int batch_id = 0;
    NodeID lastAssignedNodeID = -1;

    // Spawn consumer thread (Update phase)
    std::thread t1(dequeAndInsertEdge, opts.type, struc, &queue, &q_lock, opts.algorithm, &still_reading);   
    
    // Set thread t1 to run exclusively on CPU core 1
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);
    int rc = pthread_setaffinity_np(t1.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
    

    LOG_PRINT("Thread 0: Starting CSV read from: " + opts.filename);
    while (!file.eof()) {        
        EdgeList el = readBatchFromCSV(
	    file,
	    opts.batch_size,
	    batch_id,
	    opts.weighted,
	    VMAP,
	    lastAssignedNodeID);
	q_lock.lock();     
        queue.push(el);
	q_lock.unlock();
	batch_id++;          
    }
    file.close();
    LOG_PRINT("Thread 0: CSV read finished");

    bool allEmpty = false;
    while (!allEmpty) {
        LOG_PRINT("Thread 0: Taking lock");
        q_lock.lock();

        allEmpty = queue.empty();
        LOG_PRINT("Thread 0: queue.empty() is: " << std::boolalpha << allEmpty);

        LOG_PRINT("Thread 0: Releasing lock");
        q_lock.unlock();

        sleep(20);
    }
    
    still_reading = false;

    LOG_PRINT("Thread 0: Ready to join");
    t1.join();
    LOG_PRINT("Thread 0: Threads joined");
    
    //cout << "Started printing queues " << endl;
    //printEdgeBatchQueue(queue);
    //cout << "Done printing queues " << endl;    
    struc->print();
}