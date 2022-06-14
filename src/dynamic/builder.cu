#include "builder.h"

#include <iostream>
#include <fstream>

#include "topAlg_cu.h"
#include "topDataStruc_cu.h"
#include "../common/timer.h"

void* dequeAndInsertEdge(
		std::string dtype,
		dataStruc *ds,
    EdgeBatchQueue *q,
    std::mutex *q_lock,
    std::string algorithm,
    bool *still_reading)
{	
	//std::cout << "Thread dequeAndInsertEdge: on CPU " << sched_getcpu() << "\n";
	LOG_PRINT("Spawning thread 1");

	int batch = 0;
	EdgeList el;
	Algorithm alg(algorithm, ds, dtype);

	LOG_PRINT("Thread 1: Taking lock");
	q_lock->lock();

	while (*still_reading || !q->empty()) {
		if (!q->empty()) {
			LOG_PRINT("Thread 1: Queue is not empty");

			el = q->front();
			LOG_PRINT("Thread 1: Popping Queue");
			q->pop();

			LOG_PRINT("Thread 1: Releasing lock");
			q_lock->unlock();

			// Update Phase
			Timer t; t.Start();
			LOG_PRINT("Thread 1: Performing Update Phase");
			ds->update(el);	
			t.Stop();

			LOG_PRINT("Thread 1: Writing to Update.csv");
			ofstream out("Update.csv", std::ios_base::app);   
			out << t.Seconds() << std::endl;    
			out.close();

			std::cout << "Updated Batch: " << batch << std::endl;
			batch++;

			// Compute Phase
			LOG_PRINT("Thread 1: Performing Compute Phase");
			alg.performAlg();

			// Write to file to compare results
			// if(ds->num_edges == 234370166)
			// {
			// 	ofstream myfile;
			// 	myfile.open("/home/tmathew/sfuhome/dataset/cudaMcDyn" + std::to_string(batch) + ".csv");
			// 	for (int i=0; i < ds->property.size(); i++)
			// 	{
			// 		myfile << i << ", " << ds->property[i] << "\n";
					
			// 	}
			// 	myfile.close();
			// }
		} 
		else {
			LOG_PRINT("Thread 1: Releasing lock");
			q_lock->unlock();

			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		LOG_PRINT("Thread 1: Taking lock");
		q_lock->lock();
	}

	LOG_PRINT("Thread 1: Releasing lock");
	q_lock->unlock();
    
    // ##################### CORRECTNESS CHECK ############################
    // LJ: batch == 138
    // Orkut: batch == 235
    // Pokec: batch == 62
    // Wiki: batch == 58
    // 15_30m: batch == 61

    /*for (int64_t i = 0; i < ds->num_nodes; ++i) {
	std::cout << "Property[" << i << "] = "
		  << ds->property[i] << std::endl;
    }*/

    /*if ((algorithm == "prdyn") && (dtype == "adListChunked")) {
	ofstream out("PRDynAdListOrkut.csv"); 
	for(int64_t i =0; i < ds->num_nodes; i++){
	    out << ds->property[i] << endl;
	}                       
	out.close(); 
    } else if ((algorithm == "prfromscratch") && (dtype == "adListChunked")) {
	ofstream out("PRStatAdListOrkut.csv"); 
	for(int64_t i =0; i < ds->num_nodes; i++){
	    out << ds->property[i] << endl;
	}                       
	out.close();
    } else if ((algorithm == "prdyn") && (dtype == "degAwareRHH")) {
	ofstream out("PRDynDarhhOrkut.csv"); 
	for(int64_t i =0; i < ds->num_nodes; i++){
	    out << ds->property[i] << endl;
	}                       
	out.close();
    } else if ((algorithm == "prfromscratch") && (dtype == "degAwareRHH")) {
	ofstream out("PRStatDarhhOrkut.csv");
	for(int64_t i =0; i < ds->num_nodes; i++){
	    out << ds->property[i] << endl;
	}                       
	out.close();
    }*/

	LOG_PRINT("Thread 1: Ready to join");
  return 0;
}
