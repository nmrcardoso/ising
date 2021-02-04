#include <iostream>
#include <math.h> 
#include <time.h> 
#include <random>
#include <vector> 
#include <fstream>
#include <omp.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "timer.h"
#include "cuda_error_check.h"
#include "alloc.h"
#include "reduce_block_1d.h"
#include "complex.h"

//#include "special_functions.cuh"

#include "update.h"
#include "enum.h"

#include "tune.h"
#include "index.h"


namespace ising{







__global__ void kernel_InitLattice(int *lat, cuRNGState *rng_state, int type){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= HalfVolume() ) return ;
    
    if(type == 0){
		cuRNGState localState = rng_state[ id ];
		for(int parity = 0; parity < 2; ++parity){
			double rand = Random<double>(localState, 0., 1.);
			int spin = -1;
			if(rand > 0.5) spin = 1.0;
			lat[id + parity * HalfVolume()] = spin;
		}
		rng_state[ id ] = localState;
	}
	else{
		for(int parity = 0; parity < 2; ++parity)
			lat[id + parity * HalfVolume()] = type;
	}
}


/*
type = -1   , all spins at -1
type = 1    , all spins at 1
type = 0    , random 
*/
void InitLattice(Array<int> *lat, CudaRNG *rng_state, int type){
	if(!((type == 0) || (type == 1) || (type == -1))){
		std::cout << "Invalid option: " << type << ". Should only be -1, 0 or 1." << std::endl;
		exit(1);
	}
	if(lat->Location()==Host){
		std::uniform_real_distribution<double> rand01(0,1);
		#pragma omp parallel for
		for(int i = 0; i < lat->Size(); i++){
			if(type==0){
				double b = rand01(generator[omp_get_thread_num()]);
				int spin = -1;
				if(b > 0.5) spin = 1.0;
				lat->at(i) = spin;
			}
			else lat->at(i) = type;
		
		}
	}
	else{
		// kernel number of threads per block and number os blocks
		int threads = 128;
		int blocks = (HalfVolume() + threads - 1) / threads;
		kernel_InitLattice<<<blocks,threads>>>(lat->getPtr(), rng_state->getPtr(), type);
	}
}




InlineHostDevice int Neigbours(const int *lat, const int id, const int parity){
	int spinsum = 0.;	
	for(int nu = 0; nu < Dirs(); nu++){
		spinsum += lat[indexEO_neg(id, parity, nu, 1)];
		spinsum += lat[indexEO_neg(id, parity, nu, -1)];
	}
	return spinsum;
}


InlineHostDevice void UpdateSite(int* lat, int id, int parity, double rand){

	int spin = lat[id + parity * HalfVolume()];
	int spinsum = Neigbours(lat, id, parity);
	double inter = -Jconst() * double(spin * spinsum);

	if(1){
		if( inter > 0 ) spin = -spin;
		else{
			double boltz = exp(2.0 * inter * Beta());
			if( rand < boltz ) spin = -spin;
		}
	}
	else{    
		double boltz = exp(2.0 * inter * Beta());
		if( boltz > rand ) spin = -spin;
	}
	lat[id + parity * HalfVolume()] = spin;
}



__global__ void kernel_metropolis(int *lat, int parity, cuRNGState *rng_state){
    uint id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= HalfVolume() ) return ;
    cuRNGState localState = rng_state[ id ];
	double rand = Random<double>(localState);
    rng_state[ id ] = localState;
    
    UpdateSite(lat, id, parity, rand);
}

	
	

	
	




class Metropolis: Tunable{
private:
	Array<int>* lat;
	CudaRNG *rng_state;
	int metrop;
	int parity;
	int mu;
	int size;
	double timesec;
#ifdef TIMMINGS
    Timer time;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	kernel_metropolis<<<tp.grid,tp.block, 0, stream>>>(lat->getPtr(), parity, rng_state->getPtr());
}
public:
   Metropolis(Array<int>* lat, CudaRNG *rng_state, int metrop) : lat(lat), rng_state(rng_state), metrop(metrop){
	size = HalfVolume();
	timesec = 0.0;  
}
   ~Metropolis(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	for(int m = 0; m < metrop; ++m)
	for(parity = 0; parity < 2; ++parity)
	    apply(stream);
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync();
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
}
   void Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "Metropolis:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size;
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() {
  	lat->Backup();
	rng_state->Backup();
  }
  void postTune() {  
	lat->Restore();
	rng_state->Restore();
 }

};



void UpdateLattice(Array<int> *lattice, CudaRNG *rng_state, int metrop){

	if(lattice->Location()==Host){
		std::uniform_real_distribution<double> rand01(0,1);		
		int* lat = lattice->getPtr();
		for(int m = 0; m < metrop; ++m)
		for(int parity = 0; parity < 2; ++parity){
			#pragma omp parallel for
			for(int id = 0; id < HalfVolume(); id++){
				double rand = rand01(generator[omp_get_thread_num()]);
				UpdateSite(lat, id, parity, rand);
			}
		}
	}
	else{
		// metropolis algorithm
		Metropolis mtp(lattice, rng_state, metrop);
		mtp.Run();
	}
}







































__global__ void kernel_calcValues(int *lat, int *avgspin, double *energy){
    uint ids = threadIdx.x + blockDim.x * blockIdx.x;
    int id = ids;
    int parity = 0;
	if( id >= HalfVolume() ) {
		id -= HalfVolume();
		parity = 1;
	}
	int spin = 0;
	int spinsum = 0;
	if( ids < Volume() ){
		spin = lat[id + parity * HalfVolume()];
		spinsum = Neigbours(lat, id, parity);
	}
	double inter = -Jconst() * double(spin * spinsum);
	
	reduce_block_1d<int>(avgspin, spin);
	reduce_block_1d<double>(energy, spinsum);
}





class CalcValues: Tunable{
private:
	Array<int>* lat;
	double *avgspin;
	double *energy;
	int *avgspin_dev;
	double *energy_dev;
	int size;
	double timesec;
#ifdef TIMMINGS
    Timer time;
#endif

   unsigned int sharedBytesPerThread() const { return sizeof(double); }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	cudaSafeCall(cudaMemset(avgspin_dev, 0, sizeof(int)));
	cudaSafeCall(cudaMemset(energy_dev, 0, sizeof(double)));
	kernel_calcValues<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), avgspin_dev, energy_dev);
}
public:
   CalcValues(Array<int>* lat, double *avgspin, double *energy) : lat(lat), avgspin(avgspin), energy(energy) {
	size = Volume();
	avgspin_dev = (int*)dev_malloc(sizeof(int));
	energy_dev = (double*)dev_malloc(sizeof(double));
	timesec = 0.0;  
}
   ~CalcValues(){ dev_free(avgspin_dev); dev_free(energy_dev);};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	int avg_spin = 0;
	cudaSafeCall(cudaMemcpy(&avg_spin, avgspin_dev, sizeof(int), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(energy, energy_dev, sizeof(double), cudaMemcpyDeviceToHost));
	*avgspin = double(avg_spin) / double(Volume());
	*energy /= double(Volume());
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
}
   void Run(){ Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "CalcValues:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size;
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { }
  void postTune() { }

};








void ComputeAVG(Array<int> *lattice, double *avgspin, double *energy){
	if(lattice->Location()==Host){
		std::uniform_real_distribution<double> rand01(0,1);		
		int* lat = lattice->getPtr();
		int mag = 0;
		double energy_s = 0.0;
		for(int parity = 0; parity < 2; ++parity){
			#pragma omp parallel for shared(lattice, parity)  reduction(+: mag, energy_s)
			for(int id = 0; id < HalfVolume(); id++){
				int spin = lat[id + parity * HalfVolume()];
				int spinsum = Neigbours(lat, id, parity);
				mag += spin;			
				energy_s += -Jconst() * double(spin * spinsum);
			}
		}
		*avgspin = double(mag) / double(Volume());
		*energy = energy_s / double(Volume());
	}
	else{
		CalcValues calc(lattice, avgspin, energy);
		calc.Run();
	}
} 





}
