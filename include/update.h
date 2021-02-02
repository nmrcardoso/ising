#ifndef __UPDATE__
#define __UPDATE__


#include "random.h"
#include "array.h"

namespace ising{



void ComputeAVG(Array<int> *dev_lat, double *avgspin, double *energy);


void InitLattice(Array<int> *dev_lat, CudaRNG *rng_state, int type);

void UpdateLattice(Array<int> *dev_lat, CudaRNG *rng_state, int metrop);

}

#endif
