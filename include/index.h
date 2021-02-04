#ifndef __INDEX__
#define __INDEX__

#include <cuda.h>

#include "complex.h"

#include "parameters.h"


namespace ising{




InlineHostDevice int indexId(const int x[]){
	int id = x[0];
	for(int i = 1; i < Dirs(); i++)
		id += x[i] * Offset(i);
	return id;
}

InlineHostDevice int GetParity(const int x[]){
	int oddbit = 0;
	for(int i = 0; i < Dirs(); i++) oddbit += x[i];
	return (oddbit & 1);
}




InlineHostDevice void indexNO(const int id, int x[4]){
	x[3] = id/(Grid(0) * Grid(1) * Grid(2));
	x[2] = (id/(Grid(0) * Grid(1))) % Grid(2);
	x[1] = (id/Grid(0)) % Grid(1);
	x[0] = id % Grid(0);
}



InlineHostDevice int indexNO_neg(const int id, const int mu, const int lmu){
	int x[4];
	indexNO(id, x);
	x[mu] = (x[mu]+lmu+Grid(mu)) % Grid(mu);	
	int pos = indexId(x);
	return pos;
}
InlineHostDevice int indexNO_neg(const int id, const int mu, const int lmu, const int nu, const int lnu){
	int x[4];
	indexNO(id, x);
	x[mu] = (x[mu]+lmu+Grid(mu)) % Grid(mu);
	x[nu] = (x[nu]+lnu+Grid(nu)) % Grid(nu);	
	int pos = indexId(x);
	return pos;
}
















InlineHostDevice void indexEO(const int id, const int parity, int x[4]){
	int za = (id / (Grid(0)/2));
	int zb =  (za / Grid(1));
	x[1] = za - zb * Grid(1);
	x[3] = (zb / Grid(2));
	x[2] = zb - x[3] * Grid(2);
	int xodd = (x[1] + x[2] + x[3] + parity) & 1;
	x[0] = (2 * id + xodd )  - za * Grid(0);
 }



InlineHostDevice int indexEO_neg(const int id, const int parity, const int mu, const int lmu){
	int x[4];
	indexEO(id, parity, x);
	x[mu] = (x[mu]+lmu+Grid(mu)) % Grid(mu);	
	int pos = indexId(x) >> 1;
	int oddbit = GetParity(x);
	pos += oddbit  * HalfVolume();
	return pos;
}
InlineHostDevice int indexEO_neg(const int id, const int parity, const int mu, const int lmu, const int nu, const int lnu){
	int x[4];
	indexEO(id, parity, x);
	x[mu] = (x[mu]+lmu+Grid(mu)) % Grid(mu);
	x[nu] = (x[nu]+lnu+Grid(nu)) % Grid(nu);

	int pos = indexId(x) >> 1;
	int oddbit = GetParity(x);
	pos += oddbit  * HalfVolume();
	return pos;
}


}
#endif
