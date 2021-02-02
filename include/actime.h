
#ifndef __ACTIME_H__
#define __ACTIME_H__


#include <vector>


namespace ising{


void calculateCorTime(int miniter, int iter, std::vector<double> &gamma, int &nsweep, bool print);
void calculateCorTime1(int miniter, int iter, std::vector<double> &gamma, int &nsweep);



}

#endif

