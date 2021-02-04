#include <iostream>
#include <math.h> 
#include <time.h> 
#include <random>
#include <vector> 
#include <fstream>
#include <omp.h>

#include "timer.h"
#include "alloc.h"
#include "complex.h"

#include "parameters.h"
#include "random.h"
#include "update.h"


#include "array.h"
#include "actime.h"
#include "gnuplot.h"




using namespace std;
using namespace ising;





double mean(const vector<double> &array){
    double s = 0.0;
    for (int i = 0; i < array.size(); ++i)
        s = s + array[i];
    return s/double(array.size());
}

double2 jackknife(const vector<double> &data){
	int length = data.size();
	double mean = 0.0;
	for(int i = 0; i < length; ++i) mean += data[i];
	vector<double> trials;
	for(int i = 0; i < length; ++i) trials.push_back((mean - data[i])/double(length-1));
	mean /= double(length);
	
	double jmean = 0.0;
	for(int i = 0; i < length; ++i) jmean += trials[i];
	jmean /= double(length);
	double err0 = 0.0;
	for(int i = 0; i < length; ++i){
		double tmp = trials[i] -jmean;
		err0 += tmp * tmp;
	}
	double norm = double(length-1) / double(length);
	return make_double2(mean, sqrt(err0 * norm));
}




double standardDeviation(const vector<double> &in, int wopt){
	vector<double> array;
    for (int i = 0; i < in.size(); i+=wopt) array.push_back(in[i]);
    double mu = mean(array);
    double s = 0;
    int length = array.size();
    for (int i = 0; i < length; ++i)
        s = s + (array[i] - mu)*(array[i] - mu);
    s = s/float(length);
    return sqrt(s);
}




double2 susceptibility (const vector<double> &data){
    double mu = mean(data);
    double s = 0;
    int length = data.size();
    vector<double> jack;
    for (int i = 0; i < length; ++i){
    	double tmp = (data[i] - mu)*(data[i] - mu);
        s += tmp;
        jack.push_back(tmp);
    }
    return jackknife(jack);
}








int main(){
	Timer a0;
	a0.start();
	
	
	
	int numthreads =  omp_get_max_threads();
	cout << "Number of threads: " << numthreads << endl;
	
	omp_set_num_threads(numthreads);
	//create one RNG per thread
	for(int i = 0; i < numthreads; ++i){
		generator.push_back(std::mt19937(i));
	}
	
	//omp_set_num_threads(1);
	int gpuID = 1;
	//Start(gpuID, DEBUG_VERBOSE, TUNE_YES); 
	Start(gpuID, VERBOSE, TUNE_YES); // Important. Setup GPU id and setup tune kernels and verbosity level. See cuda_error_check.cpp/.h
	
	
	

	
	
	int dirs = 2; 
	int Npt = 32; //The number of points in each direction must be an even number!!!!!!!!!
	double beta = 1.;
	double aniso = 1.;
	int imetrop = 1;
	
	
	double jconst = 1.0;
	
	SetupLatticeParameters(Npt, dirs, beta, jconst, imetrop);
	
	std::string grid = "";
	for(int i = 0; i < Dirs(); i++) grid += ToString(PARAMS::Grid[i]) + '_';
	grid += ToString(jconst);
      
	string filename = grid + ".dat";
	
    ofstream fileout1;
    string filename11 = "Mean_cpu1_" + filename;
    fileout1.open(filename11, ios::out);// | ios::app );
    if (!fileout1.is_open()) {
    	cout << "Cannot create file: " << filename11 << endl;
    	exit(1);
	}
	cout << "Creating file: " << filename11 << endl;
    fileout1.precision(14);	
	
	
	grid = "";
	for(int i = 0; i < Dirs(); i++) grid += ToString(PARAMS::Grid[i]) + '\t';
	

	GnuplotPipe gp;
	
	double step = 0.5;
	for(double temperature= 1.0; 	temperature <= 4; temperature+=step){
	//	if(temperature> 2 && temperature < 3) step = 0.01;
		//else step = 0.1;
		
		beta = 1.0/temperature;
		//beta = 1.0/2.4;
		
		//Setup global parameters in Host and Device
		SetupLatticeParameters(Npt, dirs, beta, jconst, imetrop);
		
		int maxIter = 100000;
		int printiter = 10000;
		bool hotstart = false;
		
		//GPU Code
		Timer t0;
		t0.start();
		
		
		ofstream fileout;
		string filename1 = "Data_cpu_" + filename;
		fileout.open(filename1, ios::out);
		if (!fileout.is_open()) {
			cout << "Cannot create file: " << filename1 << endl;
			exit(1);
		}
		cout << "Creating file: " << filename1 << endl;
		fileout.precision(14);
	   
		//Only used if: export ISING_ENABLE_MANAGED_MEMORY=1
		use_managed_memory();

		//Array array to store the phases
		//Array<int> *lattice = new Array<int>(Device, Volume()*Dirs()); //also initialize aray to 0
		Array<int> *lattice = new Array<int>(Host, Volume()*Dirs()); //also initialize aray to 0
		//Initialize cuda rng
		int seed = 1234;
		CudaRNG *rng = new CudaRNG(seed, HalfVolume());
		
		InitLattice(lattice, rng, 1);
		double avgspin = 0;
		double energy = 0;
		ComputeAVG(lattice, &avgspin, &energy);
		
		//cout << PARAMS::iter << '\t' << avgspin << '\t' << energy << endl;
		fileout << PARAMS::iter << '\t' << avgspin << '\t' << energy << endl;

		int mininter = 10000;	
		int num = 0;
		double avg = 0.0;
		vector<double> mag;
		vector<double> energ;
		
		

		
		
		int wopt = 70;
		for(PARAMS::iter = 1; PARAMS::iter <= maxIter; ++PARAMS::iter){
			// metropolis algorithm 
			UpdateLattice(lattice, rng,  PARAMS::metrop);
			ComputeAVG(lattice, &avgspin, &energy);	
			if(PARAMS::iter >= mininter){
				mag.push_back(abs(avgspin));
				energ.push_back(energy);
			}
			if((PARAMS::iter%printiter)==0){
				cout << "iter: " << PARAMS::iter << '\t' << avgspin << '\t' << energy << endl;
			}
			
			if(0)if(PARAMS::iter >= mininter){
				calculateCorTime(50, PARAMS::iter, mag, wopt, false);
				//calculateCorTime1(50, PARAMS::iter, corr, nsweep);
				if((PARAMS::iter%printiter)==0)cout << "iter: " << PARAMS::iter << '\t' << avgspin << '\t' << energy << '\t' << wopt << endl;
				fileout << PARAMS::iter << '\t' << avgspin << '\t' << energy << endl;
				avg += abs(avgspin);
				num++;
			}
			
			if(0)if(PARAMS::iter > mininter && (PARAMS::iter%printiter)==0){		
				gp.sendLine("reset;");
				gp.sendLine("set terminal wxt size 900,400 enhanced font 'Verdana,10' persist");
				gp.sendLine("unset label");
				gp.sendLine("set grid");
				gp.sendLine("set mxtics 5");
				gp.sendLine("set mytics 5");
				gp.sendLine("set style line 1 linecolor rgb 'red' pointtype 5 pointsize 1");
				gp.sendLine("set style line 2 linecolor rgb '#0010ad' pointtype 5 pointsize 1");
				gp.sendLine("set xlabel \"|m|\"");
				gp.sendLine("set ylabel \"iter\"");
				gp.sendLine("set key left bottom");
				gp.sendLine("plot \""+filename1+"\" using 1:(abs($2)) ls 2  title \"Magnetization\"");
				gp.sendEndOfData();
			}
			
			
				
			if(0)if((PARAMS::iter%printiter)==0){
				ComputeAVG(lattice, &avgspin, &energy);
				cout << PARAMS::iter << '\t' << avgspin << '\t' << energy << endl;
				fileout << PARAMS::iter << '\t' << avgspin << '\t' << energy << endl;
				if(PARAMS::iter > 10000){
					avg += abs(avgspin);
					num++;
				}
			}
				
		}
		
		calculateCorTime(50, PARAMS::iter, mag, wopt, true);
		
		if(wopt < 1){
			cout << "wopt not valid...." << endl;
			exit(1);
		}
		
		vector<double> data;
		for(int i = 0; i < mag.size(); i+= wopt) data.push_back(mag[i]);
		int nconfigs = data.size();
		//cout << "#wopt: " << wopt << endl;
		//cout << "#configs: " << nconfigs << endl;
		double2 sus = susceptibility(data);
		double2 Mmean = jackknife(data);
		
		vector<double> edata;
		for(int i = 0; i < energ.size(); i+= wopt) edata.push_back(energ[i]);
		int neconfigs = edata.size();
		double2 specificHeat = susceptibility(edata);
		double2 eMmean = jackknife(edata);
		
		
		
		cout << "###################################################################################" << endl;
		cout << temperature << "\t#wopt: " << wopt << "\t#configs: " << nconfigs << "\tMean: " << Mmean.x << " ± " << Mmean.y << "  ::suceptibility: " << sus.x << " ± " << sus.y << "\teMean: " << eMmean.x << " ± " << eMmean.y << "  ::specificHeat: " << specificHeat.x << " ± " << specificHeat.y << endl;
		
		
		
		fileout1 << grid << Jconst() << '\t' << temperature << '\t' << wopt << '\t' << nconfigs << '\t' << Mmean.x << '\t' << Mmean.y << '\t' << sus.x << '\t' << sus.y << '\t' << sus.x*Volume() << '\t' << sus.y*Volume() << '\t' << eMmean.x << '\t' << eMmean.y << '\t' << specificHeat.x*Volume() << '\t' << specificHeat.y*Volume() << endl;
		
		fileout.close();
		
		delete lattice;
		delete rng;
		t0.stop();
		std::cout << "Time: " << t0.getElapsedTime() << " s" << endl;

				
		gp.sendLine("reset;");
		gp.sendLine("set terminal x11 size 1200,600 enhanced font 'Verdana,12' persist");
		gp.sendLine("set multiplot layout 2,4 rowsfirst");
		gp.sendLine("unset label");
		gp.sendLine("set grid");
		gp.sendLine("set mxtics 5");
		gp.sendLine("set mytics 5");
		gp.sendLine("set style line 1 linecolor rgb 'red' pointtype 5 pointsize 1");
		gp.sendLine("set style line 2 linecolor rgb '#0010ad' pointtype 5 pointsize 1");
		gp.sendLine("set xlabel \"T\"");
	//	gp.sendLine("set ylabel \"\"");
		gp.sendLine("set key left bottom");
		
		int dirs = 2+Dirs();
		
		gp.sendLine("plot \""+filename11+"\" using " + ToString(dirs) + ":" + ToString(1+dirs) + " ls 2  title \"wopt\"");
		gp.sendLine("plot \""+filename11+"\" using " + ToString(dirs) + ":" + ToString(2+dirs) + " ls 2  title \"#configs\"");
		gp.sendLine("plot \""+filename11+"\" using " + ToString(dirs) + ":" + ToString(3+dirs) + ":" + ToString(4+dirs) + " ls 2  title \"|Magnetization|\" with yerrorbars");
		gp.sendLine("plot \""+filename11+"\" using " + ToString(dirs) + ":" + ToString(5+dirs) + ":" + ToString(6+dirs) + " ls 2  title \"Susceptibility\" with yerrorbars");
		gp.sendLine("plot \""+filename11+"\" using " + ToString(dirs) + ":" + ToString(7+dirs) + ":" + ToString(8+dirs) + " ls 2  title \"Susceptibility*Volume\" with yerrorbars");
		gp.sendLine("plot \""+filename11+"\" using " + ToString(dirs) + ":" + ToString(9+dirs) + ":" + ToString(10+dirs) + " ls 2  title \"energy\" with yerrorbars");
		gp.sendLine("plot \""+filename11+"\" using " + ToString(dirs) + ":" + ToString(11+dirs) + ":" + ToString(12+dirs) + " ls 2  title \"specificHeat\" with yerrorbars");
		gp.sendLine("unset multiplot");
		gp.sendEndOfData();

	}	




	fileout1.close();
	
	Finalize(0); // Important to save tunned kernels to file
	return 0;

	
}

