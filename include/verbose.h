#ifndef __VERBOSE__
#define __VERBOSE__


#include <enum.h>

namespace ising{


TuneMode getTuning();

void setTuning(TuneMode kerneltunein);


void setVerbosity(Verbosity verbosein);
Verbosity getVerbosity();


}
#endif
