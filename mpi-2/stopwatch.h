#ifndef _STOPWATCH_H_
#define _STOPWATCH_H_

#include <mpi.h>

class MpiStopwatch
{
public:
	MpiStopwatch() { t = 0; }
	void Start() {
		t = MPI_Wtime();
	}
	double Stop() {
		return (MPI_Wtime() - t) * 1e3;
	}
private:
	double t;
};

#endif // _STOPWATCH_H_