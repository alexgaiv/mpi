#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "stopwatch.h"

using namespace std;

int g_size;
int g_rank;

double func(double x)
{
	return x*x*x;
}

void GenerateData(double xmin, double xmax, double(*f)(double), int numPoints, double *&x_vals, double *&f_vals)
{
	x_vals = new double[numPoints];
	f_vals = new double[numPoints];

	double step = (xmax - xmin) / (numPoints - 1);

	double x = xmin;
	for (int i = 0; i < numPoints; i++)
	{
		x_vals[i] = x;
		f_vals[i] = f(x);
		x += step;
	}
}

void FreeData(double *&x_vals, double *&f_vals)
{
	delete[] x_vals;
	delete[] f_vals;
}

double LagrangeCoeff(double x, int j, int numPoints, double *x_vals)
{
	double res = 1.0;
	double xj = x_vals[j];

	for (int i = 0; i < j; i++)
	{
		double xi = x_vals[i];
		res *= (x - xi) / (xj - xi);
	}

	for (int i = j + 1; i < numPoints; i++)
	{
		double xi = x_vals[i];
		res *= (x - xi) / (xj - xi);
	}

	return res;
}

double LagrangeInterpolate(double x, int numPoints, double *x_vals, double *f_vals)
{
	double L = 0.0;
	for (int i = 0; i < numPoints; i++)
	{
		L += f_vals[i] * LagrangeCoeff(x, i, numPoints, x_vals);
	}
	return L;
}

double LagrangeInterpolateParallel(double x, int numPoints, double *x_vals, double *f_vals)
{
	int *send_counts = new int[g_size];
	int *disps = new int[g_size];
	int stride = numPoints / g_size;
	for (int i = 0; i < g_size - 1; i++) {
		send_counts[i] = stride;
		disps[i] = i * stride;
	}
	send_counts[g_size - 1] = stride + numPoints % g_size;
	disps[g_size - 1] = (g_size - 1) * stride;

	int recv_count = send_counts[g_rank];
	double *f_recv = new double[recv_count];
	double *x_recv = x_vals;

	if (g_rank != 0)
		x_recv = new double[numPoints];

	MPI_Scatterv(f_vals, send_counts, disps, MPI_DOUBLE, f_recv, recv_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(x_recv, numPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double L_local = 0.0;
	double L = 0.0;

	for (int k = 0; k < recv_count; k++)
	{
		L_local += f_recv[k] * LagrangeCoeff(x, k + disps[g_rank], numPoints, x_recv);
	}

	MPI_Reduce(&L_local, &L, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	delete[] send_counts;
	delete[] disps;
	delete[] f_recv;
	if (g_rank != 0)
		delete[] x_recv;
	return L;
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &g_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

	MpiStopwatch timer;

	int numPoints = argc > 1 ? atoi(argv[1]) : 200;
	double xmin = argc > 2 ? atof(argv[2]) : -100.0;
	double xmax = argc > 3 ? atof(argv[3]) : 100.0;

	double *x_vals = NULL;
	double *f_vals = NULL;

	if (g_rank == 0)
		GenerateData(xmin, xmax, func, numPoints, x_vals, f_vals);

	int numTestPoints = 100;
	double step = (xmax - xmin) / (numTestPoints - 1);
	double avg_err1 = 0.0, avg_err2 = 0.0;
	double dt1 = 0.0, dt2 = 0.0;
	double x;

	timer.Start();
	x = xmin;
	for (int i = 0; i < numTestPoints; i++)
	{
		double L = LagrangeInterpolateParallel(x, numPoints, x_vals, f_vals);
		avg_err1 += abs(L - func(x));
		x += step;
	}
	avg_err1 /= numTestPoints;
	dt1 = timer.Stop();

	if (g_rank == 0)
	{
		timer.Start();
		x = xmin;
		for (int i = 0; i < numTestPoints; i++)
		{
			double L = LagrangeInterpolate(x, numPoints, x_vals, f_vals);
			//printf("%f %f\n", x, L);
			avg_err2 += abs(L - func(x));
			x += step;
		}
		avg_err2 /= numTestPoints;
		dt2 = timer.Stop();

		
		printf("Number of points: %d\n", numPoints);
		printf("parallel version: avg_err=%f, t=%f ms\n", avg_err1, dt1);
		printf("non-parallel version: avg_err=%f, t=%f ms\n", avg_err2, dt2);
		printf("acceleration: %f\n", dt2 / dt1);
	}

	if (g_rank == 0)
		FreeData(x_vals, f_vals);
	MPI_Finalize();
	if (g_size == 1) getchar();
}