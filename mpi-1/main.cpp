#include <mpi.h>
#include <stdlib.h>
#include <string>
#include <time.h>

int g_size;
int g_rank;

void GenerateData(int *arr, int size)
{
	for (int i = 0; i < size; i++)
		arr[i] = rand() % (size + 1);
}

int ComputeMax(int *arr, int count)
{
	int max = 0;
	for (int i = 0; i < count; i++) {
		if (arr[i] > max) max = arr[i];
	}
	return max;
}

int ComputeMaxParallel(int *arr, int count)
{
	int *send_counts = new int[g_size];
	int *disps = new int[g_size];
	int stride = count / g_size;
	for (int i = 0; i < g_size - 1; i++) {
		send_counts[i] = stride;
		disps[i] = i * stride;
	}
	send_counts[g_size - 1] = stride + count % g_size;
	disps[g_size - 1] = (g_size - 1) * stride;

	int recvCount = send_counts[g_rank];
	int *recv = new int[recvCount];

	MPI_Scatterv(arr, send_counts, disps, MPI_INT, recv, recvCount, MPI_INT, 0, MPI_COMM_WORLD);

	int max_i = ComputeMax(recv, recvCount);

	int *max_arr = NULL;
	if (g_rank == 0)
		max_arr = new int[g_size];

	MPI_Gather(&max_i, 1, MPI_INT, max_arr, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int max = 0;
	if (g_rank == 0) {
		max = ComputeMax(max_arr, g_size);
		delete[] max_arr;
	}

	delete[] send_counts;
	delete[] disps;
	delete[] recv;
	return max;
}

int main(int argc, char *argv[])
{
	srand((unsigned)time(NULL));

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &g_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
	
	const int dataSize = argc >= 2 ? atoi(argv[1]) : 100;
	int *data = NULL;

	if (g_rank == 0) {
		data = new int[dataSize];
		GenerateData(data, dataSize);
	}

	int max1, max2;
	double t, dt1, dt2;

	t = MPI_Wtime();
	max1 = ComputeMaxParallel(data, dataSize);
	dt1 = (MPI_Wtime() - t) * 1e3;

	if (g_rank == 0)
	{
		t = MPI_Wtime();
		max2 = ComputeMax(data, dataSize);
		dt2 = (MPI_Wtime() - t) * 1e3;

		printf("parallel version: max=%d (%f ms)\n", max1, dt1);
		printf("non-parallel version: max=%d (%f ms)\n", max2, dt2);
		printf("acceleration: %f\n", dt2 / dt1);
		printf(max1 == max2 ? "results are the same" : "results are differ\n");
		delete[] data;
	}

	MPI_Finalize();
	return 0;
}