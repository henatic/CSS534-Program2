#include <iostream>
#include <stdlib.h>
#include "Timer.h"
#include <mpi.h>
#include <omp.h>

using namespace std;

int default_size = 100; // default system size
int defaultCellWidth = 8;
double c = 1.0;  // wave speed
double dt = 0.1; // time quantum
double dd = 2.0; // change in system

int main(int argc, char *argv[])
{
    // MPI initialization
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 5)
    {
        if (rank == 0)
            cerr << "usage: Wave2D_mpi size max_time interval #threads" << endl;
        MPI_Finalize();
        return -1;
    }

    int size = atoi(argv[1]);
    int max_time = atoi(argv[2]);
    int interval = atoi(argv[3]);
    int nthreads = atoi(argv[4]);

    if (size < 100 || max_time < 3 || interval < 0 || nthreads < 1)
    {
        if (rank == 0)
        {
            cerr << "usage: Wave2D_mpi size max_time interval #threads" << endl;
            cerr << "       where size >= 100 && time >= 3 && interval >= 0 && #threads >= 1" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    // Set number of OpenMP threads for this process
    omp_set_num_threads(nthreads);

    // compute partitioning: divide rows (i axis) among ranks
    int base = size / nprocs;
    int rest = size % nprocs;
    int my_start, my_end;
    if (rank < rest)
    {
        my_start = rank * (base + 1);
        my_end = my_start + (base + 1) - 1;
    }
    else
    {
        my_start = rank * base + rest;
        my_end = my_start + base - 1;
    }

    // Print out ranges (each rank prints its range)
    // To keep order more deterministic, have rank 0 print first then others
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "rank[" << rank << "]" << "'s range = " << my_start << " ~ " << my_end << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // allocate local buffers: for simplicity we allocate full width in j but only rows my_start..my_end
    int local_rows = my_end - my_start + 1;
    // allocate 3 x local_rows x size
    double ***z = new double **[3];
    for (int p = 0; p < 3; p++)
    {
        z[p] = new double *[local_rows];
        for (int i = 0; i < local_rows; i++)
        {
            z[p][i] = new double[size];
            for (int j = 0; j < size; j++)
                z[p][i][j] = 0.0;
        }
    }

    // initialize time = 0 in local region
    int weight = size / default_size;
    for (int gi = my_start; gi <= my_end; gi++)
    {
        int li = gi - my_start;
        for (int j = 0; j < size; j++)
        {
            if (gi > 40 * weight && gi < 60 * weight && j > 40 * weight && j < 60 * weight)
            {
                z[0][li][j] = 20.0;
            }
            else
            {
                z[0][li][j] = 0.0;
            }
        }
    }

    // Timer
    Timer timer;
    timer.start();

    // TODO: implement time=1 initialization (compute z[1] for local rows)
    // TODO: in main loop t=2..max_time-1 do:
    //   - exchange boundary rows with neighbors using MPI_Sendrecv
    //   - compute interior rows using OpenMP for parallel loop over rows and cols
    //   - rank 0 gathers strips from other ranks when printing is required
    //   - rotate buffers (use modulo index or swap pointers)

    // For now we won't implement full functionality; we'll just run a placeholder loop to measure basic timing
    for (int t = 1; t < max_time; t++)
    {
        // placeholder: no-op compute
    }

    double elapsed = timer.lap();
    if (rank == 0)
        cerr << "Elapsed time = " << elapsed << endl;

    // free buffers
    for (int p = 0; p < 3; p++)
    {
        for (int i = 0; i < local_rows; i++)
            delete[] z[p][i];
        delete[] z[p];
    }
    delete[] z;

    MPI_Finalize();
    return 0;
}
