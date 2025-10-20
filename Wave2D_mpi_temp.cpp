// Wave2D_mpi.cpp
// Updated: MPI + OpenMP implementation
// Based on the user's original Wave2D_mpi.cpp. (original file provided by user)
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <limits>
#include "Timer.h"
#include <mpi.h>
#include <omp.h>

using namespace std;

int default_size = 100; // default system size
int defaultCellWidth = 8;
double c = 1.0;  // wave speed
double dt = 0.1; // time quantum
double dd = 2.0; // spatial step (given)

inline double &at(double *buf, int stride, int li, int j)
{
    // li uses 0..local_rows+1 indexing where 0 and local_rows+1 are ghost rows
    return buf[li * stride + j];
}

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
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "rank[" << rank << "]'s range = " << my_start << " ~ " << my_end << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    int local_rows = my_end - my_start + 1;
    // We'll allocate arrays with 2 ghost rows: index 0 = top ghost, 1..local_rows = real rows, local_rows+1 = bottom ghost
    int alloc_rows = local_rows + 2;
    int stride = size; // number of columns

    // allocate 3 time-step buffers as contiguous blocks
    double **z = new double *[3];
    for (int p = 0; p < 3; p++)
    {
        z[p] = new double[alloc_rows * stride];
        // initialize to zero
        for (int idx = 0; idx < alloc_rows * stride; ++idx)
            z[p][idx] = 0.0;
    }

    // initialize time = 0 in local region (store into li = 1..local_rows)
    int weight = size / default_size;
    for (int gi = my_start; gi <= my_end; gi++)
    {
        int li = gi - my_start + 1; // offset 1 for ghost row
        for (int j = 0; j < size; j++)
        {
            if (gi > 40 * weight && gi < 60 * weight && j > 40 * weight && j < 60 * weight)
            {
                at(z[0], stride, li, j) = 20.0;
            }
            else
            {
                at(z[0], stride, li, j) = 0.0;
            }
        }
    }

    // Ensure ghost rows are zero (Dirichlet boundary)
    for (int j = 0; j < size; ++j)
    {
        at(z[0], stride, 0, j) = 0.0;
        at(z[0], stride, alloc_rows - 1, j) = 0.0;
    }

    // Pre-compute constant r = (c*dt/dd)^2
    double r = (c * dt / dd) * (c * dt / dd);

    // gather local_rows sizes on root (needed for Gatherv)
    vector<int> rows_per_rank(nprocs);
    int my_rows = local_rows;
    MPI_Gather(&my_rows, 1, MPI_INT, rows_per_rank.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // root prepares recvcounts and displacements (for doubles)
    vector<int> recvcounts;
    vector<int> displs;
    if (rank == 0 && interval > 0)
    {
        recvcounts.resize(nprocs);
        displs.resize(nprocs);
        int offset = 0;
        for (int rnk = 0; rnk < nprocs; ++rnk)
        {
            recvcounts[rnk] = rows_per_rank[rnk] * size; // number of doubles to receive from rank rnk
            displs[rnk] = offset;
            offset += recvcounts[rnk];
        }
    }

    // Timer
    Timer timer;
    MPI_Barrier(MPI_COMM_WORLD); // Start timer after all ranks are set up
    timer.start();

    // --- Initialize time = 1 (z[1]) using half-step: z1 = z0 + 0.5 * r * Laplacian(z0)
    // We need neighbors; for internal j, j=0 and j=size-1 are boundary (remain zero)
    // Exchange ghosts for z[0] prior to computing z[1]
    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == nprocs - 1) ? MPI_PROC_NULL : rank + 1;

    // Send our first real row (li=1) upward, receive into top ghost (li=0)
    MPI_Sendrecv(&at(z[0], stride, 1, 0), stride, MPI_DOUBLE, up, 100,
                 &at(z[0], stride, alloc_rows - 1, 0), stride, MPI_DOUBLE, down, 100,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Send our last real row (li=local_rows) downward, receive into bottom ghost (li=alloc_rows-1)
    // Note: the above call already did an exchange to bottom ghost; but to be safe do symmetric:
    MPI_Sendrecv(&at(z[0], stride, local_rows, 0), stride, MPI_DOUBLE, down, 101,
                 &at(z[0], stride, 0, 0), stride, MPI_DOUBLE, up, 101,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// compute z[1]
#pragma omp parallel for collapse(1)
    for (int li = 1; li <= local_rows; ++li)
    {
        // global row index:
        int gi = my_start + (li - 1);
        for (int j = 0; j < size; ++j)
        {
            // boundary columns remain zero
            if (j == 0 || j == size - 1 || gi == 0 || gi == size - 1)
            {
                at(z[1], stride, li, j) = 0.0;
            }
            else
            {
                // Laplacian using neighbors
                double center = at(z[0], stride, li, j);
                double upv = at(z[0], stride, li - 1, j);
                double downv = at(z[0], stride, li + 1, j);
                double leftv = at(z[0], stride, li, j - 1);
                double rightv = at(z[0], stride, li, j + 1);
                double lap = upv + downv + leftv + rightv - 4.0 * center;
                at(z[1], stride, li, j) = center + 0.5 * r * lap;
            }
        }
    }

    // main time loop t = 1 .. max_time-1 produce z[t+1] from z[t] and z[t-1]
    for (int t = 1; t < max_time; ++t)
    {
        int cur = t % 3;        // z[t]
        int prev = (t - 1) % 3; // z[t-1]
        int next = (t + 1) % 3; // z[t+1] will be written

        // Exchange boundary rows of z[cur] with neighbors into ghost rows
        // Send our first real row (li=1) to up, receive bottom ghost from down into li=alloc_rows-1
        MPI_Sendrecv(&at(z[cur], stride, 1, 0), stride, MPI_DOUBLE, up, 200 + t,
                     &at(z[cur], stride, alloc_rows - 1, 0), stride, MPI_DOUBLE, down, 200 + t,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Send our last real row (li=local_rows) to down, receive top ghost from up into li=0
        MPI_Sendrecv(&at(z[cur], stride, local_rows, 0), stride, MPI_DOUBLE, down, 300 + t,
                     &at(z[cur], stride, 0, 0), stride, MPI_DOUBLE, up, 300 + t,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// compute interior for next time step.
// note: boundaries (global edges) remain 0; we skip j=0 and j=size-1
#pragma omp parallel for
        for (int li = 1; li <= local_rows; ++li)
        {
            int gi = my_start + (li - 1);
            for (int j = 0; j < size; ++j)
            {
                if (j == 0 || j == size - 1 || gi == 0 || gi == size - 1)
                {
                    at(z[next], stride, li, j) = 0.0;
                }
                else
                {
                    double center = at(z[cur], stride, li, j);
                    double upv = at(z[cur], stride, li - 1, j);
                    double downv = at(z[cur], stride, li + 1, j);
                    double leftv = at(z[cur], stride, li, j - 1);
                    double rightv = at(z[cur], stride, li, j + 1);
                    double lap = upv + downv + leftv + rightv - 4.0 * center;
                    // Wave update: z_{n+1} = 2*z_n - z_{n-1} + r * laplacian(z_n)
                    at(z[next], stride, li, j) = 2.0 * center - at(z[prev], stride, li, j) + r * lap;
                }
            }
        }

        // Optionally gather full grid to rank 0 every 'interval' steps (and when interval>0)
        if (interval > 0 && (t % interval == 0))
        {
            // pack local real rows (li=1..local_rows) into contiguous send buffer
            vector<double> sendbuf(local_rows * size);
            for (int li = 1; li <= local_rows; ++li)
            {
                for (int j = 0; j < size; ++j)
                {
                    sendbuf[(li - 1) * size + j] = at(z[next], stride, li, j);
                }
            }

            // gather into root
            vector<double> recvbuf;
            if (rank == 0)
            {
                int total = 0;
                for (int rnk = 0; rnk < nprocs; ++rnk)
                    total += rows_per_rank[rnk] * size;
                recvbuf.resize(total);
            }

            MPI_Gatherv(sendbuf.data(), local_rows * size, MPI_DOUBLE,
                        (rank == 0) ? recvbuf.data() : nullptr,
                        (rank == 0) ? recvcounts.data() : nullptr,
                        (rank == 0) ? displs.data() : nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                // simple summary print (min/max)
                double gmin = numeric_limits<double>::infinity();
                double gmax = -numeric_limits<double>::infinity();
                int totalVals = 0;
                // iterate recvbuf
                for (size_t idx = 0; idx < recvbuf.size(); ++idx)
                {
                    double v = recvbuf[idx];
                    if (v < gmin)
                        gmin = v;
                    if (v > gmax)
                        gmax = v;
                    totalVals++;
                }
                cout << "t=" << t << " gathered grid summary: min=" << gmin << " max=" << gmax << " vals=" << totalVals << endl;
            }
        }

        // rotate buffers implicitly by indices (we used modulo indices), no need to swap pointers
    }

    double elapsed = timer.lap();
    if (rank == 0)
        cerr << "Elapsed time = " << elapsed << endl;

    // free buffers
    for (int p = 0; p < 3; p++)
    {
        delete[] z[p];
    }
    delete[] z;

    MPI_Finalize();
    return 0;
}
