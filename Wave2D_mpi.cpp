#include <iostream>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include "Timer.h"

using namespace std;

const int default_size = 100;
const double c = 1.0;
const double dt = 0.1;
const double dd = 2.0;

inline double &at(vector<double> &z, int stride, int i, int j)
{
    return z[i * stride + j];
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 5)
    {
        if (rank == 0)
            cerr << "usage: Wave2D_mpi_omp size max_time interval threads" << endl;
        MPI_Finalize();
        return -1;
    }

    int size = atoi(argv[1]);
    int max_time = atoi(argv[2]);
    int interval = atoi(argv[3]);
    int nthreads = atoi(argv[4]);
    omp_set_num_threads(nthreads);

    if (size < 100 || max_time < 3 || interval < 0)
    {
        if (rank == 0)
        {
            cerr << "usage: Wave2D size max_time interval" << endl;
            cerr << "       where size >= 100 && time >= 3 && interval >= 0" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    // --- Divide work among ranks (row-wise)
    int base = size / nprocs;
    int rest = size % nprocs;
    int my_start = (rank < rest) ? rank * (base + 1) : rank * base + rest;
    int my_rows = (rank < rest) ? base + 1 : base;
    int my_end = my_start + my_rows - 1;

    // Print each rank's assigned range (debug/info)
    MPI_Barrier(MPI_COMM_WORLD);
    cerr << "rank[" << rank << "]'s range = " << my_start << " ~ " << my_end << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == nprocs - 1) ? MPI_PROC_NULL : rank + 1;

    int local_rows = my_rows + 2; // plus ghost rows
    int stride = size;

    vector<double> z0(local_rows * stride, 0.0);
    vector<double> z1(local_rows * stride, 0.0);
    vector<double> z2(local_rows * stride, 0.0);

    double coef = (c * dt / dd) * (c * dt / dd);
    int weight = size / default_size;

    // --- Initialize (time=0)
    for (int gi = my_start; gi <= my_end; ++gi)
    {
        int li = gi - my_start + 1;
        for (int j = 0; j < size; ++j)
        {
            if (gi > 40 * weight && gi < 60 * weight && j > 40 * weight && j < 60 * weight)
                at(z0, stride, li, j) = 20.0;
            else
                at(z0, stride, li, j) = 0.0;
        }
    }

    Timer timer;
    MPI_Barrier(MPI_COMM_WORLD);
    timer.start();

    auto exchange = [&](vector<double> &z)
    {
        MPI_Sendrecv(&at(z, stride, 1, 0), stride, MPI_DOUBLE, up, 0,
                     &at(z, stride, local_rows - 1, 0), stride, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&at(z, stride, local_rows - 2, 0), stride, MPI_DOUBLE, down, 1,
                     &at(z, stride, 0, 0), stride, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    };

    // --- Print t=0 (only rank 0)
    if (interval > 0 && 0 % interval == 0 && rank == 0)
    {
        cout << 0 << endl;
        for (int j = 0; j < size; j++)
        {
            for (int i = 0; i < size; i++)
            {
                cout << at(z0, stride, i + 1, j); // shift +1 for ghost row alignment
                if (i < size - 1)
                    cout << " ";
            }
            cout << endl;
        }
    }

    // --- Compute t=1
    exchange(z0);

#pragma omp parallel for
    for (int li = 1; li <= my_rows; ++li)
    {
        int gi = my_start + li - 1;
        for (int j = 0; j < size; ++j)
        {
            if (gi == 0 || gi == size - 1 || j == 0 || j == size - 1)
                at(z1, stride, li, j) = 0.0;
            else
            {
                double neigh = at(z0, stride, li - 1, j) + at(z0, stride, li + 1, j) +
                               at(z0, stride, li, j - 1) + at(z0, stride, li, j + 1);
                at(z1, stride, li, j) = at(z0, stride, li, j) + 0.5 * coef * (neigh - 4.0 * at(z0, stride, li, j));
            }
        }
    }

    // --- Gather & print t=1 (exact same as serial)
    if (interval > 0 && 1 % interval == 0)
    {
        vector<double> local(my_rows * size);
        for (int li = 1; li <= my_rows; ++li)
            for (int j = 0; j < size; ++j)
                local[(li - 1) * size + j] = at(z1, stride, li, j);

        vector<int> counts(nprocs), displs(nprocs);
        int offset = 0;
        for (int r = 0; r < nprocs; ++r)
        {
            counts[r] = ((r < rest) ? base + 1 : base) * size;
            displs[r] = offset;
            offset += counts[r];
        }

        vector<double> global;
        if (rank == 0)
            global.resize(offset);

        MPI_Gatherv(local.data(), my_rows * size, MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr,
                    counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            cout << 1 << endl;
            for (int j = 0; j < size; ++j)
            {
                for (int i = 0; i < size; ++i)
                {
                    cout << global[i * size + j];
                    if (i < size - 1)
                        cout << " ";
                }
                cout << endl;
            }
        }
    }

    // --- Main loop (t >= 2)
    for (int t = 2; t < max_time; ++t)
    {
        exchange(z1);

#pragma omp parallel for
        for (int li = 1; li <= my_rows; ++li)
        {
            int gi = my_start + li - 1;
            for (int j = 0; j < size; ++j)
            {
                if (gi == 0 || gi == size - 1 || j == 0 || j == size - 1)
                    at(z2, stride, li, j) = 0.0;
                else
                {
                    double neigh = at(z1, stride, li - 1, j) + at(z1, stride, li + 1, j) +
                                   at(z1, stride, li, j - 1) + at(z1, stride, li, j + 1);
                    at(z2, stride, li, j) =
                        2.0 * at(z1, stride, li, j) - at(z0, stride, li, j) +
                        coef * (neigh - 4.0 * at(z1, stride, li, j));
                }
            }
        }

        if (interval > 0 && t % interval == 0)
        {
            vector<double> local(my_rows * size);
            for (int li = 1; li <= my_rows; ++li)
                for (int j = 0; j < size; ++j)
                    local[(li - 1) * size + j] = at(z2, stride, li, j);

            vector<int> counts(nprocs), displs(nprocs);
            int offset = 0;
            for (int r = 0; r < nprocs; ++r)
            {
                counts[r] = ((r < rest) ? base + 1 : base) * size;
                displs[r] = offset;
                offset += counts[r];
            }

            vector<double> global;
            if (rank == 0)
                global.resize(offset);

            MPI_Gatherv(local.data(), my_rows * size, MPI_DOUBLE,
                        rank == 0 ? global.data() : nullptr,
                        counts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                cout << t << endl;
                for (int j = 0; j < size; ++j)
                {
                    for (int i = 0; i < size; ++i)
                    {
                        cout << global[i * size + j];
                        if (i < size - 1)
                            cout << " ";
                    }
                    cout << endl;
                }
            }
        }

        z0.swap(z1);
        z1.swap(z2);
    }

    if (rank == 0)
        cerr << "Elapsed time = " << timer.lap() << endl;

    MPI_Finalize();
    return 0;
}
