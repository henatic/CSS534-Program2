#include <iostream>
#include <stdlib.h>
#include "Timer.h"
#include <mpi.h>
#include <omp.h>

using namespace std;

// Global constants
int default_size = 100; // default system size
double c = 1.0;         // wave speed
double dt = 0.1;        // time quantum
double dd = 2.0;        // change in system

// Pre-calculated constant factor for the wave equations
const double k_factor = (c * c * dt * dt) / (dd * dd);

int main(int argc, char *argv[])
{
    // MPI initialization
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Argument verification
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

    // 1. MPI Stripe Partitioning (Using logic from Prog2_hints.docx)
    int stripe_base = size / nprocs; // general stripe size
    int remainder = size % nprocs;   // in case if size is not divisible
    int stripe_size;                 // # of rows for this rank's computation (excluding halo)
    int stripe_begin;                // Global index of the first computed row (i)
    int stripe_end;                  // Global index of the last computed row (i)

    if (rank < remainder)
    {
        stripe_size = stripe_base + 1;
        stripe_begin = rank * stripe_size;
        stripe_end = stripe_begin + stripe_size - 1;
    }
    else
    {
        stripe_size = stripe_base;
        stripe_begin = rank * stripe_base + remainder;
        stripe_end = stripe_begin + stripe_size - 1;
    }

    // Print out ranges (Sequential print to match verification output)
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank != 0)
    {
        MPI_Recv(NULL, 0, MPI_INT, rank - 1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    cerr << "rank[" << rank << "]" << "'s range = " << stripe_begin << " ~ " << stripe_end << endl;
    if (rank < nprocs - 1)
    {
        MPI_Send(NULL, 0, MPI_INT, rank + 1, 99, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 2. Data Allocation (With Ghost Cells)
    // local_rows_with_halo = stripe_size (computed) + 2 (ghost cells)
    int local_rows_with_halo = stripe_size + 2;

    // Allocate 3 x local_rows_with_halo x size
    double ***z = new double **[3];
    for (int p = 0; p < 3; p++)
    {
        z[p] = new double *[local_rows_with_halo];
        for (int i = 0; i < local_rows_with_halo; i++)
        {
            z[p][i] = new double[size];
            // Initialize to 0.0 (including ghost cells)
            for (int j = 0; j < size; j++)
                z[p][i][j] = 0.0;
        }
    }

    // 3. Rank 0 Output Buffers (CRITICAL: Allocate Contiguous Buffer for Gatherv)
    double *contiguous_buffer = nullptr;
    double **full_z_out = nullptr; // Pointer array to view contiguous buffer as 2D

    int *rcounts = nullptr;
    int *displs = nullptr;

    if (rank == 0)
    {
        if (interval > 0)
        {
            // Allocate a single contiguous block for N*N doubles
            contiguous_buffer = new double[size * size];
            full_z_out = new double *[size];
            for (int i = 0; i < size; i++)
            {
                full_z_out[i] = contiguous_buffer + i * size;
            }
        }

        // Setup Gatherv parameters (needed even if interval=0 for verification)
        rcounts = new int[nprocs];
        displs = new int[nprocs];
        int current_disp = 0;
        for (int r = 0; r < nprocs; r++)
        {
            int r_size = (r < remainder) ? stripe_base + 1 : stripe_base;
            rcounts[r] = r_size * size;
            displs[r] = current_disp;
            current_disp += rcounts[r];
        }
    }

    // Tags for MPI communication
    const int TAG_UP = 10;
    const int TAG_DOWN = 20;

    // Timer
    Timer timer;
    MPI_Barrier(MPI_COMM_WORLD);
    timer.start();

    // 4. time = 0: Initialization (z[0])
    int weight = size / default_size;
    for (int gi = stripe_begin; gi <= stripe_end; gi++)
    {
        int li = gi - stripe_begin + 1; // li=1..stripe_size is the local computed range
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

    // Print initial state (t=0) if required
    if (interval > 0 && 0 % interval == 0)
    {
        if (rank == 0)
        {
            cout << 0 << endl;
        }

        // Gatherv z[0] data for printing
        int r_size = stripe_size * size;
        MPI_Gatherv(
            z[0][1], r_size, MPI_DOUBLE,
            (rank == 0) ? contiguous_buffer : NULL,
            rcounts, displs, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    cout << full_z_out[i][j] << endl;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 5. time = 1: Calculate z[1] based on z[0]
    // -------------------------------------------------------------------------

    // Indices for t=1: p=1 (current), q=0 (t-1), r=2 (t-2, not used)
    int t1 = 1;
    int p = 1, q = 0, r = 2;

    int li, j, gi;
    double laplacian;

// OpenMP Parallel Computation for t=1
#pragma omp parallel for collapse(2) private(li, j, gi, laplacian) default(none) \
    shared(size, p, q, z, stripe_begin, stripe_end, stripe_size, k_factor)
    for (li = 1; li <= stripe_size; li++)
    { // li = 1 to stripe_size (local computed rows)
        for (j = 0; j < size; j++)
        {
            gi = stripe_begin + li - 1; // Global row index

            // Boundary Condition
            if (gi == 0 || gi == size - 1 || j == 0 || j == size - 1)
            {
                z[p][li][j] = 0.0;
            }
            else
            {
                // Neighbors are at li+1, li-1, j+1, j-1. All are locally available in z[0]
                laplacian = z[q][li + 1][j] + z[q][li - 1][j] + z[q][li][j + 1] + z[q][li][j - 1] - 4.0 * z[q][li][j];
                // Formula for t=1: Zt = Zt-1 + (k/2) * Laplacian(Zt-1)
                z[p][li][j] = z[q][li][j] + (k_factor / 2.0) * laplacian;
            }
        }
    }

    // Print state at t=1 if required
    if (t1 < max_time && interval > 0 && t1 % interval == 0)
    {
        int r_size = stripe_size * size;
        MPI_Gatherv(
            z[p][1], r_size, MPI_DOUBLE,
            (rank == 0) ? contiguous_buffer : NULL,
            rcounts, displs, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            cout << t1 << endl;
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    cout << full_z_out[i][j] << endl;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 6. Main Loop: simulate wave diffusion from time = 2
    // -------------------------------------------------------------------------

    for (int t = 2; t < max_time; t++)
    {
        // Array Rotation:
        // p: Zt (current, writing to this buffer)
        // q: Zt-1 (previous, used for neighbors and Zt-1 term)
        // r: Zt-2 (oldest, used for Zt-2 term)
        p = t % 3;
        q = (t + 2) % 3; // equivalent to (t-1) % 3
        r = (t + 1) % 3; // equivalent to (t-2) % 3

        // A. HALO EXCHANGE (Ghost Cell Communication on Z_{t-1} data at index 'q')
        if (nprocs > 1)
        {

            // Deadlock Avoidance: Stagger communication by rank parity
            // Phase 1: Even ranks SEND, Odd ranks RECEIVE
            if (rank % 2 == 0)
            { // Even ranks send first
                if (rank > 0)
                { // Send UP
                    MPI_Send(z[q][1], size, MPI_DOUBLE, rank - 1, TAG_UP, MPI_COMM_WORLD);
                }
                if (rank < nprocs - 1)
                { // Send DOWN
                    MPI_Send(z[q][stripe_size], size, MPI_DOUBLE, rank + 1, TAG_DOWN, MPI_COMM_WORLD);
                }
                if (rank > 0)
                { // Recv UP
                    MPI_Recv(z[q][0], size, MPI_DOUBLE, rank - 1, TAG_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                if (rank < nprocs - 1)
                { // Recv DOWN
                    MPI_Recv(z[q][stripe_size + 1], size, MPI_DOUBLE, rank + 1, TAG_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            // Phase 2: Odd ranks SEND, Even ranks RECEIVE
            else
            { // Odd ranks send first
                if (rank < nprocs - 1)
                { // Send DOWN
                    MPI_Send(z[q][stripe_size], size, MPI_DOUBLE, rank + 1, TAG_DOWN, MPI_COMM_WORLD);
                }
                if (rank > 0)
                { // Send UP
                    MPI_Send(z[q][1], size, MPI_DOUBLE, rank - 1, TAG_UP, MPI_COMM_WORLD);
                }
                if (rank < nprocs - 1)
                { // Recv DOWN
                    MPI_Recv(z[q][stripe_size + 1], size, MPI_DOUBLE, rank + 1, TAG_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                if (rank > 0)
                { // Recv UP
                    MPI_Recv(z[q][0], size, MPI_DOUBLE, rank - 1, TAG_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }

// B. HYBRID COMPUTATION (OpenMP)
#pragma omp parallel for collapse(2) private(li, j, gi, laplacian) default(none) \
    shared(size, p, q, r, z, stripe_size, stripe_begin, stripe_end, k_factor)
        for (li = 1; li <= stripe_size; li++)
        { // li = 1 to stripe_size (local computed rows)
            for (j = 0; j < size; j++)
            {
                gi = stripe_begin + li - 1; // Global row index

                // 1. Boundary Condition
                if (gi == 0 || gi == size - 1 || j == 0 || j == size - 1)
                {
                    z[p][li][j] = 0.0;
                }
                else
                {
                    // Neighbors are accessed via li-1 and li+1 (li=0/li=stripe_size+1 will be ghost cells)
                    laplacian = z[q][li + 1][j] + z[q][li - 1][j] + z[q][li][j + 1] + z[q][li][j - 1] - 4.0 * z[q][li][j];
                    // Formula for t>=2: Zt = 2*Zt-1 - Zt-2 + k * Laplacian(Zt-1)
                    z[p][li][j] = 2.0 * z[q][li][j] - z[r][li][j] + k_factor * laplacian;
                }
            }
        }

        // C. GATHERING AND PRINTING
        if (interval > 0 && t % interval == 0)
        {
            // Gatherv z[p] data for printing
            int r_size = stripe_size * size;
            MPI_Gatherv(
                z[p][1], r_size, MPI_DOUBLE,
                (rank == 0) ? contiguous_buffer : NULL,
                rcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                cout << t << endl;
                for (int i = 0; i < size; i++)
                {
                    for (int j = 0; j < size; j++)
                    {
                        cout << full_z_out[i][j] << endl;
                    }
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    } // end of simulation

    // finish the timer
    double elapsed = timer.lap();
    if (rank == 0)
        cerr << "Elapsed time = " << elapsed << endl;

    // 7. Cleanup
    for (int p = 0; p < 3; p++)
    {
        for (int i = 0; i < local_rows_with_halo; i++)
            delete[] z[p][i];
        delete[] z[p];
    }
    delete[] z;

    if (rank == 0)
    {
        if (interval > 0)
        {
            delete[] contiguous_buffer;
            delete[] full_z_out;
        }
        delete[] rcounts;
        delete[] displs;
    }

    MPI_Finalize();
    return 0;
}