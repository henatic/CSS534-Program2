#include <iostream>
#include "Timer.h"
#include <stdlib.h> // atoi

int default_size = 100; // the default system size
int defaultCellWidth = 8;
double c = 1.0;  // wave speed
double dt = 0.1; // time quantum
double dd = 2.0; // change in system

using namespace std;

int main(int argc, char *argv[])
{
    // verify arguments
    if (argc != 4)
    {
        cerr << "usage: Wave2D size max_time interval" << endl;
        return -1;
    }
    int size = atoi(argv[1]);
    int max_time = atoi(argv[2]);
    int interval = atoi(argv[3]);

    if (size < 100 || max_time < 3 || interval < 0)
    {
        cerr << "usage: Wave2D size max_time interval" << endl;
        cerr << "       where size >= 100 && time >= 3 && interval >= 0" << endl;
        return -1;
    }

    // create a simulation space
    double z[3][size][size];
    for (int p = 0; p < 3; p++)
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                z[p][i][j] = 0.0; // no wave

    // start a timer
    Timer time;
    time.start();

    // time = 0;
    // initialize the simulation space: calculate z[0][][]
    int weight = size / default_size;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i > 40 * weight && i < 60 * weight &&
                j > 40 * weight && j < 60 * weight)
            {
                z[0][i][j] = 20.0;
            }
            else
            {
                z[0][i][j] = 0.0;
            }
        }
    }

    // time = 1
    // calculate z[1][][]
    // cells not on edge
    // print time = 0 if requested
    if (interval > 0 && 0 % interval == 0)
    {
        cout << 0 << endl;
        int p0 = 0;
        for (int j = 0; j < size; j++)
        {
            for (int i = 0; i < size; i++)
            {
                cout << z[p0][i][j];
                if (i < size - 1)
                    cout << " ";
            }
            cout << endl;
        }
    }

    double c2 = c * c;
    double coef = (c2) * (dt / dd) * (dt / dd);

    int p = 1;
    int p_prev = 0;  // t-1
    int p_prev2 = 2; // unused for t==1 but set for completeness

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i == 0 || i == size - 1 || j == 0 || j == size - 1)
            {
                z[p][i][j] = 0.0;
            }
            else
            {
                double neigh = z[p_prev][i + 1][j] + z[p_prev][i - 1][j] + z[p_prev][i][j + 1] + z[p_prev][i][j - 1];
                z[p][i][j] = z[p_prev][i][j] + 0.5 * coef * (neigh - 4.0 * z[p_prev][i][j]);
            }
        }
    }

    // print time = 1 if requested
    if (interval > 0 && 1 % interval == 0)
    {
        cout << 1 << endl;
        int pp = 1;
        for (int j = 0; j < size; j++)
        {
            for (int i = 0; i < size; i++)
            {
                cout << z[pp][i][j];
                if (i < size - 1)
                    cout << " ";
            }
            cout << endl;
        }
    }

    // simulate wave diffusion from time = 2
    for (int t = 2; t < max_time; t++)
    {
        // compute indices for rotating buffers
        int cur = t % 3;         // will hold Z^t
        int prev = (t - 1) % 3;  // Z^{t-1}
        int prev2 = (t - 2) % 3; // Z^{t-2}

        // compute interior points
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                if (i == 0 || i == size - 1 || j == 0 || j == size - 1)
                {
                    z[cur][i][j] = 0.0;
                }
                else
                {
                    double neigh = z[prev][i + 1][j] + z[prev][i - 1][j] + z[prev][i][j + 1] + z[prev][i][j - 1];
                    z[cur][i][j] = 2.0 * z[prev][i][j] - z[prev2][i][j] + coef * (neigh - 4.0 * z[prev][i][j]);
                }
            }
        }

        // print current time step if requested
        if (interval > 0 && t % interval == 0)
        {
            cout << t << endl;
            for (int jj = 0; jj < size; jj++)
            {
                for (int ii = 0; ii < size; ii++)
                {
                    cout << z[cur][ii][jj];
                    if (ii < size - 1)
                        cout << " ";
                }
                cout << endl;
            }
        }
    } // end of simulation

    // finish the timer
    cerr << "Elapsed time = " << time.lap() << endl;
    return 0;
}
