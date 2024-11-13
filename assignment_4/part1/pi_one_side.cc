#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int fnz(long long *temp_count, int size)
{
    int res = 0;

    for (int i = 0; i < size; i++)
    {
        if (temp_count[i] != 0)
            res += 1;
    }

    return (res == size);
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // MPI_Status status;
    long long num_iter = tosses / world_size;
    unsigned int seed = time(NULL) + world_rank;
    long long count = 0;

    for(int i = 0; i < num_iter; i++){
        double x = (double)rand_r(&seed) / RAND_MAX;  // Random x between -1 and 1
        double y = (double)rand_r(&seed) / RAND_MAX;  // Random y between -1 and 1
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) count++;
    }

    if (world_rank == 0)
    {
        // Master
        long long *tmp = (long long *)calloc(world_size, sizeof(long long));
        MPI_Win_create(tmp, world_size * sizeof(long long), sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        int ready = 0;
        while (!ready)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            ready = fnz(tmp, world_size - 1);
            MPI_Win_unlock(0, win);
        }

        for (int i = 1; i < world_size; ++i)
        {
            count += tmp[i - 1];
        }
    }
    else
    {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&count, 1, MPI_LONG_LONG, 0, world_rank - 1, 1, MPI_LONG_LONG, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 * (double)count / tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
