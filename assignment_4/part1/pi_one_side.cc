#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int fnz (long long int *schedule, long long int *oldschedule, int size)
{
    int res = 0;

    for (int i = 1; i < size; i++){
        if (schedule[i] == 0){
            res++;
        }
        else if (schedule[i] != oldschedule[i]){
            oldschedule[i] = schedule[i];
        }
    }

    return (res == 0);
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
    long long count = 0, tmp = 0;
    
    for(int i = 0; i < num_iter; i++){
        double x = (double)rand_r(&seed) / RAND_MAX;  // Random x between -1 and 1
        double y = (double)rand_r(&seed) / RAND_MAX;  // Random y between -1 and 1
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) tmp++;
    }

    if (world_rank == 0)
    {
        // Master
        long long int  *oldschedule = (long long *) malloc(world_size * sizeof(long long));
        // Use MPI to allocate memory for the target window
        long long int  *schedule;
        MPI_Alloc_mem(world_size * sizeof(long long), MPI_INFO_NULL, &schedule);

        for (int i = 0; i < world_size; i++)
        {
           schedule[i] = 0;
           oldschedule[i] = -1;
        }

        // Create a window. Set the displacement unit to sizeof(int) to simplify
        // the addressing at the originator processes
        MPI_Win_create(schedule, world_size * sizeof(long long), sizeof(long long), MPI_INFO_NULL,
           MPI_COMM_WORLD, &win);

        int ready = 0;
        while (!ready)
        {
           // Without the lock/unlock schedule stays forever filled with 0s
           MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
           ready = fnz(schedule, oldschedule, world_size);
           MPI_Win_unlock(0, win);
        }

        count = tmp;
        for (int i = 1; i < world_size; i++){
            count += schedule[i];
        }

        // Release the window
        MPI_Win_free(&win);
        // Free the allocated memory
        MPI_Free_mem(schedule);
        free(oldschedule);
    }
    else
    {
        // Workers
        // Worker processes do not expose memory in the window
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        // Register with the master
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&tmp, 1, MPI_LONG_LONG, 0, world_rank, 1, MPI_LONG_LONG, win);
        MPI_Win_unlock(0, win);

        // Release the window
        MPI_Win_free(&win);
    }

    if (world_rank == 0)
    {
        // TODO: handle PI result
	    pi_result = 4 * count / (double) tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
