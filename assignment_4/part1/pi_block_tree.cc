#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Status status;
    long long num_iter = tosses / world_size;
    unsigned int seed = time(NULL) + world_rank;
    long long count = 0;

    for(int i = 0; i < num_iter; i++){
        double x = (double)rand_r(&seed) / RAND_MAX;  // Random x between -1 and 1
        double y = (double)rand_r(&seed) / RAND_MAX;  // Random y between -1 and 1
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) count++;
    }

    // TODO: binary tree redunction
    long long tmp = 0;
    for(int i = 2; i <= world_size; i *= 2){
        if(world_rank % i == 0){
            MPI_Recv(&tmp, 1, MPI_LONG_LONG, world_rank + (i / 2), 0, MPI_COMM_WORLD, &status);
            count += tmp;
        }
        else if(world_rank % i == (i / 2)){
            MPI_Send(&count, 1, MPI_LONG_LONG, world_rank - (i / 2), 0, MPI_COMM_WORLD);
        }
    }


    if (world_rank == 0)
    {
        // TODO: PI result
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
