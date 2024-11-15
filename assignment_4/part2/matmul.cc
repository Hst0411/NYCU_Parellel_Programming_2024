#include <mpi.h>
#include <fstream>
#include <iostream>

#define MASTER  0
#define FROM_MASTER 1 /* setting a message type */
#define FROM_WORKER 2 /* setting a message type */

void construct_matrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){

    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0){
        in >> *n_ptr >> *m_ptr >> *l_ptr;

        int n = *n_ptr;
        int m = *m_ptr;
        int l = *l_ptr;

        *a_mat_ptr = (int *)malloc(n * m * sizeof(int));
        *b_mat_ptr = (int *)malloc(m * l * sizeof(int));

        for(int i = 0; i < n*m; i++){
            // read ptr instead of address
            in >> *a_mat_ptr[i];
        }

        for (int i = 0; i < m; i++){
            for (int j = 0; j < l; j++){
                in >> *b_mat_ptr[j * m + i];
            }
        }
        /* test input
        for(int i = 0; i < n*m; i++){
            printf("%d ", *a_mat_ptr[i]);
        }
        printf("\n");
        for(int i = 0; i < m*l; i++){
            printf("%d ", *b_mat_ptr[i]);
        }
        */
    }



}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
    
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Status status;


    int numtasks, /* number of tasks in partition */
        taskid, /* a task identifier */
        numworkers, /* number of worker tasks */
        source, /* task id of message source */
        dest, /* task id of message destination */
        mtype, /* message type */
        rows,  /* rows of matrix A sent to each worker */
        averow, extra, offset; /* used to determine rows sent to each worker */
    
    numworkers = world_size - 1;
    averow = n / numworkers;
    extra = n % numworkers;
    offset = 0;
    mtype = FROM_MASTER;

    /**************************** master task ************************************/
    if(world_rank == 0){

        /* Send matrix data to the worker tasks */
        for(dest = 1; dest <= numworkers; dest++){
            rows = (dest <= extra) ? (averow + 1) : averow;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            printf("master send info to rank %d\n", dest);
            MPI_Send(&a_mat[offset*m], rows*m, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&b_mat[0], m*l, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            offset += rows;
        }

        /* Receive results from worker tasks */
        int *c = (int *)malloc(n * l * sizeof(int));
        mtype = FROM_WORKER;
        for (int i = 1; i <= numworkers; i++){
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset * l], rows*l, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
            printf("master receive from %d\n", source);
        }
        free(c);
    }

    /**************************** worker task ************************************/
    if(world_rank > MASTER){
        int *a = (int *)malloc(n * m * sizeof(int));
        int *b = (int *)malloc(m * l * sizeof(int));
        int *c = (int *)malloc(n * l * sizeof(int));
        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        printf("rank %d receive from master\n", world_rank);
        MPI_Recv(&a[0], rows*m, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b[0], m*l, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        printf("\n");
        for (int i = 0; i < rows * m; i++) printf("a[%d]: %d\n", i, a[i]);
        for (int i = 0; i < l * m; i++) printf("b[%d]: %d\n", i, b[i]);
        printf("\n");

        for (int k = 0; k < l; k++){
            for (int i = 0; i < rows; i++){
                c[i * l + k] = 0.0;
                for (int j = 0; j < m; j++){
                    c[i * l + k] = c[i * l + k] + a_mat[i * m + j] * b_mat[j * l + k];
                    printf("a[%d][%d] = %d\n", i, j, a[i*m + j]);
		            printf("b[%d][%d] = %d\n", j, k, b[j*l + k]);

                }
            }
        }    

        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c[0], rows*l, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        printf("rank %d send result\n", world_rank);
        free(a);
        free(b);
        free(c);

    }
}


void destruct_matrices(int *a_mat, int *b_mat){
    free(a_mat);
    free(b_mat);
}
