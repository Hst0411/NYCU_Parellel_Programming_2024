# include <mpi.h>
# include <fstream>
# include <iostream>

# define MASTER 0
# define FROM_MASTER 1
# define FROM_WORKER 2


void construct_matrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == MASTER){
        in >> *n_ptr >> *m_ptr >> *l_ptr;
        int n = *n_ptr, m = *m_ptr, l = *l_ptr;
        // printf("Matrix dimensions: n=%d, m=%d, l=%d\n", *n_ptr, *m_ptr, *l_ptr);

        *a_mat_ptr = (int*)malloc(sizeof(int) * n * m);
        *b_mat_ptr = (int*)malloc(sizeof(int) * m * l);

        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                in >> (*a_mat_ptr)[i * m + j];
            }
        }

        for (int i = 0; i < m; i++){
            for (int j = 0; j < l; j++){
                in >> (*b_mat_ptr)[i * l + j];
            }
        }

        /* debug purpose */
        /* test input
        for(int i = 0; i < n*m; i++){
            printf("a_mat_ptr: %d ", *a_mat_ptr[i]);
        }
        printf("\n");
        for(int i = 0; i < m*l; i++){
            printf("b_mat_ptr: %d ", *b_mat_ptr[i]);
        }
        */
    }

    MPI_Bcast(n_ptr, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (world_rank != 0)
    {
        *a_mat_ptr = (int *)malloc(*n_ptr * *m_ptr * sizeof(int));
        *b_mat_ptr = (int *)malloc(*l_ptr * *m_ptr * sizeof(int));
    }

    MPI_Bcast(*a_mat_ptr, *n_ptr * *m_ptr, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, *l_ptr * *m_ptr, MPI_INT, 0, MPI_COMM_WORLD);
}


void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
    int world_size, world_rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int numworker, source, dest, mtype, rows, averow, extra, offset;
    int N, M, L;
    int i, j, k;
    numworker = world_size - 1;
    if (world_rank == MASTER){
	    int *c;
    	c = (int*)malloc(sizeof(int) * n * l);
        /* Send matrix data to the worker tasks */
        averow = n / numworker;
        extra = n % numworker;
        offset = 0;
        mtype = FROM_MASTER;
        for (dest = 1; dest <= numworker; dest++){
            MPI_Send(&n, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            rows = (dest <= extra) ? averow + 1 : averow;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
	        // printf("master send info to rank %d\n", dest);
            MPI_Send(&a_mat[offset * m], rows * m, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&b_mat[0], m * l, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            offset += rows;
        }
        /* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (i = 1; i <= numworker; i++){
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset * l], rows * l, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
	        // printf("master receive from %d\n", source);
        }
        /* Print results */
        for (i = 0; i < n; i++){
            for (j = 0; j < l; j++){
                printf("%d", c[i * l + j]);
                if (j != l-1) printf(" ");
            }
            printf("\n");
        }
	    free(c);
    }
    if (world_rank > MASTER){
        mtype = FROM_MASTER;
        MPI_Recv(&N, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&M, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&L, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
	    int *a;
    	int *b;
    	int *c;
    	a = (int*)malloc(sizeof(int) * N * M);
    	b = (int*)malloc(sizeof(int) * M * L);
    	c = (int*)malloc(sizeof(int) * N * L);
	    // printf("n: %d, m: %d, l: %d\n", N, M, L);
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
	    // printf("rank %d receive from master\n", rank);
        MPI_Recv(&a[0], rows * M, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b[0], M * L, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        // printf("\n");
        // for (int i = 0; i < rows * M; i++) printf("a[%d]: %d\n", i, a[i]);
        // for (int i = 0; i < L * M; i++) printf("b[%d]: %d\n", i, b[i]);
        // printf("\n");

        for (k = 0; k < L; k++){
            for (i = 0; i < rows; i++){
                c[i * L + k] = 0;
                for (j = 0; j < M; j++){
                    c[i * L + k] += a[i * M + j] * b[j * L + k];
                    // printf("a[%d][%d] = %d\n", i, j, a[i*M + j]);
                    // printf("b[%d][%d] = %d\n", j, k, b[j*L + k]);
        	    }
            }
        }

        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c[0], rows * L, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
	    // printf("rank %d send result\n", rank);
	    free(a);
    	free(b);
	    free(c);
    }
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    free(a_mat);
    free(b_mat);
}
