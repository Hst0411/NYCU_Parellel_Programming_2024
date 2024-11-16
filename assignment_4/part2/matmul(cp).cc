# include <mpi.h>
# include <fstream>
# include <iostream>
# include <cstring>

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

        for (int i = 0; i < n*m; i++){
            in >> (*a_mat_ptr)[i];
        }

        for (int i = 0; i < m; i++){
            for (int j = 0; j < l; j++){
                in >> (*b_mat_ptr)[j * m + i];
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
    int *local_answer = (int *)malloc(n * l * sizeof(int));
    int *global_answer = (int *)malloc(n * l * sizeof(int));
    memset(local_answer, 0, n * l * sizeof(int));
    memset(global_answer, 0, n * l * sizeof(int));

    int tile_size = 32;

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int start_row = (n / world_size) * world_rank;
    int end_row = (world_rank == world_size - 1) ? n : start_row + (n / world_size);

    for (int ii = start_row; ii < end_row; ii += tile_size)
    {
        for (int jj = 0; jj < l; jj += tile_size)
        {
            for (int i = ii; i < std::min(ii + tile_size, end_row); ++i)
            {
                for (int j = jj; j < std::min(jj + tile_size, l); ++j)
                {
                    int count = 0;
                    for (int k = 0; k < m; ++k)
                    {
                        count += a_mat[i * m + k] * b_mat[j * m + k];
                    }
                    local_answer[i * l + j] = count;
                }
            }
        }
    }


    MPI_Reduce(local_answer, global_answer, n * l, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        char *output_buffer = (char *)malloc(n * l * 11);
        char *ptr = output_buffer;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < l; ++j)
            {
                ptr += sprintf(ptr, "%d ", global_answer[i * l + j]);
            }

            ptr += sprintf(ptr, "\n");
        }

        printf("%s", output_buffer);
        free(output_buffer);
    }

    free(local_answer);
    free(global_answer);
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    free(a_mat);
    free(b_mat);
}
