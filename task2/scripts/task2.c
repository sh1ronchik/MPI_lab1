/* task2.c
   Умножение матрицы на вектор: алгоритмы algo_row, algo_col, algo_block.
   Результаты записываются в CSV: ./task2/data/csv/<prefix>_algo_row.csv, ..._algo_col.csv, ..._algo_block.csv
   Формат CSV: procs,N,overall,comp_max,comm_max

   Usage:
     mpicc -O2 -std=c11 task2/scripts/task2.c -o task2/scripts/task2
     mpiexec -oversubscribe -n 4 ./task2/scripts/task2 1000,2000,4000 mypref
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

static void ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0755) != 0 && errno != EEXIST) {
            fprintf(stderr, "Failed to create directory %s (errno=%d)\n", path, errno);
        }
    }
}

static void append_csv(const char *dir, const char *prefix, const char *algo, int procs, int N,
                       double overall, double comp_max, double comm_max) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s_%s.csv", dir, prefix, algo);
    FILE *f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "Cannot open %s for append\n", path);
        return;
    }
    fprintf(f, "%d,%d,%.9f,%.9f,%.9f\n", procs, N, overall, comp_max, comm_max);
    fclose(f);
}

static int parse_sizes(const char *s, int **out) {
    int cnt = 0;
    char *tmp = strdup(s);
    char *p = strtok(tmp, ",");
    int *arr = NULL;
    while (p) {
        arr = (int*)realloc(arr, sizeof(int)*(cnt+1));
        arr[cnt++] = atoi(p);
        p = strtok(NULL, ",");
    }
    free(tmp);
    *out = arr;
    return cnt;
}

/* ---------------- Row-wise (algo_row) ----------------
   Распределение матрицы по строкам (rows). 
   Root формирует матрицу и вектор, используется MPI_Scatterv для раздачи блоков,
   MPI_Bcast для вектора, локальный вычислительный участок (row-wise) и MPI_Gatherv для сборки результата.
   Комментарии общего плана — детали реализации в коде.
*/
static void algo_row(int N, const char *csv_dir, const char *prefix, MPI_Comm comm) {
    int rank, procs; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &procs);

    int base = N / procs; int rem = N % procs;
    int *rows = malloc(procs * sizeof(int));
    int *row_disp = malloc(procs * sizeof(int));
    int off = 0;
    for (int i=0;i<procs;i++){ rows[i]=base+(i<rem?1:0); row_disp[i]=off; off+=rows[i]; }
    int local_rows = rows[rank];

    double *mat_local = malloc((size_t)local_rows * N * sizeof(double));
    double *vec = malloc(N * sizeof(double));
    double *y_local = malloc(local_rows * sizeof(double));
    double *y = NULL;

    if (rank==0) {
        double *mat = malloc((size_t)N * N * sizeof(double));
        srand((unsigned)time(NULL));
        for (int i=0;i<N*N;i++) mat[i] = (double)(rand()%10);
        for (int i=0;i<N;i++) vec[i] = (double)(rand()%10);

        int *sendcounts = malloc(procs*sizeof(int));
        int *senddispls = malloc(procs*sizeof(int));
        for (int i=0;i<procs;i++){ sendcounts[i]=rows[i]*N; senddispls[i]=row_disp[i]*N; }

        MPI_Scatterv(mat, sendcounts, senddispls, MPI_DOUBLE, mat_local, sendcounts[rank], MPI_DOUBLE, 0, comm);

        free(sendcounts); free(senddispls); free(mat);
    } else {
        int recvcount = local_rows * N;
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, mat_local, recvcount, MPI_DOUBLE, 0, comm);
    }

    MPI_Bcast(vec, N, MPI_DOUBLE, 0, comm);

    MPI_Barrier(comm);
    double overall_start = MPI_Wtime();

    double comp_start = MPI_Wtime();
    for (int i=0;i<local_rows;i++){
        double s=0.0;
        double *rowptr = mat_local + (size_t)i * N;
        for (int j=0;j<N;j++) s += rowptr[j] * vec[j];
        y_local[i] = s;
    }
    double comp_end = MPI_Wtime(); double comp_local = comp_end - comp_start;

    double comm_start = MPI_Wtime();
    if (rank==0) {
        y = malloc(N * sizeof(double));
        int *recvcounts = malloc(procs*sizeof(int));
        int *recvdispls = malloc(procs*sizeof(int));
        for (int i=0;i<procs;i++){ recvcounts[i]=rows[i]; recvdispls[i]=row_disp[i]; }
        MPI_Gatherv(y_local, local_rows, MPI_DOUBLE, y, recvcounts, recvdispls, MPI_DOUBLE, 0, comm);
        free(recvcounts); free(recvdispls);
    } else {
        MPI_Gatherv(y_local, local_rows, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, comm);
    }
    double comm_end = MPI_Wtime(); double comm_local = comm_end - comm_start;

    MPI_Barrier(comm);
    double overall_end = MPI_Wtime();

    double comp_max=0.0, comm_max=0.0;
    MPI_Reduce(&comp_local, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&comm_local, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank==0) {
        append_csv(csv_dir, prefix, "algo_row", procs, N, overall_end-overall_start, comp_max, comm_max);
    }

    free(mat_local); free(vec); free(y_local); free(rows); free(row_disp); if(y) free(y);
}

/* ---------------- Column-wise (algo_col) ----------------
   Распределение по столбцам (cols). 
   Root готовит колонки в формат для MPI_Scatterv, каждому процессу даётся подматрица и соответствующая часть вектора.
   Локальное суммирование по колонкам, затем глобальное MPI_Reduce (SUM) для получения окончательного y.
*/
static void algo_col(int N, const char *csv_dir, const char *prefix, MPI_Comm comm) {
    int rank, procs; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &procs);

    int base = N / procs; int rem = N % procs;
    int *cols = malloc(procs*sizeof(int));
    int *col_disp = malloc(procs*sizeof(int));
    int off = 0;
    for (int i=0;i<procs;i++){ cols[i]=base+(i<rem?1:0); col_disp[i]=off; off+=cols[i]; }
    int local_cols = cols[rank];

    double *mat_sub = malloc((size_t)N * local_cols * sizeof(double));
    double *vec_sub = malloc(local_cols * sizeof(double));
    double *partial_y = malloc(N * sizeof(double));

    if (rank==0) {
        double *mat = malloc((size_t)N * N * sizeof(double));
        double *vec = malloc(N * sizeof(double));
        srand((unsigned)time(NULL)+1234);
        for (int i=0;i<N*N;i++) mat[i] = (double)(rand()%10);
        for (int i=0;i<N;i++) vec[i] = (double)(rand()%10);

        int *sendcounts = malloc(procs*sizeof(int));
        int *senddispls = malloc(procs*sizeof(int));
        int pos = 0;
        for (int p=0;p<procs;p++){ sendcounts[p]=N*cols[p]; senddispls[p]=pos; pos+=sendcounts[p]; }

        double *bigbuf = malloc((size_t)N * N * sizeof(double));
        int bufpos=0;
        for (int p=0;p<procs;p++){
            for (int c=col_disp[p]; c<col_disp[p]+cols[p]; c++){
                for (int r=0;r<N;r++) bigbuf[bufpos++] = mat[r*N + c];
            }
        }
        double *bigvec = malloc((size_t)N * sizeof(double));
        int vpos=0;
        for (int p=0;p<procs;p++){
            for (int c=col_disp[p]; c<col_disp[p]+cols[p]; c++) bigvec[vpos++] = vec[c];
        }

        MPI_Scatterv(bigbuf, sendcounts, senddispls, MPI_DOUBLE, mat_sub, N*local_cols, MPI_DOUBLE, 0, comm);
        MPI_Scatterv(bigvec, cols, col_disp, MPI_DOUBLE, vec_sub, local_cols, MPI_DOUBLE, 0, comm);

        free(sendcounts); free(senddispls); free(bigbuf); free(bigvec); free(mat); free(vec);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, mat_sub, N*local_cols, MPI_DOUBLE, 0, comm);
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, vec_sub, local_cols, MPI_DOUBLE, 0, comm);
    }

    MPI_Barrier(comm);
    double overall_start = MPI_Wtime();

    double comp_start = MPI_Wtime();
    for (int i=0;i<N;i++) partial_y[i]=0.0;
    for (int c=0;c<local_cols;c++){
        double v = vec_sub[c];
        double *colptr = mat_sub + (size_t)c * N;
        for (int r=0;r<N;r++) partial_y[r] += colptr[r] * v;
    }
    double comp_end = MPI_Wtime(); double comp_local = comp_end - comp_start;

    double comm_start = MPI_Wtime();
    double *y = NULL;
    if (rank==0) y = malloc(N * sizeof(double));
    MPI_Reduce(partial_y, y, N, MPI_DOUBLE, MPI_SUM, 0, comm);
    double comm_end = MPI_Wtime(); double comm_local = comm_end - comm_start;

    MPI_Barrier(comm);
    double overall_end = MPI_Wtime();

    double comp_max=0.0, comm_max=0.0;
    MPI_Reduce(&comp_local, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&comm_local, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank==0) append_csv(csv_dir, prefix, "algo_col", procs, N, overall_end-overall_start, comp_max, comm_max);

    free(mat_sub); free(vec_sub); free(partial_y); free(cols); free(col_disp); if(y) free(y);
}

/* ---------------- Block-wise (algo_block) ----------------
   Блочное распределение по двум осям (2D Cartesian, MPI_Cart_create).
   Формирование блоков (Ablock, vblock) на root и рассылка по cart topology.
   Локальное вычисление для блока, затем редукции по строкам и сбор на global root.
   Комментарии общего плана — подробности в коде (теги сообщений, row-root, world-rank и т.п. сохранены).
*/
static void algo_block(int N, const char *csv_dir, const char *prefix, MPI_Comm comm) {
    int rank, procs; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &procs);

    int dims[2] = {0,0};
    MPI_Dims_create(procs, 2, dims);
    int nprow = dims[0], npcol = dims[1];

    int periods[2]={0,0};
    MPI_Comm cart;
    MPI_Cart_create(comm, 2, dims, periods, 1, &cart);

    int cart_rank; MPI_Comm_rank(cart, &cart_rank);
    int coords[2]; MPI_Cart_coords(cart, cart_rank, 2, coords);

    int *rows_per = malloc(nprow * sizeof(int));
    int *rows_disp = malloc(nprow * sizeof(int));
    int base_r = N / nprow, rem_r = N % nprow; int off=0;
    for (int i=0;i<nprow;i++){ rows_per[i]=base_r + (i<rem_r?1:0); rows_disp[i]=off; off+=rows_per[i]; }

    int *cols_per = malloc(npcol * sizeof(int));
    int *cols_disp = malloc(npcol * sizeof(int));
    int base_c = N / npcol, rem_c = N % npcol; off=0;
    for (int j=0;j<npcol;j++){ cols_per[j]=base_c + (j<rem_c?1:0); cols_disp[j]=off; off+=cols_per[j]; }

    int brow = rows_per[coords[0]];
    int bcol = cols_per[coords[1]];

    double *Ablock = malloc((size_t)brow * bcol * sizeof(double));
    double *vblock = malloc((size_t)bcol * sizeof(double));
    double *y_partial = malloc((size_t)brow * sizeof(double));

    if (rank==0) {
        double *A = malloc((size_t)N * N * sizeof(double));
        double *v = malloc((size_t)N * sizeof(double));
        srand((unsigned)time(NULL)+999);
        for (int i=0;i<N*N;i++) A[i] = (double)(rand()%10);
        for (int i=0;i<N;i++) v[i] = (double)(rand()%10);

        for (int pr=0; pr<nprow; pr++){
            for (int pc=0; pc<npcol; pc++){
                int dest_coords[2] = {pr, pc};
                int dest_rank; MPI_Cart_rank(cart, dest_coords, &dest_rank);
                int br = rows_per[pr], bc = cols_per[pc];
                double *buf = malloc((size_t)br*bc*sizeof(double));
                int p=0;
                for (int i=rows_disp[pr]; i<rows_disp[pr]+br; i++){
                    for (int j=cols_disp[pc]; j<cols_disp[pc]+bc; j++){
                        buf[p++] = A[i*N + j];
                    }
                }
                if (dest_rank==0) memcpy(Ablock, buf, (size_t)br*bc*sizeof(double));
                else MPI_Send(buf, br*bc, MPI_DOUBLE, dest_rank, 100 + pr*npcol + pc, comm);
                free(buf);
            }
        }
        for (int pc=0; pc<npcol; pc++){
            int bc = cols_per[pc];
            double *bufv = malloc(bc * sizeof(double));
            for (int j=0;j<bc;j++) bufv[j] = v[cols_disp[pc] + j];
            for (int pr=0; pr<nprow; pr++){
                int dest_coords[2] = {pr, pc};
                int dest_rank; MPI_Cart_rank(cart, dest_coords, &dest_rank);
                if (dest_rank==0) memcpy(vblock, bufv, bc*sizeof(double));
                else MPI_Send(bufv, bc, MPI_DOUBLE, dest_rank, 200 + pc, comm);
            }
            free(bufv);
        }
        free(A); free(v);
    } else {
        MPI_Recv(Ablock, brow*bcol, MPI_DOUBLE, 0, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        MPI_Recv(vblock, bcol, MPI_DOUBLE, 0, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(comm);
    double overall_start = MPI_Wtime();

    double comp_start = MPI_Wtime();
    for (int i=0;i<brow;i++){
        double s=0.0;
        for (int j=0;j<bcol;j++) s += Ablock[i*bcol + j] * vblock[j];
        y_partial[i] = s;
    }
    double comp_end = MPI_Wtime(); double comp_local = comp_end - comp_start;

    MPI_Comm row_comm;
    MPI_Comm_split(cart, coords[0], coords[1], &row_comm);
    int row_rank; MPI_Comm_rank(row_comm, &row_rank);

    double comm_start = MPI_Wtime();
    double *y_row = NULL;
    if (row_rank==0) y_row = malloc((size_t)brow * sizeof(double));
    MPI_Reduce(y_partial, y_row, brow, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    if (row_rank==0) {
        /* row-root (процесс с column coordinate == 0 в строке) отправляет свой агрегат на global root (world-rank 0), tag = 300 + row_index */
        int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        if (world_rank == 0) {
            /* global root собирает блоки от всех row-root'ов (включая собственный) */
            double *y = malloc((size_t)N * sizeof(double));
            /* копируем свой блок в итоговый буфер */
            memcpy(y + rows_disp[coords[0]], y_row, rows_per[coords[0]] * sizeof(double));
            /* приём блоков от остальных row-root'ов */
            for (int rr = 0; rr < nprow; rr++) {
                if (rr == coords[0]) continue;
                MPI_Status st; MPI_Recv(y + rows_disp[rr], rows_per[rr], MPI_DOUBLE, MPI_ANY_SOURCE, 300 + rr, MPI_COMM_WORLD, &st);
            }
            free(y);
        } else {
            /* отправляем на global root с tag = 300 + row_index */
            MPI_Send(y_row, rows_per[coords[0]], MPI_DOUBLE, 0, 300 + coords[0], MPI_COMM_WORLD);
        }
    }
    double comm_end = MPI_Wtime(); double comm_local = comm_end - comm_start;

    MPI_Barrier(comm);
    double overall_end = MPI_Wtime();

    double comp_max=0.0, comm_max=0.0;
    MPI_Reduce(&comp_local, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&comm_local, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank==0) append_csv(csv_dir, prefix, "algo_block", procs, N, overall_end-overall_start, comp_max, comm_max);

    if (row_rank==0 && y_row) free(y_row);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&cart);
    free(Ablock); free(vblock); free(y_partial);
    free(rows_per); free(rows_disp); free(cols_per); free(cols_disp);
}

/* ---------------- main ----------------
   Парсинг аргументов, создание директорий, запуск последовательности алгоритмов для каждого N.
*/
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank==0) fprintf(stderr, "Usage: %s <N[,N,...]> [prefix]\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    int *sizes = NULL; int cnt = parse_sizes(argv[1], &sizes);
    const char *prefix = (argc >= 3) ? argv[2] : "mpi_mv";
    const char *csv_dir = "./task2/data/csv";

    if (rank==0) { ensure_dir("./task2"); ensure_dir("./task2/data"); ensure_dir(csv_dir); }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i=0;i<cnt;i++){
        int N = sizes[i];
        if (rank==0) printf("Running algorithms for N=%d\n", N);
        algo_row(N, csv_dir, prefix, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        algo_col(N, csv_dir, prefix, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        algo_block(N, csv_dir, prefix, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank==0) printf("Done N=%d\n", N);
    }

    free(sizes);
    MPI_Finalize();
    return 0;
}
