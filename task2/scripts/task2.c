/* task2.c — поддерживает прямоугольные матрицы R x C
   Три алгоритма:
     - algo_row   (разбиение по строкам)
     - algo_col   (разбиение по столбцам)
     - algo_block (2D блочное разбиение)

   CSV: ./task2/data/csv/<prefix>_algo_row.csv и т.п.
   Строка: procs,rows,cols,overall,comp_max,comm_max

   Сборка:
     mpicc -O2 -std=c11 task2/scripts/task2.c -o task2/scripts/task2

   Запуск (пример):
     mpiexec -n 4 ./task2/scripts/task2 "1000x800,2000x500,3000" myprefix
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>
#include <math.h>

/* --- Утилиты файловой системы --- */
static void ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0755) != 0 && errno != EEXIST) {
            fprintf(stderr, "Не удалось создать каталог %s (errno=%d)\n", path, errno);
        }
    }
}

/* --- Запись результата в CSV --- */
static void append_csv(const char *dir, const char *prefix, const char *algo,
                       int procs, int rows, int cols,
                       double overall, double comp_max, double comm_max) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s_%s.csv", dir, prefix, algo);
    FILE *f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "Не могу открыть %s для дозаписи\n", path);
        return;
    }
    /* Формат: procs,rows,cols,overall,comp_max,comm_max */
    fprintf(f, "%d,%d,%d,%.9f,%.9f,%.9f\n", procs, rows, cols, overall, comp_max, comm_max);
    fclose(f);
}

/* parse_sizes_scan
   Разбирает строку вида "R1xC1,R2xC2,...", поддерживает также одиночные числа "N" => N x N.
*/
static int parse_sizes_scan(const char *s, int **out_rows, int **out_cols) {
    if (!s) { *out_rows = NULL; *out_cols = NULL; return 0; }

    int cap = 8;
    int cnt = 0;
    int *rows = malloc(cap * sizeof(int));
    int *cols = malloc(cap * sizeof(int));
    if (!rows || !cols) { free(rows); free(cols); return 0; }

    const char *p = s;
    while (*p) {
        /* пропускаем пробелы и запятые */
        while (*p == ' ' || *p == '\t' || *p == ',' ) ++p;
        if (!*p) break;

        /* читаем первое число R с помощью strtol */
        char *endptr = NULL;
        long r = strtol(p, &endptr, 10);
        if (endptr == p) {
            /* не число — пропускаем до следующей запятой */
            while (*p && *p != ',') ++p;
            continue;
        }
        p = endptr;

        int got_c = 0;
        long c = r;

        /* если следующий символ 'x' или 'X' — читаем второе число */
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == 'x' || *p == 'X') {
            ++p;
            while (*p == ' ' || *p == '\t') ++p;
            char *endptr2 = NULL;
            long t = strtol(p, &endptr2, 10);
            if (endptr2 == p) {
                /* неверный формат 'Nx' без второго числа: пропускаем до запятой */
                while (*p && *p != ',') ++p;
                continue;
            }
            c = t;
            p = endptr2;
            got_c = 1;
        }

        /* теперь у нас есть r и c (c = r если не было 'x') */
        if (r > 0 && c > 0) {
            if (cnt >= cap) {
                int newcap = cap * 2;
                int *nr = realloc(rows, newcap * sizeof(int));
                int *nc = realloc(cols, newcap * sizeof(int));
                if (!nr || !nc) { free(rows); free(cols); return 0; }
                rows = nr; cols = nc; cap = newcap;
            }
            rows[cnt] = (int)r;
            cols[cnt] = (int)c;
            cnt++;
        }

        /* пропускаем всё до следующей запятой (если есть) и продолжим */
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }

    *out_rows = rows;
    *out_cols = cols;
    return cnt;
}


/* --- Вспомогательные: расчёт counts/displs для строк и столбцов --- */
static void build_rows_counts(int R, int procs, int *rows, int *row_disp) {
    int base = R / procs;
    int rem = R % procs;
    int off = 0;
    for (int i = 0; i < procs; ++i) {
        rows[i] = base + (i < rem ? 1 : 0);
        row_disp[i] = off;
        off += rows[i];
    }
}

static void build_cols_counts(int C, int procs, int *cols, int *col_disp) {
    int base = C / procs;
    int rem = C % procs;
    int off = 0;
    for (int i = 0; i < procs; ++i) {
        cols[i] = base + (i < rem ? 1 : 0);
        col_disp[i] = off;
        off += cols[i];
    }
}

/* --- Безопасный malloc: выделяет как минимум 1 байт, чтобы не передавать NULL в MPI --- */
static void *safe_malloc(size_t bytes) {
    if (bytes == 0) bytes = 1;
    return malloc(bytes);
}

/* --- Функции для печати матрицы/вектора --- */
static void print_matrix(const char *name, int R, int C, const double *A) {
    printf("%s (%dx%d):\n", name, R, C);
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            printf("%8.3f ", A[i * (size_t)C + j]);
        }
        printf("\n");
    }
}

/* Печать вектора */
static void print_vector(const char *name, int n, const double *v) {
    printf("%s (len=%d):", name, n);
    for (int i = 0; i < n; ++i) {
        printf(" %8.3f", v[i]);
    }
    printf("\n");
}

/* Генерация матрицы R x C и вектора длины C с заданным seed.
   Возвращает выделенные массивы A и v (нужно free). */
static void generate_mat_vec(int R, int C, unsigned int seed, double **outA, double **outv) {
    *outA = malloc((size_t)R * (size_t)C * sizeof(double));
    *outv = malloc((size_t)C * sizeof(double));
    srand((unsigned)seed);
    for (long long i = 0; i < (long long)R * C; ++i) (*outA)[i] = (double)(rand() % 10);
    for (int j = 0; j < C; ++j) (*outv)[j] = (double)(rand() % 10);
}

/* Вычисление последовательного результата y = A * v (A: R x C, v: C) */
static void seq_matvec(int R, int C, const double *A, const double *v, double *y_out) {
    for (int i = 0; i < R; ++i) {
        double s = 0.0;
        const double *rowptr = A + (size_t)i * C;
        for (int j = 0; j < C; ++j) s += rowptr[j] * v[j];
        y_out[i] = s;
    }
}

/* ---------------- algo_row ----------------
   Разбиение по строкам:
   - root генерирует матрицу R x C и вектор длины C (фиксированный seed),
   - root делает MPI_Scatterv по строкам (каждый получает rows[i] строк),
   - MPI_Bcast вектор, локальное умножение (local_rows x C),
   - MPI_Gatherv собирает результат y (длиной R) на root.
*/
static void algo_row(int R, int C, const char *csv_dir, const char *prefix, MPI_Comm comm, unsigned int fixed_seed) {
    int rank, procs; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &procs);

    /* build rows/disp — сколько строк у каждого процесса */
    int *rows = malloc(procs * sizeof(int));
    int *row_disp = malloc(procs * sizeof(int));
    build_rows_counts(R, procs, rows, row_disp);
    int local_rows = rows[rank];

    /* локальные буферы */
    double *mat_local = (double*) safe_malloc((size_t)local_rows * (size_t)C * sizeof(double));
    double *vec = NULL; 
    double *y_local = (double*) safe_malloc((size_t)local_rows * sizeof(double));
    double *y = NULL; /* итоговый вектор */
    double *mat = NULL;

    /* root генерирует полные A и v, затем упаковывает / scatterv */
    if (rank == 0) {
        generate_mat_vec(R, C, fixed_seed, &mat, &vec);
    }

    /* синхронизация и замер времени */
    MPI_Barrier(comm);
    double overall_start = MPI_Wtime();

    double comm_start1 = MPI_Wtime();
    if (rank == 0) {
        int *sendcounts = malloc(procs * sizeof(int));
        int *senddispls = malloc(procs * sizeof(int));
        for (int i = 0; i < procs; ++i) { sendcounts[i] = rows[i] * C; senddispls[i] = row_disp[i] * C; }

        MPI_Scatterv(mat, sendcounts, senddispls, MPI_DOUBLE,
                     mat_local, sendcounts[rank], MPI_DOUBLE, 0, comm);

        free(sendcounts); free(senddispls);
        free(mat);
    } else {
        int recvcount = local_rows * C;
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, mat_local, recvcount, MPI_DOUBLE, 0, comm);

        /* ненулевые ранги выделяют буфер vec, чтобы получить Bcast */
        vec = (double*) safe_malloc((size_t)C * sizeof(double));
    }

    /* раздаём вектор всем */
    MPI_Bcast(vec, C, MPI_DOUBLE, 0, comm);
    double comm_end1 = MPI_Wtime(); double comm_local1 = comm_end1 - comm_start1;
    
    /* локальная часть: умножение локальных строк на вектор */
    double comp_start = MPI_Wtime();
    for (int i = 0; i < local_rows; ++i) {
        double s = 0.0;
        double *rowptr = mat_local + (size_t)i * C;
        for (int j = 0; j < C; ++j) s += rowptr[j] * vec[j];
        y_local[i] = s;
    }
    double comp_end = MPI_Wtime(); double comp_local = comp_end - comp_start;

    /* сбор локальных результатов на root */
    double comm_start = MPI_Wtime();
    if (rank == 0) {
        y = malloc((size_t)R * sizeof(double));
        int *recvcounts = malloc(procs * sizeof(int));
        int *recvdispls = malloc(procs * sizeof(int));
        for (int i = 0; i < procs; ++i) { recvcounts[i] = rows[i]; recvdispls[i] = row_disp[i]; }
        MPI_Gatherv(y_local, local_rows, MPI_DOUBLE, y, recvcounts, recvdispls, MPI_DOUBLE, 0, comm);
        free(recvcounts); free(recvdispls);
    } else {
        MPI_Gatherv(y_local, local_rows, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, comm);
    }
    double comm_end = MPI_Wtime(); double comm_local = comm_end - comm_start;

    MPI_Barrier(comm);
    double overall_end = MPI_Wtime();

    comm_local += comm_local1;
    /* глобальные сводные времена (максимум по процессам) */
    double comp_max = 0.0, comm_max = 0.0;
    MPI_Reduce(&comp_local, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&comm_local, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    /* Запись результата в CSV */
    if (rank == 0) {
        append_csv(csv_dir, prefix, "algo_row", procs, R, C, overall_end - overall_start, comp_max, comm_max);
    }

    /* Если матрица небольшая, выведем матрицу, вектор и результат.
       и вычислим эталонный результат для сравнения. */
    if (rank == 0 && R <= 5 && C <= 5) {
        double *A_ref = NULL, *v_ref = NULL;
        generate_mat_vec(R, C, fixed_seed, &A_ref, &v_ref);
        double *y_ref = malloc((size_t)R * sizeof(double));
        seq_matvec(R, C, A_ref, v_ref, y_ref);

        printf("\n[algo_row] Проверка на root (R<=5 && C<=5)\n");
        print_matrix("A", R, C, A_ref);
        print_vector("v", C, v_ref);
        print_vector("y (алгоритм)", R, y);
        print_vector("y (реф)", R, y_ref);

        free(A_ref); free(v_ref); free(y_ref);
    }

    free(mat_local); free(vec); free(y_local); free(rows); free(row_disp);
    if (y) free(y);
}

/* ---------------- algo_col ----------------
   Разбиение по столбцам:
   - root генерирует A (R x C) и v (C), пакует столбцы последовательно в bigbuf,
   - MPI_Scatterv раздаёт каждому процессу блок R x local_cols и соответствующую часть вектора,
   - каждый процесс вычисляет partial_y (размер R),
   - MPI_Reduce(SUM) собирает полный y на root.
*/
static void algo_col(int R, int C, const char *csv_dir, const char *prefix, MPI_Comm comm, unsigned int fixed_seed) {
    int rank, procs; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &procs);

    int *cols = malloc(procs * sizeof(int));
    int *col_disp = malloc(procs * sizeof(int));
    build_cols_counts(C, procs, cols, col_disp);
    int local_cols = cols[rank];

    double *mat_sub = (double*) safe_malloc((size_t)R * (size_t)local_cols * sizeof(double));
    double *vec_sub = (double*) safe_malloc((size_t)local_cols * sizeof(double));
    double *partial_y = (double*) safe_malloc((size_t)R * sizeof(double));
    double *y = NULL; /* итоговый вектор на root */

    double *mat = NULL;
    double *vec = NULL;

    /* root генерирует, упаковывает в bigbuf столбцы подряд (каждая колонка — R элементов) */
    if (rank == 0) {
        generate_mat_vec(R, C, fixed_seed, &mat, &vec);
    }

    MPI_Barrier(comm);
    double overall_start = MPI_Wtime();

    double comm_start1 = MPI_Wtime();
    if (rank == 0) {
        int *sendcounts = malloc(procs * sizeof(int));
        int *senddispls = malloc(procs * sizeof(int));
        int pos = 0;
        for (int p = 0; p < procs; ++p) { sendcounts[p] = R * cols[p]; senddispls[p] = pos; pos += sendcounts[p]; }

        double *bigbuf = malloc((size_t)R * (size_t)C * sizeof(double));
        int bufpos = 0;

        for (int p = 0; p < procs; ++p) {
            for (int c = col_disp[p]; c < col_disp[p] + cols[p]; ++c) {
                for (int r = 0; r < R; ++r) bigbuf[bufpos++] = mat[r * C + c];
            }
        }

        double *bigvec = malloc((size_t)C * sizeof(double));
        int vpos = 0;
        for (int p = 0; p < procs; ++p) {
            for (int c = col_disp[p]; c < col_disp[p] + cols[p]; ++c) bigvec[vpos++] = vec[c];
        }

        MPI_Scatterv(bigbuf, sendcounts, senddispls, MPI_DOUBLE, mat_sub, R * local_cols, MPI_DOUBLE, 0, comm);
        MPI_Scatterv(bigvec, cols, col_disp, MPI_DOUBLE, vec_sub, local_cols, MPI_DOUBLE, 0, comm);

        free(sendcounts); free(senddispls); free(bigbuf); free(bigvec);
        free(mat); free(vec);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, mat_sub, R * local_cols, MPI_DOUBLE, 0, comm);
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, vec_sub, local_cols, MPI_DOUBLE, 0, comm);
    }

    double comm_end1 = MPI_Wtime(); double comm_local1 = comm_end1 - comm_start1;

    /* локальная часть: суммируем вклады от всех локальных колонок в partial_y (размер R) */
    double comp_start = MPI_Wtime();
    for (int i = 0; i < R; ++i) partial_y[i] = 0.0;
    for (int c = 0; c < local_cols; ++c) {
        double vloc = vec_sub[c];
        double *colptr = mat_sub + (size_t)c * R; 
        for (int r = 0; r < R; ++r) partial_y[r] += colptr[r] * vloc;
    }
    double comp_end = MPI_Wtime(); double comp_local = comp_end - comp_start;

    /* редукция по суммированию partial_y в y (на root) */
    double comm_start = MPI_Wtime();
    if (rank == 0) y = malloc((size_t)R * sizeof(double));
    MPI_Reduce(partial_y, y, R, MPI_DOUBLE, MPI_SUM, 0, comm);
    double comm_end = MPI_Wtime(); double comm_local = comm_end - comm_start;

    MPI_Barrier(comm);
    double overall_end = MPI_Wtime();
    
    comm_local += comm_local1;
    double comp_max = 0.0, comm_max = 0.0;
    MPI_Reduce(&comp_local, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&comm_local, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) append_csv(csv_dir, prefix, "algo_col", procs, R, C, overall_end - overall_start, comp_max, comm_max);

    if (rank == 0 && R <= 5 && C <= 5) {
        double *A_ref = NULL, *v_ref = NULL;
        generate_mat_vec(R, C, fixed_seed, &A_ref, &v_ref);
        double *y_ref = malloc((size_t)R * sizeof(double));
        seq_matvec(R, C, A_ref, v_ref, y_ref);

        printf("\n[algo_col] Проверка на root (R<=5 && C<=5)\n");
        print_matrix("A", R, C, A_ref);
        print_vector("v", C, v_ref);
        print_vector("y (алгоритм)", R, y);
        print_vector("y (реф)", R, y_ref);

        free(A_ref); free(v_ref); free(y_ref);
    }

    free(mat_sub); free(vec_sub); free(partial_y); free(cols); free(col_disp);
    if (y) free(y);
}

/* ---------------- algo_block ----------------
   2D-блочное разбиение:
   - создаём декартовый коммуникатор (MPI_Cart_create) с reorder=0,
   - root упаковывает каждый блок (rows_per[pr] x cols_per[pc]) и рассылает их
   - локальная часть умножает Ablock (brow x bcol) на vblock (bcol),
   - выполняется редукция по строкам (MPI_Reduce в row_comm), чтобы собрать суммы по строкам,
   - row-root'ы отправляют свои агрегированные блоки на global root (world rank 0),
   - на global root собирается итоговый y длины R.
*/
static void algo_block(int R, int C, const char *csv_dir, const char *prefix, MPI_Comm comm, unsigned int fixed_seed) {
    int rank_world, procs; MPI_Comm_rank(comm, &rank_world); MPI_Comm_size(comm, &procs);

    int dims[2] = {0, 0};
    MPI_Dims_create(procs, 2, dims); /* распределит числа процессов по двум измерениям */
    int nprow = dims[0], npcol = dims[1];

    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(comm, 2, dims, periods, 0, &cart); /* reorder = 0 чтобы cart-rank совпадал с world-rank */

    int cart_rank; MPI_Comm_rank(cart, &cart_rank);
    int coords[2]; MPI_Cart_coords(cart, cart_rank, 2, coords);

    /* распределяем строки по nprow и столбцы по npcol */
    int *rows_per = malloc(nprow * sizeof(int));
    int *rows_disp = malloc(nprow * sizeof(int));
    int base_r = R / nprow, rem_r = R % nprow, off = 0;
    for (int i = 0; i < nprow; ++i) { rows_per[i] = base_r + (i < rem_r ? 1 : 0); rows_disp[i] = off; off += rows_per[i]; }

    int *cols_per = malloc(npcol * sizeof(int));
    int *cols_disp = malloc(npcol * sizeof(int));
    int base_c = C / npcol, rem_c = C % npcol; off = 0;
    for (int j = 0; j < npcol; ++j) { cols_per[j] = base_c + (j < rem_c ? 1 : 0); cols_disp[j] = off; off += cols_per[j]; }

    int brow = rows_per[coords[0]]; /* количество строк в локальном блоке */
    int bcol = cols_per[coords[1]]; /* количество столбцов в локальном блоке */

    double *Ablock = (double*) safe_malloc((size_t)brow * (size_t)bcol * sizeof(double));
    double *vblock = (double*) safe_malloc((size_t)bcol * sizeof(double));
    double *y_partial = (double*) safe_malloc((size_t)brow * sizeof(double));
    double *y_global = NULL; /* итоговый вектор на global root (world rank 0) */
    double *A = NULL, *v = NULL;
    
    /* root (world rank 0) формирует полную матрицу и рассылает блоки.*/
    if (rank_world == 0) {
        generate_mat_vec(R, C, fixed_seed, &A, &v);
    }

    MPI_Barrier(comm);
    double overall_start = MPI_Wtime();

    double comm_start1 = MPI_Wtime();
    if (rank_world == 0) {
        for (int pr = 0; pr < nprow; ++pr) {
            for (int pc = 0; pc < npcol; ++pc) {
                int dest_coords[2] = { pr, pc };
                int dest_rank; MPI_Cart_rank(cart, dest_coords, &dest_rank);
                int br = rows_per[pr], bc = cols_per[pc];
                double *buf = malloc((size_t)br * (size_t)bc * sizeof(double));
                int p = 0;
                /* упаковываем блок поблочно (по строкам блока и по столбцам блока) */
                for (int i = rows_disp[pr]; i < rows_disp[pr] + br; ++i) {
                    for (int j = cols_disp[pc]; j < cols_disp[pc] + bc; ++j) {
                        buf[p++] = A[i * (size_t)C + j];
                    }
                }
                if (dest_rank == cart_rank) {
                    /* если блок предназначен самому root'у — просто копируем в Ablock */
                    memcpy(Ablock, buf, (size_t)br * (size_t)bc * sizeof(double));
                } else {
                    MPI_Send(buf, br * bc, MPI_DOUBLE, dest_rank, 100 + pr * npcol + pc, MPI_COMM_WORLD);
                }
                free(buf);
            }
        }

        /* рассылка кусочков вектора по колонкам: для каждой колонки создаём буфер и шлём всем процессам,
           у которых соответствующая колонка принадлежит их блоку */
        for (int pc = 0; pc < npcol; ++pc) {
            int bc = cols_per[pc];
            double *bufv = malloc((size_t)bc * sizeof(double));
            for (int j = 0; j < bc; ++j) bufv[j] = v[cols_disp[pc] + j];
            for (int pr = 0; pr < nprow; ++pr) {
                int dest_coords[2] = { pr, pc };
                int dest_rank; MPI_Cart_rank(cart, dest_coords, &dest_rank);
                if (dest_rank == cart_rank) {
                    memcpy(vblock, bufv, bc * sizeof(double));
                } else {
                    MPI_Send(bufv, bc, MPI_DOUBLE, dest_rank, 200 + pc, MPI_COMM_WORLD);
                }
            }
            free(bufv);
        }
        free(A); free(v);
    } else {
        /* остальные процессы принимают свой блок и фрагмент вектора */
        MPI_Recv(Ablock, brow * bcol, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(vblock, bcol, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double comm_end1 = MPI_Wtime(); double comm_local1 = comm_end1 - comm_start1;

    /* локальная часть умножения блока */
    double comp_start = MPI_Wtime();
    for (int i = 0; i < brow; ++i) {
        double s = 0.0;
        for (int j = 0; j < bcol; ++j) s += Ablock[i * bcol + j] * vblock[j];
        y_partial[i] = s;
    }
    double comp_end = MPI_Wtime(); double comp_local = comp_end - comp_start;

    /*
      Теперь по строкам (фиксируем координату строки в grid) делаем MPI_Reduce.
    */
    MPI_Comm row_comm;
    MPI_Comm_split(cart, coords[0], coords[1], &row_comm);
    int row_rank; MPI_Comm_rank(row_comm, &row_rank);

    double comm_start = MPI_Wtime();
    double *y_row = NULL;
    if (row_rank == 0) y_row = malloc((size_t)brow * sizeof(double));
    MPI_Reduce(y_partial, y_row, brow, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    /* row-root'ы отправляют свои агрегированные блоки на global root (world rank 0). */
    if (row_rank == 0) {
        int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        if (world_rank == 0) {
            y_global = malloc((size_t)R * sizeof(double));
            /* копируем собственный блок */
            memcpy(y_global + rows_disp[coords[0]], y_row, rows_per[coords[0]] * sizeof(double));
            /* принимаем блоки от остальных row-root'ов */
            for (int rr = 0; rr < nprow; ++rr) {
                if (rr == coords[0]) continue;
                MPI_Status st;
                MPI_Recv(y_global + rows_disp[rr], rows_per[rr], MPI_DOUBLE, MPI_ANY_SOURCE, 300 + rr, MPI_COMM_WORLD, &st);
            }
        } else {
            /* отправляем на global root */
            MPI_Send(y_row, rows_per[coords[0]], MPI_DOUBLE, 0, 300 + coords[0], MPI_COMM_WORLD);
        }
    }
    double comm_end = MPI_Wtime(); double comm_local = comm_end - comm_start;

    MPI_Barrier(comm);
    double overall_end = MPI_Wtime();

    comm_local += comm_local1;
    double comp_max = 0.0, comm_max = 0.0;
    MPI_Reduce(&comp_local, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&comm_local, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank_world == 0) append_csv(csv_dir, prefix, "algo_block", procs, R, C, overall_end - overall_start, comp_max, comm_max);

    if (rank_world == 0 && R <= 5 && C <= 5) {
        double *A_ref = NULL, *v_ref = NULL;
        generate_mat_vec(R, C, fixed_seed, &A_ref, &v_ref);
        double *y_ref = malloc((size_t)R * sizeof(double));
        seq_matvec(R, C, A_ref, v_ref, y_ref);

        printf("\n[algo_block] Проверка на global root (R<=5 && C<=5)\n");
        print_matrix("A", R, C, A_ref);
        print_vector("v", C, v_ref);
        print_vector("y (алгоритм)", R, y_global ? y_global : y_ref);
        print_vector("y (реф)", R, y_ref);

        free(A_ref); free(v_ref); free(y_ref);
    }

    if (row_rank == 0 && y_row) free(y_row);
    if (y_global) free(y_global);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&cart);

    free(Ablock); free(vblock); free(y_partial);
    free(rows_per); free(rows_disp); free(cols_per); free(cols_disp);
}

/* ---------------- main ----------------
   Поддерживается список размеров: "R1xC1,R2xC2,..." или просто "N1,N2,..."
*/
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2 || !argv[1] || argv[1][0] == '\0') {
        if (rank == 0) fprintf(stderr, "Usage: %s <R1xC1,R2xC2,... | N1,N2,...> [prefix]\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    /* парсим список размеров */
    int *rows_list = NULL;
    int *cols_list = NULL;
    int cnt = parse_sizes_scan(argv[1], &rows_list, &cols_list);
    if (cnt <= 0) {
        if (rank == 0) fprintf(stderr, "No valid sizes parsed from '%s'\n", argv[1]);
        MPI_Finalize();
        return 1;
    }
    

    const char *prefix = (argc >= 3) ? argv[2] : "task2";
    const char *csv_dir = "./task2/data/csv";

    if (rank == 0) { ensure_dir("./task2"); ensure_dir("./task2/data"); ensure_dir(csv_dir); }
    MPI_Barrier(MPI_COMM_WORLD);

    unsigned int fixed_seed = (unsigned int)time(NULL); /* единый seed для всех алгоритмов */

    for (int i = 0; i < cnt; ++i) {
        int R = rows_list[i], C = cols_list[i];
        if (rank == 0) printf("Запускаем алгоритмы для R=%d C=%d\n", R, C);
        algo_row(R, C, csv_dir, prefix, MPI_COMM_WORLD, fixed_seed);
        MPI_Barrier(MPI_COMM_WORLD);
        algo_col(R, C, csv_dir, prefix, MPI_COMM_WORLD, fixed_seed);
        MPI_Barrier(MPI_COMM_WORLD);
        algo_block(R, C, csv_dir, prefix, MPI_COMM_WORLD, fixed_seed);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) printf("Готово R=%d C=%d\n", R, C);
    }

    free(rows_list); free(cols_list);
    MPI_Finalize();
    return 0;
}
