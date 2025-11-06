/* task3.c
   Реализация алгоритма Кэннона для матричного умножения C = A * B
   Поддержка:
     - квадратные матрицы N x N (в коде матрицу можно дополнить до размера кратного q)
     - p = q*q (p должно быть точным квадратом)
   Запись результатов в CSV: ./task3/data/csv/<prefix>_cannon.csv
   Формат CSV: procs,N,overall,comp_max,comm_max
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>

/* --- утилиты файловой системы --- */
static void ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0755) != 0 && errno != EEXIST) {
            fprintf(stderr, "Не удалось создать каталог %s (errno=%d)\n", path, errno);
        }
    }
}

/* --- запись CSV --- */
static void append_csv(const char *dir, const char *prefix, int procs, int N,
                       double overall, double comp_max, double comm_max) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s_cannon.csv", dir, prefix);
    FILE *f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "Не могу открыть %s для дозаписи\n", path);
        return;
    }
    fprintf(f, "%d,%d,%.9f,%.9f,%.9f\n", procs, N, overall, comp_max, comm_max);
    fclose(f);
}

/* --- печать матрицы/вектора (для маленьких размеров) --- */
static void print_matrix(const char *name, int R, int C, const double *A) {
    printf("%s (%dx%d):\n", name, R, C);
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) printf("%8.3f ", A[i*(size_t)C + j]);
        printf("\n");
    }
}
static void print_vector(const char *name, int n, const double *v) {
    printf("%s:", name);
    for (int i = 0; i < n; ++i) printf(" %8.3f", v[i]);
    printf("\n");
}

/* safe malloc */
static void *safe_malloc(size_t bytes) {
    if (bytes == 0) bytes = 1;
    return malloc(bytes);
}

/* multiply local blocks: C_local += A_local * B_local
   A_local (nb x nb), B_local (nb x nb), C_local (nb x nb)
*/
static void local_matmul_add(int nb, const double *A, const double *B, double *C) {
    for (int i = 0; i < nb; ++i) {
        for (int k = 0; k < nb; ++k) {
            double a = A[i*(size_t)nb + k];
            const double *brow = B + (size_t)k * nb;
            double *crow = C + (size_t)i * nb;
            for (int j = 0; j < nb; ++j) {
                crow[j] += a * brow[j];
            }
        }
    }
}

/* pack block (br x bc) from big matrix A (of size R x C) into buffer (row-major) */
static void pack_block(const double *A, int R, int C,
                       int r0, int c0, int br, int bc, double *buf) {
    for (int i = 0; i < br; ++i) {
        int ai = (r0 + i);
        for (int j = 0; j < bc; ++j) {
            int aj = (c0 + j);
            if (ai < R && aj < C) buf[i*(size_t)bc + j] = A[ai*(size_t)C + aj];
            else buf[i*(size_t)bc + j] = 0.0; /* padding */
        }
    }
}

/* unpack block buffer into big matrix at position r0,c0 */
static void unpack_block(double *A, int R, int C,
                         int r0, int c0, int br, int bc, const double *buf) {
    for (int i = 0; i < br; ++i) {
        int ai = r0 + i;
        if (ai >= R) break;
        for (int j = 0; j < bc; ++j) {
            int aj = c0 + j;
            if (aj >= C) break;
            A[ai*(size_t)C + aj] = buf[i*(size_t)bc + j];
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (world_rank == 0) fprintf(stderr, "Usage: %s <N> [prefix]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        if (world_rank == 0) fprintf(stderr, "Error: N must be positive integer\n");
        MPI_Finalize(); return 1;
    }
    const char *prefix = (argc >= 3) ? argv[2] : "task3";
    const char *csv_dir = "./task3/data/csv";
    if (world_rank == 0) { ensure_dir("./task3"); ensure_dir("./task3/data"); ensure_dir(csv_dir); }
    MPI_Barrier(MPI_COMM_WORLD);

    /* проверка: p должно быть точным квадратом */
    int q = (int)floor(sqrt((double)world_size) + 0.5);
    if (q * q != world_size) {
        if (world_rank == 0) fprintf(stderr, "Error: number of processes (p=%d) is not a perfect square. Cannon requires p = q^2.\n", world_size);
        MPI_Finalize();
        return 1;
    }

    /* Определяем размер блока nb (дополняем matrix до q*nb) */
    int nb = (N + q - 1) / q;         /* ceil(N/q) */
    int Npad = nb * q;                /* размер после padding */

    /* создаём декартов коммуникатор q x q */
    int dims[2] = {q, q};
    int periods[2] = {1, 1}; /* циклическая топология (для удобных сдвигов) */
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);

    int cart_rank;
    MPI_Comm_rank(cart, &cart_rank);
    int coords[2];
    MPI_Cart_coords(cart, cart_rank, 2, coords);
    int my_row = coords[0], my_col = coords[1];

    /* локальные блоки: каждое поле блока имеет размер nb x nb (row-major) */
    double *Ablock = (double*) safe_malloc((size_t)nb * nb * sizeof(double));
    double *Bblock = (double*) safe_malloc((size_t)nb * nb * sizeof(double));
    double *Cblock = (double*) safe_malloc((size_t)nb * nb * sizeof(double));
    memset(Cblock, 0, (size_t)nb * nb * sizeof(double));

    double *Afull = NULL, *Bfull = NULL;
    if (world_rank == 0) {
        /* generate A and B (fixed seed for reproducibility) */
        unsigned int seed = (unsigned int)time(NULL);
        srand((unsigned)seed);
        Afull = (double*) safe_malloc((size_t)Npad * Npad * sizeof(double));
        Bfull = (double*) safe_malloc((size_t)Npad * Npad * sizeof(double));
        /* initialize full padded array with zeros */
        for (int i = 0; i < Npad*Npad; ++i) { Afull[i] = 0.0; Bfull[i] = 0.0; }
        /* fill top-left N x N with random small numbers */
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                Afull[i*(size_t)Npad + j] = (double)(rand() % 10);
                Bfull[i*(size_t)Npad + j] = (double)(rand() % 10);
            }
    }

    /* синхронизируем и замеряем общее время */
    MPI_Barrier(MPI_COMM_WORLD);
    double overall_start = MPI_Wtime();

    /* соседи (для сдвигов влево/вправо и вверх/вниз) */
    int left, right, up, down;
    MPI_Cart_shift(cart, 1, -1, &right, &left); /* shift along column dimension: +1 moves to right; here we want left/right */
    MPI_Cart_shift(cart, 0, -1, &down, &up);    /* shift along row dimension: +1 moves down; here up/down */

    /* --- root упаковывает блоки и рассылает каждому процессу --- */
    if (world_rank == 0) {
        for (int pr = 0; pr < q; ++pr) {
            for (int pc = 0; pc < q; ++pc) {
                int dest_coords[2] = {pr, pc};
                int dest_rank;
                MPI_Cart_rank(cart, dest_coords, &dest_rank);
                int r0 = pr * nb;
                int c0 = pc * nb;
                double *bufA = (double*) malloc((size_t)nb * nb * sizeof(double));
                double *bufB = (double*) malloc((size_t)nb * nb * sizeof(double));
                pack_block(Afull, N, N, r0, c0, nb, nb, bufA); /* note: Afull is Npad x Npad but pack_block checks bounds */
                pack_block(Bfull, N, N, r0, c0, nb, nb, bufB);
                if (dest_rank == cart_rank) {
                    memcpy(Ablock, bufA, (size_t)nb * nb * sizeof(double));
                    memcpy(Bblock, bufB, (size_t)nb * nb * sizeof(double));
                } else {
                    MPI_Send(bufA, nb*nb, MPI_DOUBLE, dest_rank, 1000 + pr*q + pc, MPI_COMM_WORLD);
                    MPI_Send(bufB, nb*nb, MPI_DOUBLE, dest_rank, 2000 + pr*q + pc, MPI_COMM_WORLD);
                }
                free(bufA); free(bufB);
            }
        }
    } else {
        MPI_Status st;
        MPI_Recv(Ablock, nb*nb, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
        MPI_Recv(Bblock, nb*nb, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
    }

    /* --- Инициализационный (начальный) сдвиг:
       Для процесса с coords (i,j):
         - сдвигаем A влево на i позиций;
         - сдвигаем B вверх на j позиций.
       Реализуем как i (или j) раз последовательных сдвигов на один шаг.
       Для циклического сдвига по строке/столбцу используем MPI_Sendrecv_replace.
    */
    double comm_start_init = MPI_Wtime();
    /* shift A left by my_row times (i.e. shift along columns) */
    for (int s = 0; s < my_row; ++s) {
        MPI_Sendrecv_replace(Ablock, nb*nb, MPI_DOUBLE,
                             left,  10,  /* send to left */
                             right, 10,  /* recv from right */
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    /* shift B up by my_col times (i.e. shift along rows) */
    for (int s = 0; s < my_col; ++s) {
        MPI_Sendrecv_replace(Bblock, nb*nb, MPI_DOUBLE,
                             up,   20, /* send up */
                             down, 20, /* recv from down */
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    double comm_end_init = MPI_Wtime();
    double comm_local = comm_end_init - comm_start_init;

    /* вычислительная часть: q итераций */
    int steps = q;
    double comp_local = 0.0;
    double comm_local_steps = 0.0;
    for (int iter = 0; iter < steps; ++iter) {
        /* локальное умножение: Cblock += Ablock * Bblock */
        double t0 = MPI_Wtime();
        local_matmul_add(nb, Ablock, Bblock, Cblock);
        double t1 = MPI_Wtime();
        comp_local += t1 - t0;

        /* сдвиги: A left by 1, B up by 1 */
        double t2 = MPI_Wtime();
        MPI_Sendrecv_replace(Ablock, nb*nb, MPI_DOUBLE, left,  30, right, 30, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(Bblock, nb*nb, MPI_DOUBLE, up,    40, down,  40, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double t3 = MPI_Wtime();
        comm_local_steps += (t3 - t2);
    }
    comm_local += comm_local_steps;

    /* сбор блоков C на root */
    double comm_start_gather = MPI_Wtime();
    if (world_rank == 0) {
        double *Cfull = (double*) malloc((size_t)Npad * Npad * sizeof(double));
        /* root копирует свой блок */
        int my_pr = my_row, my_pc = my_col;
        int r0 = my_pr * nb, c0 = my_pc * nb;
        unpack_block(Cfull, N, N, r0, c0, nb, nb, Cblock);

        /* receive others */
        for (int pr = 0; pr < q; ++pr) {
            for (int pc = 0; pc < q; ++pc) {
                int src_coords[2] = {pr, pc};
                int src_rank; MPI_Cart_rank(cart, src_coords, &src_rank);
                if (src_rank == world_rank) continue;
                double *tmp = (double*) malloc((size_t)nb*nb*sizeof(double));
                MPI_Recv(tmp, nb*nb, MPI_DOUBLE, src_rank, 500 + pr*q + pc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int rr = pr * nb, cc = pc * nb;
                unpack_block(Cfull, N, N, rr, cc, nb, nb, tmp);
                free(tmp);
            }
        }

        /* для отладки: печатаем маленькие матрицы */
        if (N <= 8) {
            printf("\n[rank 0] Результат C (N=%d, padded Npad=%d):\n", N, Npad);
            print_matrix("C", N, N, Cfull);
        }

        free(Cfull);
    } else {
        MPI_Send(Cblock, nb*nb, MPI_DOUBLE, 0, 500 + my_row*q + my_col, MPI_COMM_WORLD);
    }
    double comm_end_gather = MPI_Wtime();
    comm_local += (comm_end_gather - comm_start_gather);

    /* общий конец времени */
    MPI_Barrier(MPI_COMM_WORLD);
    double overall_end = MPI_Wtime();

    /* редуцируем максимумы по процессам */
    double comp_max = 0.0, comm_max = 0.0;
    MPI_Reduce(&comp_local, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_local, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        append_csv(csv_dir, prefix, world_size, N, overall_end - overall_start, comp_max, comm_max);
        printf("Cannon: procs=%d N=%d overall=%.6f comp_max=%.6f comm_max=%.6f\n",
               world_size, N, overall_end - overall_start, comp_max, comm_max);
    }

    /* cleanup */
    free(Ablock); free(Bblock); free(Cblock);
    if (Afull) free(Afull);
    if (Bfull) free(Bfull);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
