/* task3.c
   Реализация алгоритма Кэннона для матричного умножения C = A * B
   Поддержка нескольких размеров N через командную строку: "N1,N2,..."
   Запись результатов в CSV: ./task3/data/csv/<prefix>_cannon.csv
   Формат CSV: procs,N,overall
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
void ensure_dir_exists(const char *path) {
    char tmp[512];
    strncpy(tmp, path, sizeof(tmp));
    tmp[sizeof(tmp)-1] = '\0';

    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

/* --- запись CSV --- */
static void append_csv(const char *dir, const char *prefix, int procs, int N,
                       double overall) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s_cannon.csv", dir, prefix);
    FILE *f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "Не могу открыть %s для дозаписи\n", path);
        return;
    }
    fprintf(f, "%d,%d,%.9f\n", procs, N, overall);
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

/* безопасный malloc: выделяет хотя бы 1 байт, чтобы не передавать NULL в MPI */
static void *safe_malloc(size_t bytes) {
    if (bytes == 0) bytes = 1;
    return malloc(bytes);
}

/* умножение локальных блоков: C_local += A_local * B_local
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

/* упаковать блок (br x bc) из большой матрицы A (размер R x C) в буфер (row-major) */
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

/* распаковать буфер блока в большую матрицу в позиции r0,c0 */
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

/*
  run_cannon_once
*/
static double run_cannon_once(int N, const char *prefix, const char *csv_dir) {
    int world_rank, comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* проверка: p должно быть точным квадратом */
    int q = (int)floor(sqrt((double)comm_sz) + 0.5);
    if (q * q != comm_sz) {
        if (world_rank == 0) fprintf(stderr, "Error: number of processes (p=%d) is not a perfect square. Cannon requires p = q^2.\n", comm_sz);
        return -1.0;
    }

    /* Определяем размер блока nb (дополняем matrix до q*nb) */
    int nb = (N + q - 1) / q;         /* ceil(N/q) */
    int Npad = nb * q;                /* размер после padding */

    /* создаём декартов коммуникатор q x q (periodic для удобства сдвигов) */
    int dims[2] = {q, q};
    int periods[2] = {1, 1};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);

    int cart_rank;
    MPI_Comm_rank(cart, &cart_rank);
    int coords[2];
    MPI_Cart_coords(cart, cart_rank, 2, coords);
    int my_row = coords[0], my_col = coords[1];

    /* локальные блоки: nb x nb */
    double *Ablock = (double*) safe_malloc((size_t)nb * nb * sizeof(double));
    double *Bblock = (double*) safe_malloc((size_t)nb * nb * sizeof(double));
    double *Cblock = (double*) safe_malloc((size_t)nb * nb * sizeof(double));
    memset(Cblock, 0, (size_t)nb * nb * sizeof(double));

    double *Afull = NULL, *Bfull = NULL;
    if (world_rank == 0) {
        /* генерация A и B (seed по времени, как в оригинале) */
        unsigned int seed = (unsigned int)time(NULL);
        srand((unsigned)seed);
        Afull = (double*) safe_malloc((size_t)Npad * Npad * sizeof(double));
        Bfull = (double*) safe_malloc((size_t)Npad * Npad * sizeof(double));
        /* инициализация полного заполненны массив нулями */
        for (int i = 0; i < Npad*Npad; ++i) { Afull[i] = 0.0; Bfull[i] = 0.0; }
        /* заполние верхнего левого столбеца N x N случайными маленькими числами */
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                Afull[i*(size_t)Npad + j] = (double)(rand() % 10);
                Bfull[i*(size_t)Npad + j] = (double)(rand() % 10);
            }
        if (N <= 5){
            printf("\nМатрица A (N=%d, padded Npad=%d):\n", N, Npad);
            print_matrix("A", N, N, Afull);
            printf("\nМатрица B (N=%d, padded Npad=%d):\n", N, Npad);
            print_matrix("B", N, N, Bfull);
        }
    }


    /* синхронизация перед началом замера */
    MPI_Barrier(MPI_COMM_WORLD);
    double overall_start = MPI_Wtime();

    /* определяем соседей для сдвигов */
    int left, right, up, down;
    MPI_Cart_shift(cart, 1, -1, &right, &left);
    MPI_Cart_shift(cart, 0, -1, &down, &up); 

    /* root упаковывает блоки и рассылает каждому процессу */
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
                pack_block(Afull, N, N, r0, c0, nb, nb, bufA);
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

    /* начальные сдвиги: A влево на my_row, B вверх на my_col */
    for (int s = 0; s < my_row; ++s) {
        MPI_Sendrecv_replace(Ablock, nb*nb, MPI_DOUBLE,
                             left,  10,
                             right, 10,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    for (int s = 0; s < my_col; ++s) {
        MPI_Sendrecv_replace(Bblock, nb*nb, MPI_DOUBLE,
                             up,   20,
                             down, 20,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* вычислительная часть: q итераций */
    int steps = q;
    for (int iter = 0; iter < steps; ++iter) {
        /* локальное умножение: Cblock += Ablock * Bblock */
        local_matmul_add(nb, Ablock, Bblock, Cblock);

        /* сдвиги: A влево на 1, B вверх на 1 */
        MPI_Sendrecv_replace(Ablock, nb*nb, MPI_DOUBLE, left,  30, right, 30, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(Bblock, nb*nb, MPI_DOUBLE, up,    40, down,  40, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* сбор блоков C на root */
    if (world_rank == 0) {
        double *Cfull = (double*) malloc((size_t)Npad * Npad * sizeof(double));
        /* root копирует свой блок */
        int my_pr = my_row, my_pc = my_col;
        int r0 = my_pr * nb, c0 = my_pc * nb;
        unpack_block(Cfull, N, N, r0, c0, nb, nb, Cblock);

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

    /* общий конец времени */
    MPI_Barrier(MPI_COMM_WORLD);
    double overall_end = MPI_Wtime();

    /* очистка */
    free(Ablock); free(Bblock); free(Cblock);
    if (Afull) free(Afull);
    if (Bfull) free(Bfull);
    MPI_Comm_free(&cart);

    /* root сохраняет результат в CSV и печатает */
    if (world_rank == 0) {
        append_csv(csv_dir, prefix, comm_sz, N, overall_end - overall_start);
        printf("Cannon: procs=%d N=%d overall=%.6f\n",
               comm_sz, N, overall_end - overall_start);
    }

    return overall_end - overall_start;
}

/* --- парсинг списка размеров "N1,N2,..." --- */
static char * my_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *r = (char*) malloc(len + 1);
    if (!r) return NULL;
    memcpy(r, s, len + 1);
    return r;
}

static int parse_sizes_list(const char *s, int **out) {
    if (!s) { *out = NULL; return 0; }
    char *tmp = my_strdup(s);
    if (!tmp) { *out = NULL; return 0; }
    int cap = 8, cnt = 0;
    int *arr = malloc(cap * sizeof(int));
    if (!arr) { free(tmp); *out = NULL; return 0; }

    char *tok = strtok(tmp, ",");
    while (tok) {
        while (*tok == ' ' || *tok == '\t') ++tok;
        char *end = tok + strlen(tok) - 1;
        while (end > tok && (*end == ' ' || *end == '\t')) { *end = '\0'; --end; }

        if (*tok) {
            char *xpos = strchr(tok, 'x');
            if (!xpos) xpos = strchr(tok, 'X');
            char save = 0;
            if (xpos) { save = *xpos; *xpos = '\0'; }
            long v = strtol(tok, NULL, 10);
            if (xpos) *xpos = save;
            if (v > 0) {
                if (cnt >= cap) {
                    int nc = cap * 2;
                    int *tmpa = realloc(arr, nc * sizeof(int));
                    if (!tmpa) { free(arr); free(tmp); *out = NULL; return 0; }
                    arr = tmpa; cap = nc;
                }
                arr[cnt++] = (int)v;
            }
        }
        tok = strtok(NULL, ",");
    }

    free(tmp);
    *out = arr;
    return cnt;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc < 2) {
        if (my_rank == 0) fprintf(stderr, "Usage: %s <N1,N2,...> [prefix]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* парсим список размеров */
    int *sizes = NULL;
    int cnt = parse_sizes_list(argv[1], &sizes);
    if (cnt <= 0) {
        if (my_rank == 0) fprintf(stderr, "No valid sizes parsed from '%s'\n", argv[1]);
        MPI_Finalize();
        return 1;
    }

    const char *prefix = (argc >= 3) ? argv[2] : "task3";
    const char *csv_dir = "./task3/data/csv";

    if (my_rank == 0) { ensure_dir_exists(csv_dir); }
    MPI_Barrier(MPI_COMM_WORLD);

    /* для каждого размера вызываем run_cannon_once */
    for (int i = 0; i < cnt; ++i) {
        int N = sizes[i];
        if (my_rank == 0) printf("Запускаем Cannon для N=%d\n", N);
        run_cannon_once(N, prefix, csv_dir);
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0) printf("Готово N=%d\n", N);
    }

    free(sizes);
    MPI_Finalize();
    return 0;
}
