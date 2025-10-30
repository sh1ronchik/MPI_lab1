/* task1.c
   MPI Monte Carlo оценка pi — три варианта коммуникации:
   A) blocking Send/Recv (root собирает)
   B) collective Reduce
   C) non-blocking Isend/Irecv

   Root пишет CSV-строки в ./task1/data/csv/<prefix>_algo{A,B,C}.csv
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <sys/stat.h>
#include <errno.h>

/* --- Утилиты --- */
static void ensure_dir_exists(const char *path) {
    struct stat st;
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0755) != 0 && errno != EEXIST) {
            fprintf(stderr, "Не удалось создать директорию %s (errno=%d)\n", path, errno);
        }
    }
}

/* Вычисляет локальное число попаданий используя rand_r (reentrant) */
static long long compute_local_hits(long long local_n, unsigned int *seed_p) {
    long long hits = 0;
    for (long long i = 0; i < local_n; ++i) {
        double x = (rand_r(seed_p) / (double)RAND_MAX) * 2.0 - 1.0;
        double y = (rand_r(seed_p) / (double)RAND_MAX) * 2.0 - 1.0;
        if (x*x + y*y <= 1.0) ++hits;
    }
    return hits;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int comm_sz, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc < 2) {
        if (my_rank == 0) fprintf(stderr, "Usage: %s <total_points> [out_prefix]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    long long total_points = atoll(argv[1]);
    if (total_points <= 0) total_points = 1000000LL;
    const char *prefix = (argc >= 3) ? argv[2] : "task1";

    /* Создаём директории для CSV (структура аналогична другим задачам) */
    const char *csv_dir = "./task1/data/csv";
    ensure_dir_exists(".");
    ensure_dir_exists("./task1");
    ensure_dir_exists("./task1/data");
    ensure_dir_exists(csv_dir);

    /* распределение точек между процессами */
    long long base = total_points / comm_sz;
    int rem = (int)(total_points % comm_sz);
    long long local_n = base + (my_rank < rem ? 1 : 0);

    /* уникальное семя на процесс */
    unsigned int seed_base = (unsigned int)time(NULL);
    unsigned int seed = seed_base ^ (unsigned int)(my_rank * 7919u);

    if (my_rank == 0) {
        printf("MPI Monte Carlo: processes=%d, total_points=%lld, base=%lld, rem=%d\n",
               comm_sz, total_points, base, rem);
    }

    /* Контейнеры для измерений */
    double comp_time_local = 0.0, comm_time_local = 0.0;
    double max_comp_time = 0.0, max_comm_time = 0.0;
    double overall_start, overall_end, overall_time;

    /* ---------- Algorithm A: blocking Send/Recv ---------- */
    MPI_Barrier(MPI_COMM_WORLD);
    overall_start = MPI_Wtime();

    double t0 = MPI_Wtime();
    unsigned int seedA = seed; /* копия семени для варианта A */
    long long local_hits = compute_local_hits(local_n, &seedA);
    double t1 = MPI_Wtime();
    comp_time_local = t1 - t0;

    double t_comm_s = MPI_Wtime();
    long long total_hits_A = 0;
    if (my_rank == 0) {
        total_hits_A = local_hits;
        for (int src = 1; src < comm_sz; ++src) {
            long long recv_val = 0;
            MPI_Recv(&recv_val, 1, MPI_LONG_LONG, src, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_hits_A += recv_val;
        }
    } else {
        MPI_Send(&local_hits, 1, MPI_LONG_LONG, 0, 100, MPI_COMM_WORLD);
    }
    double t_comm_e = MPI_Wtime();
    comm_time_local = t_comm_e - t_comm_s;

    MPI_Barrier(MPI_COMM_WORLD);
    overall_end = MPI_Wtime();
    overall_time = overall_end - overall_start;

    MPI_Reduce(&comp_time_local, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time_local, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        double pi_est = 4.0 * (double)total_hits_A / (double)total_points;
        printf("A Send/Recv: pi=%.12f, overall=%.6f, comp_max=%.6f, comm_max=%.6f\n",
               pi_est, overall_time, max_comp_time, max_comm_time);

        char fname[512];
        snprintf(fname, sizeof(fname), "%s/%s_algoA.csv", csv_dir, prefix);
        FILE *f = fopen(fname, "a");
        if (f) {
            /* столбцы: procs,total_points,overall_time,comp_max,comm_max,pi */
            fprintf(f, "%d,%lld,%.9f,%.9f,%.9f,%.12f\n",
                    comm_sz, total_points, overall_time, max_comp_time, max_comm_time, pi_est);
            fclose(f);
        } else {
            fprintf(stderr, "Не удалось открыть %s для записи\n", fname);
        }
    }

    /* ---------- Algorithm B: MPI_Reduce ---------- */
    MPI_Barrier(MPI_COMM_WORLD);
    overall_start = MPI_Wtime();

    t0 = MPI_Wtime();
    unsigned int seedB = seed ^ 0x9e3779b9u;
    local_hits = compute_local_hits(local_n, &seedB);
    t1 = MPI_Wtime();
    comp_time_local = t1 - t0;

    t_comm_s = MPI_Wtime();
    long long total_hits_B = 0;
    MPI_Reduce(&local_hits, &total_hits_B, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    t_comm_e = MPI_Wtime();
    comm_time_local = t_comm_e - t_comm_s;

    MPI_Barrier(MPI_COMM_WORLD);
    overall_end = MPI_Wtime();
    overall_time = overall_end - overall_start;

    MPI_Reduce(&comp_time_local, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time_local, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        double pi_est = 4.0 * (double)total_hits_B / (double)total_points;
        printf("B Reduce:    pi=%.12f, overall=%.6f, comp_max=%.6f, comm_max=%.6f\n",
               pi_est, overall_time, max_comp_time, max_comm_time);

        char fname[512];
        snprintf(fname, sizeof(fname), "%s/%s_algoB.csv", csv_dir, prefix);
        FILE *f = fopen(fname, "a");
        if (f) {
            fprintf(f, "%d,%lld,%.9f,%.9f,%.9f,%.12f\n",
                    comm_sz, total_points, overall_time, max_comp_time, max_comm_time, pi_est);
            fclose(f);
        } else {
            fprintf(stderr, "Не удалось открыть %s для записи\n", fname);
        }
    }

    /* ---------- Algorithm C: non-blocking Isend/Irecv ---------- */
    MPI_Barrier(MPI_COMM_WORLD);
    overall_start = MPI_Wtime();

    t0 = MPI_Wtime();
    unsigned int seedC = seed ^ 0x7f4a7c15u;
    local_hits = compute_local_hits(local_n, &seedC);
    t1 = MPI_Wtime();
    comp_time_local = t1 - t0;

    t_comm_s = MPI_Wtime();
    long long total_hits_C = 0;
    if (comm_sz == 1) {
        total_hits_C = local_hits;
    } else if (my_rank == 0) {
        total_hits_C = local_hits;
        long long *recv_buf = (long long*)malloc((comm_sz-1) * sizeof(long long));
        MPI_Request *reqs = (MPI_Request*)malloc((comm_sz-1) * sizeof(MPI_Request));
        if (!recv_buf || !reqs) {
            fprintf(stderr, "Root: malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int src = 1; src < comm_sz; ++src) {
            MPI_Irecv(&recv_buf[src-1], 1, MPI_LONG_LONG, src, 102, MPI_COMM_WORLD, &reqs[src-1]);
        }
        MPI_Waitall(comm_sz-1, reqs, MPI_STATUSES_IGNORE);
        for (int i = 0; i < comm_sz-1; ++i) total_hits_C += recv_buf[i];
        free(reqs); free(recv_buf);
    } else {
        MPI_Request req;
        MPI_Isend(&local_hits, 1, MPI_LONG_LONG, 0, 102, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    t_comm_e = MPI_Wtime();
    comm_time_local = t_comm_e - t_comm_s;

    MPI_Barrier(MPI_COMM_WORLD);
    overall_end = MPI_Wtime();
    overall_time = overall_end - overall_start;

    MPI_Reduce(&comp_time_local, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time_local, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        double pi_est = 4.0 * (double)total_hits_C / (double)total_points;
        printf("C Isend/Irecv: pi=%.12f, overall=%.6f, comp_max=%.6f, comm_max=%.6f\n",
               pi_est, overall_time, max_comp_time, max_comm_time);

        char fname[512];
        snprintf(fname, sizeof(fname), "%s/%s_algoC.csv", csv_dir, prefix);
        FILE *f = fopen(fname, "a");
        if (f) {
            fprintf(f, "%d,%lld,%.9f,%.9f,%.9f,%.12f\n",
                    comm_sz, total_points, overall_time, max_comp_time, max_comm_time, pi_est);
            fclose(f);
        } else {
            fprintf(stderr, "Не удалось открыть %s для записи\n", fname);
        }
    }

    MPI_Finalize();
    return 0;
}
