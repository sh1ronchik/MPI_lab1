/* task1.c
   MPI Monte Carlo оценка pi — три варианта коммуникации:
   A) blocking Send/Recv (root собирает)
   B) collective Reduce
   C) non-blocking Isend/Irecv

   Root пишет CSV-строки в ./task1/data/csv/<prefix>_algo{Isend_Irecv,Reduce,Send_Recv}.csv
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <sys/stat.h>
#include <errno.h>

/* --- Утилиты файловой системы --- */
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

/* Печать и запись в CSV */
static void report_result(const char *csv_dir, const char *prefix, const char *algo,
                          int procs, long long total_points,
                          double overall_time, double pi_est)
{
    /* печать в stdout */
    printf("%s: pi=%.12f, overall=%.6f\n", algo, pi_est, overall_time);

    /* формируем имя файла и дописываем строку */
    char fname[512];
    snprintf(fname, sizeof(fname), "%s/%s_%s.csv", csv_dir, prefix, algo);
    FILE *f = fopen(fname, "a");
    if (!f) {
        fprintf(stderr, "Не удалось открыть %s для записи\n", fname);
        return;
    }

    fprintf(f, "%d,%lld,%.9f,%.12f\n", procs, total_points, overall_time, pi_est);
    fclose(f);
}


/* Вычисляет локальное число попаданий */
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
    if (total_points <= 0) {
        if (my_rank == 0) {
            fprintf(stderr, "Error: total_points must be a positive integer (> 0), got %s\n", argv[1]);
        }
        MPI_Finalize();
        return 1;
    }
    const char *prefix = (argc >= 3) ? argv[2] : "task1";

    /* распределение точек между процессами */
    long long base = total_points / comm_sz;
    int rem = (int)(total_points % comm_sz);
    long long local_n = base + (my_rank < rem ? 1 : 0);

    /* уникальное семя на процесс */
    unsigned int seed_base = (unsigned int)time(NULL);
    unsigned int seed = seed_base ^ (unsigned int)(my_rank * 7919u);
    
    const char *csv_dir = "./task1/data/csv";
    if (my_rank == 0) {
        /* Создаём директории для CSV */
        ensure_dir_exists(csv_dir);

        printf("MPI Monte Carlo: processes=%d, total_points=%lld, base=%lld, rem=%d\n",
               comm_sz, total_points, base, rem);
    }

    /* Контейнеры для измерений */
    double overall_start, overall_end, overall_time;

    /* ---------- Algorithm A: blocking Send/Recv ---------- */
    unsigned int seedA = seed;
    
    MPI_Barrier(MPI_COMM_WORLD);
    overall_start = MPI_Wtime();

    long long local_hits = compute_local_hits(local_n, &seedA);

    long long total_hits_A = 0;
    if (my_rank == 0) {
        total_hits_A = local_hits;
        MPI_Status st;
        for (int src = 1; src < comm_sz; ++src) {
            long long recv_val = 0;
            MPI_Recv(&recv_val, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, 100, MPI_COMM_WORLD, &st);
            total_hits_A += recv_val;
        }
    } else {
        MPI_Send(&local_hits, 1, MPI_LONG_LONG, 0, 100, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    overall_end = MPI_Wtime();
    overall_time = overall_end - overall_start;

    if (my_rank == 0) {
        double pi_est = 4.0 * (double)total_hits_A / (double)total_points;
        report_result(csv_dir, prefix, "Send_Recv", comm_sz, total_points, overall_time, pi_est);
    }

    /* ---------- Algorithm B: MPI_Reduce ---------- */
    unsigned int seedB = seed;
    MPI_Barrier(MPI_COMM_WORLD);
    overall_start = MPI_Wtime();

    local_hits = compute_local_hits(local_n, &seedB);

    long long total_hits_B = 0;
    MPI_Reduce(&local_hits, &total_hits_B, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    overall_end = MPI_Wtime();
    overall_time = overall_end - overall_start;


    if (my_rank == 0) {
        double pi_est = 4.0 * (double)total_hits_B / (double)total_points;
        report_result(csv_dir, prefix, "Reduce", comm_sz, total_points, overall_time, pi_est);
    }

    /* ---------- Algorithm C: non-blocking Isend/Irecv ---------- */
    unsigned int seedC = seed;
    MPI_Barrier(MPI_COMM_WORLD);
    overall_start = MPI_Wtime();

    local_hits = compute_local_hits(local_n, &seedC);

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
            MPI_Irecv(&recv_buf[src-1], 1, MPI_LONG_LONG, MPI_ANY_SOURCE, 102, MPI_COMM_WORLD, &reqs[src-1]);
        }
        MPI_Waitall(comm_sz-1, reqs, MPI_STATUSES_IGNORE);
        for (int i = 0; i < comm_sz-1; ++i) total_hits_C += recv_buf[i];
        free(reqs); free(recv_buf);
    } else {
        MPI_Request req;
        MPI_Isend(&local_hits, 1, MPI_LONG_LONG, 0, 102, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    overall_end = MPI_Wtime();
    overall_time = overall_end - overall_start;


    if (my_rank == 0) {
        double pi_est = 4.0 * (double)total_hits_C / (double)total_points;
        report_result(csv_dir, prefix, "Isend_Irecv", comm_sz, total_points, overall_time, pi_est);
    }

    MPI_Finalize();
    return 0;
}
