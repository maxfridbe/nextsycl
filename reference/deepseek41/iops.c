// iops.c: random reads of BS bytes (default 4096; BS=512 the engine's reads) of a file with O_DIRECT from N threads (each one read in flight: queue depth N) - the
// Engram table's access pattern (a row is a 4 KiB page read for its 256 bytes, another for its 8-byte scale).
//   cc -O2 -pthread iops.c -o iops && ./iops FILE SECONDS QD [QD ...]
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int fd;
static off_t pages;
static long bs = 4096;
static double secs;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct { long n; double lat; unsigned seed; } job;

static void* run(void* a) {
    job* j = a;
    void* buf;
    if (posix_memalign(&buf, 4096, 4096)) return NULL;
    double end = now() + secs;
    while (now() < end) {
        uint64_t r = ((uint64_t) rand_r(&j->seed) << 31) ^ rand_r(&j->seed);
        double t0 = now();
        if (pread(fd, buf, bs, (off_t) (r % pages) * bs) != bs) { perror("pread"); break; }
        j->lat += now() - t0;
        j->n++;
    }
    free(buf);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "iops FILE SECONDS QD...\n"); return 2; }
    fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (fd < 0) { perror(argv[1]); return 1; }
    struct stat st;
    fstat(fd, &st);
    if (getenv("BS")) bs = atol(getenv("BS"));
    pages = st.st_size / bs;
    secs = atof(argv[2]);
    for (int a = 3; a < argc; ++a) {
        int qd = atoi(argv[a]);
        pthread_t th[1024];
        job jobs[1024] = {0};
        for (int i = 0; i < qd; ++i) { jobs[i].seed = 1234u + 77u * i + 1000u * a; pthread_create(&th[i], NULL, run, &jobs[i]); }
        long n = 0;
        double lat = 0;
        for (int i = 0; i < qd; ++i) { pthread_join(th[i], NULL); n += jobs[i].n; lat += jobs[i].lat; }
        printf("QD %4d: %8.0f IOPS (%6.1f MB/s), mean latency %6.1f us\n", qd, n / secs, n * (double) bs / secs / 1e6, lat / n * 1e6);
        fflush(stdout);
    }
    return 0;
}
