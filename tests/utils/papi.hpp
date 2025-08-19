#ifndef PAPI_UTIL_IMPL_H
#define PAPI_UTIL_IMPL_H
#include <iostream>
#include <string>
#include <cassert>
#include <thread>
#include <iomanip>

// #include "parlay/scheduler.h"
// #include "parlay/parallel.h"
#ifdef USE_PAPI
#include <papi.h>
#endif

#ifndef MAX_THREADS_POW2
    #define MAX_THREADS_POW2 128 // MUST BE A POWER OF TWO, since this is used for some bitwise operations
#endif
#ifndef LOGICAL_PROCESSORS
    #define LOGICAL_PROCESSORS MAX_THREADS_POW2
#endif

#ifndef SOFTWARE_BARRIER
#   define SOFTWARE_BARRIER asm volatile("": : :"memory")
#endif

// the following definition is only used to pad data to avoid false sharing.
// although the number of words per cache line is actually 8, we inflate this
// figure to counteract the effects of prefetching multiple adjacent cache lines.
#define PREFETCH_SIZE_WORDS 16
#define PREFETCH_SIZE_BYTES 128
#define BYTES_IN_CACHE_LINE 64

#define CAT2(x, y) x##y
#define CAT(x, y) CAT2(x, y)

#define PAD64 volatile char CAT(___padding, __COUNTER__)[64]
#define PAD volatile char CAT(___padding, __COUNTER__)[128]

#define CASB __sync_bool_compare_and_swap
#define CASV __sync_val_compare_and_swap
#define FAA __sync_fetch_and_add

#ifdef USE_PAPI
bool counter_turnon = false;
bool *thread_local_counter_on = nullptr;
#endif

int all_cpu_counters[] = {
#ifdef USE_PAPI
    //    PAPI_L1_DCM, // works on amd17h
    // PAPI_L2_DCM,  // works on amd17h also
    PAPI_L3_TCM,  // does not work on amd17h
    // PAPI_TOT_CYC,
    PAPI_TOT_INS,
//    PAPI_RES_STL,
//    PAPI_TLB_DM,
#endif
};

std::string all_cpu_counters_strings[] = {
#ifdef USE_PAPI
    //    "PAPI_L1_DCM",
    // "PAPI_L2_TCM",
    "PAPI_L3_TCM",
    // "PAPI_TOT_CYC",
    "PAPI_TOT_INS",
//    "PAPI_RES_STL",
//    "PAPI_TLB_DM",
#endif
};
#ifdef USE_PAPI
const int nall_cpu_counters =
    sizeof(all_cpu_counters) / sizeof(all_cpu_counters[0]);
#endif

#ifdef USE_PAPI
int event_sets[MAX_THREADS_POW2];
int64_t counter_values[nall_cpu_counters];
#endif

char *cpu_counter(int c) {
#ifdef USE_PAPI
    char counter[PAPI_MAX_STR_LEN];

    PAPI_event_code_to_name(c, counter);
    return strdup(counter);
#endif
    return NULL;
}

void papi_init_program(int num_workers) {
#ifdef USE_PAPI
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "Error: Failed to init PAPI\n");
        exit(2);
    }

    if (PAPI_thread_init(pthread_self) != PAPI_OK) {
        fprintf(stderr, "PAPI_ERROR: failed papi_thread_init()\n");
        exit(2);
    }
    bool* counters = new bool[num_workers];
    for (int i = 0; i < num_workers; i ++) {
        counters[i] = false;
    }
    SOFTWARE_BARRIER;
    thread_local_counter_on = counters;
    // thread_local_counter_on = new bool[parlay::num_workers()];
#endif
}

void papi_deinit_program() {
#ifdef USE_PAPI
    bool* counters = thread_local_counter_on;
    thread_local_counter_on = nullptr;
    SOFTWARE_BARRIER;
    delete [] counters;
    PAPI_shutdown();
#endif
}

void papi_reset_counters() {
#ifdef USE_PAPI
    for (int i = 0; i < nall_cpu_counters; ++i) {
        counter_values[i] = 0;
    }
#endif
}

bool papi_turn_counters(bool on) {
#ifdef USE_PAPI
    bool ret = (counter_turnon != on);
    // assert(counter_turnon != on);
    counter_turnon = on;
    return ret;
#endif
    return false;
}

void papi_wait_counters(bool on, int num_workers) {
#ifdef USE_PAPI
    // int tcnt = parlay::num_workers();
    int tcnt = num_workers;
    // printf("Start waiting for %d threads.\n", tcnt);
    // fflush(stdout);
    for (int i = 0; i < tcnt; i ++) {
        while (thread_local_counter_on[i] != on) {
            // std::this_thread::sleep_for(std::chrono::nanoseconds(2 * tcnt * 100));
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            // printf("Waiting for %d\n", i);
        }
    }
#endif
}

void papi_start_counters(int id) {
#ifdef USE_PAPI
    {
        event_sets[id] = PAPI_NULL;
        int *event_set = &event_sets[id];
        int result;
        if ((result = PAPI_register_thread()) != PAPI_OK) {
            fprintf(stderr,
                    "PAPI_ERROR: thread %d cannot register.\n", id);
            exit(2);
        }
        if ((result = PAPI_create_eventset(event_set)) != PAPI_OK) {
            fprintf(stderr,
                    "PAPI_ERROR: thread %d cannot create event set: %s\n", id,
                    PAPI_strerror(result));
            exit(2);
        }

        // std::cout<<id<<' '<<*event_set<<std::endl;
        // PAPI_add_named_event(*event_set, "PAPI_L3_TCM");
        for (int i = 0; i < nall_cpu_counters; i++) {
            // std::cout<<id<<' '<<i<<' '<<nall_cpu_counters<<std::endl;
            int c = all_cpu_counters[i];
            if ((result = PAPI_query_event(c)) != PAPI_OK) {
                std::cout << "warning: PAPI event " << cpu_counter(c)
                          << " could not be successfully queried: "
                          << PAPI_strerror(result) << std::endl;
                continue;
            }
            if ((result = PAPI_add_event(*event_set, c)) != PAPI_OK) {
                if (result != PAPI_ECNFLCT) {
                    fprintf(
                        stderr,
                        "PAPI ERROR: thread %d unable to add event %s: %s\n",
                        id, cpu_counter(c), PAPI_strerror(result));
                    exit(2);
                }
                /* Not enough hardware resources, disable this counter and move
                 * on.
                 */
                std::cout << "warning: could not add PAPI event "
                          << cpu_counter(c) << "... disabled it." << std::endl;
                all_cpu_counters[i] = PAPI_END + 1;
            }
        }
    }
    {
        int *event_set = &event_sets[id];
        int result;
        if ((result = PAPI_start(*event_set)) != PAPI_OK) {
            fprintf(stderr, "PAPI ERROR: thread %d unable to start counters: %s\n",
                    id, PAPI_strerror(result));
            std::cout << "relevant event_set is for tid=" << id << " and has value "
                    << (*event_set) << std::endl;
            exit(2);
        }
    }
#endif
}

void papi_stop_counters(int id) {
#ifdef USE_PAPI
    int *event_set = &event_sets[id];
    long long values[nall_cpu_counters];
    for (int i = 0; i < nall_cpu_counters; i++) values[i] = 0;

    int r;

    /* Get cycles from hardware to account for time stolen by co-scheduled
     * threads. */
    if ((r = PAPI_stop(*event_set, values)) != PAPI_OK) {
        fprintf(stderr, "PAPI ERROR: thread %d unable to stop counters: %s\n",
                id, PAPI_strerror(r));
        exit(2);
    }
    int j = 0;
    for (int i = 0; i < nall_cpu_counters; i++) {
        int c = all_cpu_counters[i];
        if (PAPI_query_event(c) != PAPI_OK) continue;
        __sync_fetch_and_add(&counter_values[j], values[j]);
        j++;
    }
    if ((r = PAPI_cleanup_eventset(*event_set)) != PAPI_OK) {
        fprintf(stderr,
                "PAPI ERROR: thread %d unable to cleanup event set: %s\n", id,
                PAPI_strerror(r));
        exit(2);
    }
    if ((r = PAPI_destroy_eventset(event_set)) != PAPI_OK) {
        fprintf(stderr,
                "PAPI ERROR: thread %d unable to destroy event set: %s\n", id,
                PAPI_strerror(r));
        exit(2);
    }
    if ((r = PAPI_unregister_thread()) != PAPI_OK) {
        fprintf(stderr,
                "PAPI ERROR: thread %d unable to unregister thread: %s\n", id,
                PAPI_strerror(r));
        exit(2);
    }
#endif
}
void papi_print_counters(long long num_operations) {
#ifdef USE_PAPI
    int i, j;
    for (i = j = 0; i < nall_cpu_counters; i++) {
        int c = all_cpu_counters[i];
        if (PAPI_query_event(c) != PAPI_OK) {
            std::cout << all_cpu_counters_strings[i] << "=-1" << std::endl;
            continue;
        }
        std::cout << std::setprecision(5);
        std::cout << all_cpu_counters_strings[i] << "="
                  << ((double)counter_values[j] / num_operations) << std::endl;
        // printf("%s=%.3f\n", cpu_counter(c),
        // (double)counter_values[j]/num_operations);
        j++;
    }
#endif
}

bool papi_check_counters(int id) {
#ifdef USE_PAPI
    if (thread_local_counter_on == nullptr) {
        return false;
    }
    // int id = parlay::worker_id();
    // assert(id < parlay::num_workers());
    if ((!thread_local_counter_on[id]) && counter_turnon) {
        // printf("Thread %d turn on counters\n", id);
        papi_start_counters(id);
        thread_local_counter_on[id] = true;
        return true;
    } else if ((thread_local_counter_on[id]) && !counter_turnon) {
        // printf("Thread %d turn off counters\n", id);
        // printf("tlco=%d\n", thread_local_counter_on[id]);
        papi_stop_counters(id);
        thread_local_counter_on[id] = false;
        // printf("tlco=%d\n", thread_local_counter_on[id]);
        return true;
    }
#endif
    return false;
}

#endif
