#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <iostream>
#include <cstdint>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstddef>

#define USE_PAPI (1)

#ifdef USE_PAPI
#include "utils/papi.hpp"
#endif

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "testFramework.h"
#include "utils/random_generator.hpp"

using namespace std;

/* Argv for main function */
int64_t total_insert_size;
int NR_DIMENSION;
int test_batch_size;
int test_round;
int test_type; /* 1: Point search; 2: Box range count; 3: Box fetch; 4: kNN */
int expected_box_size;
std::string file_name;

void host_parse_arguments(int argc, char *argv[]) {
    file_name          = (argc >= 2  ?      argv[1]    : "uniform");
    NR_DIMENSION       = (argc >= 3  ? stoi(argv[2] )  : 3       );
    total_insert_size  = (argc >= 4  ? stoi(argv[3] )  : 500000  );
    test_type          = (argc >= 5  ? stoi(argv[4] )  : 0       );
    test_batch_size    = (argc >= 6  ? stoi(argv[5] )  : 10000   );
    test_round         = (argc >= 7  ? stoi(argv[6] )  : 2       );
    expected_box_size  = (argc >= 8  ? stoi(argv[7] )  : 100     );
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char *argv[]) {
    using vectorT = PointType<coord, 3>;
    using point = PointType<coord, 3>;
    using tree = ParallelKDtree<vectorT>;
    using box = typename tree::box;
    using node = typename tree::node;
    using nn_pair = std::pair<std::reference_wrapper<point>, coord>;
    using points = typename tree::points;

    printf("------------------- Start ---------------------\n");
    host_parse_arguments(argc, argv);
    srand(0);
    rn_gen::init();
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    double avg_time = 0.0;

    printf("------------- Data Structure Init ------------\n");
    tree pkd;

    parlay::sequence<vectorT> vectors_from_file(1);
    parlay::sequence<vectorT> vectors_to_insert(1);
    size_t varden_counter = 0;
    coord COORD_MAX = INT64_MAX;

    if(file_name == "uniform") {
        printf("Uniform\n");
        vectors_to_insert.resize(total_insert_size);
        parlay::parallel_for(0, vectors_to_insert.size(), [&](size_t i) {
            vectors_to_insert[i].pnt[0] = abs(rn_gen::parallel_rand());
            vectors_to_insert[i].pnt[1] = abs(rn_gen::parallel_rand());
            vectors_to_insert[i].pnt[2] = abs(rn_gen::parallel_rand());
        });
    }
    else {
        printf("File: %s\n", file_name.c_str());
        read_points(file_name.c_str(), vectors_from_file, 100);
        vectors_to_insert = parlay::tabulate(total_insert_size, [&](size_t i) { return vectors_from_file[i]; });
        if(test_type == 2 || test_type == 3) {
            for(int i = 0; i < NR_DIMENSION; i++) {
                coord gap = parlay::reduce(parlay::delayed_tabulate(vectors_from_file.size(), [&](size_t j) {
                    return vectors_from_file[j].pnt[i];
                }), parlay::maximum<coord>()) - parlay::reduce(parlay::delayed_tabulate(vectors_from_file.size(), [&](size_t j) {
                    return vectors_from_file[j].pnt[i];
                }), parlay::minimum<coord>());
                if(i == 0) COORD_MAX = gap;
                else if(gap > COORD_MAX) COORD_MAX = gap;
            }
        }
    }
    printf("------------- Finish Data Init ------------\n");

    buildTree<vectorT>(NR_DIMENSION, vectors_to_insert, 1, pkd);
    printf("------------- Finish Tree Build ------------\n");

#ifdef USE_PAPI
    papi_init_program(parlay::num_workers());
#endif

    if(test_type == 1) {
        printf("------------- Insert ------------\n");
        parlay::sequence<vectorT> vec_to_search(test_batch_size);
        for(int i = 0, offset = total_insert_size; i < test_round; i++, offset += test_batch_size) {
            printf("Round: %d; Time: ", i);
            if(file_name == "uniform") {
                parlay::parallel_for(0, test_batch_size, [&](size_t j) {
                    vec_to_search[j].pnt[0] = abs(rn_gen::parallel_rand());
                    vec_to_search[j].pnt[1] = abs(rn_gen::parallel_rand());
                    vec_to_search[j].pnt[2] = abs(rn_gen::parallel_rand());
                });
            }
            else {
                vec_to_search = parlay::tabulate(test_batch_size, [&](size_t j) {
                    return vectors_from_file[offset + j];
                });
            }
#ifdef USE_PAPI
            papi_reset_counters();
            papi_turn_counters(true);
            parlay::parallel_for(0, parlay::num_workers(), [&](size_t j) { papi_check_counters(j); });
            papi_wait_counters(true, parlay::num_workers());
#endif
            start_time = std::chrono::high_resolution_clock::now();
            pkd.batchInsert(parlay::make_slice(vec_to_search), NR_DIMENSION);
            end_time = std::chrono::high_resolution_clock::now();
#ifdef USE_PAPI
            papi_turn_counters(false);
            parlay::parallel_for(0, parlay::num_workers(), [&](size_t j) { papi_check_counters(j); });
            papi_wait_counters(false, parlay::num_workers());
#endif
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
            printf("%f\n", d.count());
            avg_time += d.count();
        }
    }
    else if(test_type == 2 || test_type == 3) {
        printf("------------- Box ------------\n");
        double box_edge_size = COORD_MAX / pow(total_insert_size / expected_box_size, 1.0 / NR_DIMENSION) / 2.0;
        parlay::sequence<box> boxes(test_batch_size);
        for(int i = 0, offset = total_insert_size; i < test_round; i++, offset += test_batch_size) {
            printf("Round: %d; Time: ", i);
            if(file_name == "uniform") {
                parlay::parallel_for(0, test_batch_size, [&](size_t j) {
                    boxes[j].first.pnt[0] = abs(rn_gen::parallel_rand());
                    boxes[j].first.pnt[1] = abs(rn_gen::parallel_rand());
                    boxes[j].first.pnt[2] = abs(rn_gen::parallel_rand());
                    boxes[j].second.pnt[0] = boxes[j].first.pnt[0] + box_edge_size * 2;
                    boxes[j].second.pnt[1] = boxes[j].first.pnt[1] + box_edge_size * 2;
                    boxes[j].second.pnt[2] = boxes[j].first.pnt[2] + box_edge_size * 2;
                });
            }
            else {
                boxes = parlay::tabulate(test_batch_size, [&](size_t j) {
                    vectorT max_edge = vectors_from_file[offset + j];
                    max_edge.pnt[0] += box_edge_size * 2;
                    max_edge.pnt[1] += box_edge_size * 2;
                    max_edge.pnt[2] += box_edge_size * 2;
                    return std::make_pair(vectors_from_file[offset + j], max_edge);
                });
            }
#ifdef USE_PAPI
            papi_reset_counters();
            papi_turn_counters(true);
            parlay::parallel_for(0, parlay::num_workers(), [&](size_t j) { papi_check_counters(j); });
            papi_wait_counters(true, parlay::num_workers());
#endif
            start_time = std::chrono::high_resolution_clock::now();
            parlay::parallel_for(0, test_batch_size, [&](size_t j) {
                size_t visLeafNum, visInterNum;
                size_t cnt = pkd.range_count(boxes[j], visLeafNum, visInterNum);
                if(test_type == 3) {
                    parlay::sequence<vectorT> res(cnt);
                    pkd.range_query_serial(boxes[j], res);
                }
            });
            end_time = std::chrono::high_resolution_clock::now();
#ifdef USE_PAPI
            papi_turn_counters(false);
            parlay::parallel_for(0, parlay::num_workers(), [&](size_t j) { papi_check_counters(j); });
            papi_wait_counters(false, parlay::num_workers());
#endif
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
            printf("%f\n", d.count());
            avg_time += d.count();
        }
    }
    else if(test_type == 4) {
        printf("------------- kNN ------------\n");
        parlay::sequence<vectorT> vec_to_search(test_batch_size);
        for(int i = 0, offset = total_insert_size; i < test_round; i++, offset += test_batch_size) {
            printf("Round: %d; Time: ", i);
            if(file_name == "uniform") {
                parlay::parallel_for(0, test_batch_size, [&](size_t j) {
                    vec_to_search[j].pnt[0] = abs(rn_gen::parallel_rand());
                    vec_to_search[j].pnt[1] = abs(rn_gen::parallel_rand());
                    vec_to_search[j].pnt[2] = abs(rn_gen::parallel_rand());
                });
            }
            else {
                vec_to_search = parlay::tabulate(test_batch_size, [&](size_t j) {
                    return vectors_from_file[offset + j];
                });
            }
            node* KDParallelRoot = pkd.get_root();
            auto bx = pkd.get_root_box();

            points wp = points::uninitialized(test_batch_size);
            parlay::copy(vec_to_search, wp);
            parlay::sequence<nn_pair> Out(expected_box_size * test_batch_size, nn_pair(std::ref(wp[0]), 0));
            parlay::sequence<kBoundedQueue<point, nn_pair>> bq =
                parlay::sequence<kBoundedQueue<point, nn_pair>>::uninitialized(test_batch_size);
            parlay::parallel_for(0, test_batch_size, [&](size_t i) {
                bq[i].resize(Out.cut(i * expected_box_size, i * expected_box_size + expected_box_size));
            });
#ifdef USE_PAPI
            papi_reset_counters();
            papi_turn_counters(true);
            parlay::parallel_for(0, parlay::num_workers(), [&](size_t j) { papi_check_counters(j); });
            papi_wait_counters(true, parlay::num_workers());
#endif
            start_time = std::chrono::high_resolution_clock::now();
            parlay::parallel_for(0, test_batch_size, [&](size_t j) {
                size_t visNodeNum = 0;
                pkd.k_nearest(KDParallelRoot, vec_to_search[j], NR_DIMENSION, bq[j], bx, visNodeNum);
            });
            end_time = std::chrono::high_resolution_clock::now();
#ifdef USE_PAPI
            papi_turn_counters(false);
            parlay::parallel_for(0, parlay::num_workers(), [&](size_t j) { papi_check_counters(j); });
            papi_wait_counters(false, parlay::num_workers());
#endif
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
            printf("%f\n", d.count());
            avg_time += d.count();
        }
    }
#ifdef USE_PAPI
    papi_print_counters(1);
#endif
    avg_time /= test_round;
    printf("Average Time: %f\n", avg_time);
    return 0;
}
