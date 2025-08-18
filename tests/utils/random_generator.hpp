#pragma once

#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>
#include <parlay/utilities.h>
using namespace std;

class rn_gen {
   public:
    inline static vector<rn_gen> rand_gens;

    static int64_t randint64_rand() {
        uint64_t v = rand();
        v = (v << 31) + rand();
        v = (v << 2) + (rand() & 3);
        if (v >= (1ull << 63)) {
            return v - (1ull << 63);
        } else {
            return v - (1ull << 63);
        }
    }

    static void init() {
        srand(time(NULL));
        // srand(137);

        int thread_num = parlay::num_workers();
        cout<<"rand init: tn = " << thread_num << endl;
        for (int i = 0; i < thread_num; i++) {
            rand_gens.emplace_back(randint64_rand());
        }
    }

    static int64_t parallel_rand() {
        int tid = parlay::worker_id();
        return rand_gens[tid].next();
    }

    rn_gen(size_t seed) : state(seed){};
    rn_gen() : state(0){};
    int64_t random(int64_t i) { return parlay::hash64(this->state + i); }
    int64_t proceed(int64_t i) { this->state += i; return this->state; }
    int64_t next() { return parlay::hash64(this->state++); }

   private:
    size_t state = 0;
    
};