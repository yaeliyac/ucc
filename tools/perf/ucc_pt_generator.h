/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_GENERATOR_H
#define UCC_PT_GENERATOR_H

#include <cstddef>
#include <vector>

class ucc_pt_generator_base {
        public:
    virtual bool has_next() = 0;
    virtual void next() = 0;
    virtual size_t get_count() = 0;
    virtual std::vector<size_t> get_count_array() = 0;
    virtual void reset() = 0;
    virtual ~ucc_pt_generator_base() {}
};

class ucc_pt_exponential_generator : public ucc_pt_generator_base {
private:
    size_t min_count;
    size_t max_count;
    size_t mult_factor;
    size_t current_count;
public:
    ucc_pt_exponential_generator(size_t min, size_t max, size_t factor) {
        min_count = min;
        max_count = max;
        mult_factor = factor;
        current_count = min;
    }

    bool has_next() {
        return current_count * mult_factor < max_count;
    }

    void next() {
        if (!has_next()) {
            return;
        }
        current_count *= mult_factor;
    }

    size_t get_count() {
        return current_count;
    }

    std::vector<size_t> get_count_array() {
        std::vector<size_t> result(1, current_count);
        return result;
    }
    void reset() {
        current_count = min_count;
    }
};

#endif

