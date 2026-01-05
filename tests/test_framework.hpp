#pragma once

#include <string>
#include <functional>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cstdint>
#include <cmath>

extern void register_test(std::string name, std::function<void()> func);

#define ONYX_TEST(name) \
    static void test_##name(); \
    static int dummy_##name = (register_test(#name, test_##name), 0); \
    static void test_##name()

#define ONYX_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            throw std::runtime_error("Assertion failed: " #cond); \
        } \
    } while (0)

#define ONYX_ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "Assertion failed: " << #a << " == " << #b << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            throw std::runtime_error("Assertion failed: " #a " == " #b); \
        } \
    } while (0)
