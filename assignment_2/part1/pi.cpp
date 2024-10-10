#include <iostream>
#include <pthread.h>
#include <random>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>

struct ThreadData {
    long long tosses_per_thread;
    double number_in_circle;
    int thread_id;
};

pthread_mutex_t mutex;

// SIMD-aware LCG random number generator function
inline __m256i simd_random(__m256i seed) {
    __m256i a = _mm256_set1_epi32(1664525);
    __m256i c = _mm256_set1_epi32(1013904223);
    __m256i m = _mm256_set1_epi32(0x7FFFFFFF);
    
    // (a * seed + c) % m
    seed = _mm256_add_epi32(_mm256_mullo_epi32(seed, a), c);
    seed = _mm256_and_si256(seed, m);

    return seed;
}

void *Pthread_monte_carlo(void *args) {
    ThreadData *data = (ThreadData *)args;
    long long tosses_per_thread = data->tosses_per_thread;
    int thread_id = data->thread_id;
    double local_count_in_circle = 0;

    // 根據 thread_id 和 time 生成 seed
    unsigned int base_seed = time(NULL) + thread_id;
    __m256i seed;
    unsigned int seeds[8];

    // init seed[8]
    for (int i = 0; i < 8; ++i) {
        seeds[i] = base_seed + i * 12345;
    }
    seed = _mm256_set_epi32(seeds[7], seeds[6], seeds[5], seeds[4], seeds[3], seeds[2], seeds[1], seeds[0]);

    __m256 x, y, distance_squared, in_circle;
    __m256 bound = _mm256_set1_ps((float)0xFFFF);
    __m256d count1 = _mm256_set1_pd(0x0);
    __m256d count2 = _mm256_set1_pd(0x0);

    for (long long toss = 0; toss < tosses_per_thread; toss += 8) {
        // Generate random numbers in parallel
        seed = simd_random(seed);

        // x = ((rand_val & 0xFFFF) / (double)0xFFFF) * 2 - 1;
        // x 取 seed 的低 16 bits，並從 int 轉為 float
        // 一次處理8個 32bits int 並將它們轉換成8個single precision floating point
        x = _mm256_cvtepi32_ps(_mm256_and_si256(seed, _mm256_set1_epi32(0xFFFF)));
        x = _mm256_div_ps(x, bound);
        x = _mm256_mul_ps(x, _mm256_set1_ps(2.0));
        x = _mm256_sub_ps(x, _mm256_set1_ps(1.0));

        // y = ((rand_val >> 16) / (double)0xFFFF) * 2 - 1;
        // y 取 seed 的高 16 bits
        y = _mm256_cvtepi32_ps(_mm256_srai_epi32(seed, 16));
        y = _mm256_div_ps(y, bound);
        y = _mm256_mul_ps(y, _mm256_set1_ps(2.0));
        y = _mm256_sub_ps(y, _mm256_set1_ps(1.0));

        // distance_squared = x * x + y * y
        distance_squared = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));

        // If distance_squared <= 1, in_circle mask 為 1
        in_circle = _mm256_cmp_ps(distance_squared, _mm256_set1_ps((float)1.0), _CMP_LE_OQ);
        in_circle = _mm256_and_ps(in_circle, _mm256_set1_ps((float)1.0));

        // Count the number of points inside the circle
        // 將 256bits 的低 128bits 取出來，並轉換為 128bits 的 single precision floating point (__m128)
        __m128 low = _mm256_castps256_ps128(in_circle);
        __m256d low_d = _mm256_cvtps_pd(low);
        // 將 256bits 的高 128bits 取出來，並轉換為 128bits 的 single precision floating point (__m128)
        __m128 high = _mm256_extractf128_ps(in_circle, 1);
        __m256d high_d = _mm256_cvtps_pd(high);

        count1 = _mm256_add_pd(count1, low_d);
        count2 = _mm256_add_pd(count2, high_d);
    }

    double results[4];
    count1 = _mm256_add_pd(count1, count2);
    _mm256_store_pd(results, count1);
    local_count_in_circle += results[0] + results[1] + results[2] + results[3];

    pthread_mutex_lock(&mutex);
    data->number_in_circle += local_count_in_circle;
    pthread_mutex_unlock(&mutex);

    return NULL;
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number_of_threads> <number_of_tosses>" << std::endl;
        return 1;
    }

    long long number_of_threads = atoi(argv[1]);
    long long number_of_tosses = atoll(argv[2]);

    pthread_t thread[number_of_threads];
    ThreadData thread_data[number_of_threads];
    double total_in_circle = 0;

    pthread_mutex_init(&mutex, NULL);

    long long tosses_per_thread = number_of_tosses / number_of_threads;

    // Create threads
    for (long long i = 0; i < number_of_threads; ++i) {
        thread_data[i].tosses_per_thread = tosses_per_thread;
        thread_data[i].number_in_circle = 0;
        thread_data[i].thread_id = i;

        if (i > 0) pthread_create(&thread[i], NULL, Pthread_monte_carlo, (void *)&thread_data[i]);
    }

    // Main thread does its part
    Pthread_monte_carlo(&thread_data[0]);

    // Join threads
    for (long long i = 1; i < number_of_threads; ++i) {
        pthread_join(thread[i], NULL);
    }

    // Collect results
    for (long long i = 0; i < number_of_threads; ++i) {
        total_in_circle += thread_data[i].number_in_circle;
    }

    double pi_estimate = 4 * total_in_circle / number_of_tosses;
    std::cout << pi_estimate << std::endl;

    pthread_mutex_destroy(&mutex);
    return 0;
}
