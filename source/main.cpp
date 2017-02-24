////////////////////////////////////////////////////////////////

#include <array>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>

////////////////////////////////////////////////////////////////

//#define DEBUG
//#define __SSE4_1__

#ifdef _MSC_VER
#define scanf scanf_s
#endif

////////////////////////////////////////////////////////////////

typedef std::chrono::high_resolution_clock MyClock;
typedef std::chrono::duration<double> MySeconds;
typedef std::chrono::duration<double, std::milli> MyMilliseconds;
typedef std::chrono::duration<double, std::micro> MyMicroseconds;

////////////////////////////////////////////////////////////////

// Intrinsics
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#elif defined(__SSSE3__)
#include <tmmintrin.h>
#elif defined(__SSE3__)
#include <pmmintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

////////////////////////////////////////////////////////////////
// from vector classes

/****************************  vectori128.h   *******************************
* Author:        Agner Fog
* Date created:  2012-05-30
* Last modified: 2017-02-19
* Version:       1.27
* Project:       vector classes
* Description:
* Header file defining integer vector classes as interface to intrinsic
* functions in x86 microprocessors with SSE2 and later instruction sets
* up to AVX.
*
* (c) Copyright 2012-2017 GNU General Public License http://www.gnu.org/licenses
*****************************************************************************/

// Define bit-scan-reverse function. Gives index to highest set bit = floor(log2(a))
#if defined (__GNUC__) || defined(__clang__)
static inline uint32_t bit_scan_reverse(uint32_t a) __attribute__((pure));
static inline uint32_t bit_scan_reverse(uint32_t a) {
    uint32_t r;
    __asm("bsrl %1, %0" : "=r"(r) : "r"(a) : );
    return r;
}
#else
static inline uint32_t bit_scan_reverse(uint32_t a) {
    unsigned long r;
    _BitScanReverse(&r, a);                      // defined in intrin.h for MS and Intel compilers
    return r;
}
#endif

// encapsulate parameters for fast division on vector of 8 16-bit unsigned integers
class Divisor_us {
protected:
    __m128i multiplier;                                    // multiplier used in fast division
    __m128i shift1;                                        // shift count 1 used in fast division
    __m128i shift2;                                        // shift count 2 used in fast division
public:
    Divisor_us() {};                                       // Default constructor
    Divisor_us(uint16_t d) {                               // Constructor with divisor
        set(d);
    }
    Divisor_us(uint16_t m, int s1, int s2) {               // Constructor with precalculated multiplier and shifts
        multiplier = _mm_set1_epi16(m);
        shift1 = _mm_setr_epi32(s1, 0, 0, 0);
        shift2 = _mm_setr_epi32(s2, 0, 0, 0);
    }
    void set(uint16_t d) {                                 // Set or change divisor, calculate parameters
        uint16_t L, L2, sh1, sh2, m;
        switch (d) {
        case 0:
            std::cerr << "divisor \'d\' cannot be 0!\n";   // provoke error for d = 0
            m = sh1 = sh2 = 0;
            break;
        case 1:
            m = 1; sh1 = sh2 = 0;                          // parameters for d = 1
            break;
        case 2:
            m = 1; sh1 = 1; sh2 = 0;                       // parameters for d = 2
            break;
        default:                                           // general case for d > 2
            L = (uint16_t)bit_scan_reverse(d - 1) + 1;        // ceil(log2(d))
            L2 = uint16_t(1 << L);                         // 2^L, overflow to 0 if L = 16
            m = 1 + uint16_t((uint32_t(L2 - d) << 16) / d); // multiplier
            sh1 = 1;  sh2 = L - 1;                         // shift counts
        }
        multiplier = _mm_set1_epi16(m);
        shift1 = _mm_setr_epi32(sh1, 0, 0, 0);
        shift2 = _mm_setr_epi32(sh2, 0, 0, 0);
    }
    __m128i getm() const {                                 // get multiplier
        return multiplier;
    }
    __m128i gets1() const {                                // get shift count 1
        return shift1;
    }
    __m128i gets2() const {                                // get shift count 2
        return shift2;
    }
};

// vector of 8 16-bit unsigned integers
static inline __m128i operator/(const __m128i &a, const Divisor_us &d) {
    __m128i t1 = _mm_mulhi_epu16(a, d.getm()); // multiply high unsigned words
    __m128i t2 = _mm_sub_epi16(a, t1); // subtract
    __m128i t3 = _mm_srl_epi16(t2, d.gets1()); // shift right logical
    __m128i t4 = _mm_add_epi16(t1, t3); // add
    return _mm_srl_epi16(t4, d.gets2()); // shift right logical 
}

////////////////////////////////////////////////////////////////

// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

////////////////////////////////////////////////////////////////

template <int _ID>
struct Bailian
{
    void operator()() noexcept {}
};

template <>
struct Bailian<2774>
{
    typedef uint16_t T;

    static const int Nmax = 10000;
    static const int Kmax = 10000;
    static const T Lmax = 10000;

    static void main() noexcept
    {
        // Result
        int sLengthResult = 0;

        // Input
#ifdef DEBUG
        unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
        std::default_random_engine generator(seed);
        std::uniform_int_distribution<uint16_t> distributionLength(2000, Lmax);

        int N = 2000;
        int K = 10000;
        std::cout << "N=" << N << ", K=" << K << std::endl;
#else
        int N, K;
        scanf("%d %d", &N, &K);
#endif

        std::vector<T> wLengths(N); // wood lengths

        for (auto &l : wLengths)
        {
#ifdef DEBUG
            l = distributionLength(generator);
#else
            int lTemp;
            scanf("%d", &lTemp);
            l = lTemp;
#endif
        }

#ifdef DEBUG
        MyClock::time_point t1 = MyClock::now();
        const int loops = 10000;

        for (int loop = 0; loop < loops; ++loop)
#endif
        do
        {
            // Get rough maximum segment-length
            int wLengthSum = 0; // sum of wood lengths
            int wLengthMin = Lmax; // minimum wood length

            {
                const T *wLengthsPtr = wLengths.data();
                const T *wLengthsPtrUpper = wLengthsPtr + N;

#if defined(__SSE4_1__)
                const __m128i zero_si128 = _mm_setzero_si128();
                __m128i wLengthSum_v = _mm_setzero_si128(); // epi32
                __m128i wLengthMin_v = _mm_set1_epi16(static_cast<uint16_t>(wLengthMin)); // epu16

                static const ptrdiff_t simd_step = 8;
                const ptrdiff_t simd_residue = N % simd_step;
                const ptrdiff_t simd_N = N - simd_residue;

                for (const T *upper = wLengthsPtr + simd_N; wLengthsPtr < upper; wLengthsPtr += simd_step)
                {
                    const __m128i wLength_v = _mm_load_si128(reinterpret_cast<const __m128i *>(wLengthsPtr));
                    wLengthSum_v = _mm_add_epi32(wLengthSum_v, _mm_cvtepu16_epi32(_mm_hadd_epi16(wLength_v, wLength_v)));
                    wLengthMin_v = _mm_min_epu16(wLengthMin_v, wLength_v);
                }

                alignas(16) int32_t temp_epi32[4];
                wLengthSum_v = _mm_hadd_epi32(wLengthSum_v, wLengthSum_v);
                _mm_store_si128(reinterpret_cast<__m128i *>(temp_epi32), wLengthSum_v);
                wLengthSum = temp_epi32[0] + temp_epi32[1];

                alignas(16) uint16_t temp_epu16[8];
                wLengthMin_v = _mm_min_epu16(_mm_unpacklo_epi16(wLengthMin_v, zero_si128), _mm_unpackhi_epi16(wLengthMin_v, zero_si128));
                wLengthMin_v = _mm_min_epu16(_mm_unpacklo_epi16(wLengthMin_v, zero_si128), _mm_unpackhi_epi16(wLengthMin_v, zero_si128));
                _mm_store_si128(reinterpret_cast<__m128i *>(temp_epu16), wLengthMin_v);
                wLengthMin = std::min(temp_epu16[0], temp_epu16[4]);
#endif
                for (; wLengthsPtr < wLengthsPtrUpper; ++wLengthsPtr)
                {
                    wLengthSum += *wLengthsPtr;
                    if (*wLengthsPtr < wLengthMin) wLengthMin = *wLengthsPtr;
                }
            }

            int sLengthMax = std::min(wLengthMin, wLengthSum / K);

            if (sLengthMax < 2)
            {
                sLengthResult = sLengthMax;
                break;
            }

            // Get exact maximum segment-length
            for (int sLength = sLengthMax; sLength > 0; --sLength)
            {
                int segments = 0;

                const T *wLengthsPtr = wLengths.data();
                const T *wLengthsPtrUpper = wLengthsPtr + N;

#if defined(__SSE4_1__)
                Divisor_us sLength_divisor(sLength);
                __m128i segments_v = _mm_setzero_si128();

                static const ptrdiff_t simd_step = 8;
                const ptrdiff_t simd_residue = N % simd_step;
                const ptrdiff_t simd_N = N - simd_residue;

                for (const T *upper = wLengthsPtr + simd_N; wLengthsPtr < upper; wLengthsPtr += simd_step)
                {
                    const __m128i wLength_v = _mm_load_si128(reinterpret_cast<const __m128i *>(wLengthsPtr));
                    segments_v = _mm_add_epi16(segments_v, wLength_v / sLength_divisor);
                }
#endif
                for (; wLengthsPtr < wLengthsPtrUpper; ++wLengthsPtr)
                {
                    segments += *wLengthsPtr / sLength;
                }

                if (segments >= K)
                {
                    sLengthResult = sLength;
                    break;
                }
            }
        } while (0);
#ifdef DEBUG
        MySeconds time_span = std::chrono::duration_cast<MySeconds>(MyClock::now() - t1);

        std::cout << ": It took "
            << std::chrono::duration_cast<MyMicroseconds>(time_span).count() / loops
            << " microseconds.\n";
#endif

        printf("%d\n", sLengthResult);
    }
};

////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    Bailian<2774>::main();

#ifdef DEBUG
    getchar();
#endif

    return 0;
}

////////////////////////////////////////////////////////////////
