////////////////////////////////////////////////////////////////

#if defined (__GNUC__)
#pragma GCC optimize (2)
#pragma GCC diagnostic error "-fopenmp -lpthread"
#endif

#include <array>
#include <vector>
#include <forward_list>
#include <string>
#include <cstring>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>

////////////////////////////////////////////////////////////////

//#define DEBUG
#define __SSE2__
//#define __SSE4_1__

#ifdef _MSC_VER
#define scanf scanf_s
#endif

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
// Type definitions

typedef std::chrono::high_resolution_clock MyClock;
typedef std::chrono::duration<double> MySeconds;
typedef std::chrono::duration<double, std::milli> MyMilliseconds;
typedef std::chrono::duration<double, std::micro> MyMicroseconds;

////////////////////////////////////////////////////////////////
// Helper classes

class MyTimer
{
public:
    typedef MyTimer _Myt;
    typedef std::chrono::high_resolution_clock Clock;
    typedef Clock::time_point TimePoint;
    typedef std::chrono::duration<double> Seconds;
    typedef std::chrono::duration<double, std::milli> Milliseconds;
    typedef std::chrono::duration<double, std::micro> Microseconds;
    typedef std::chrono::duration<double, std::nano> Nanoseconds;

private:
    TimePoint t_start, t_end;

public:
    MyTimer()
        : t_start(Clock::now()), t_end(TimePoint())
    {}

    TimePoint Start() noexcept
    {
        t_end = TimePoint();
        t_start = Clock::now();
        return t_start;
    }

    TimePoint End() noexcept
    {
        t_end = Clock::now();
        return t_end;
    }

    template <typename _Duration>
    _Duration Duration() noexcept
    {
        if (t_end == TimePoint()) End();
        return std::chrono::duration_cast<_Duration>(t_end - t_start);
    }

    template <typename _Duration>
    _Myt &Print(const std::string &title = "Timer", double division = 1) noexcept
    {
        auto time_span = std::chrono::duration_cast<_Duration>(Duration<_Duration>()).count();
        if (division != 1) time_span /= division;
        std::cout << title << ": It took " << time_span << " " << getUnit<_Duration>() << ".\n";
        return *this;
    }

private:
    template <typename _Duration>
    std::string getUnit() noexcept
    {
        std::string unit;
        if (std::is_same<_Duration, Seconds>::value)
            return "seconds";
        else if (std::is_same<_Duration, Milliseconds>::value)
            return "milliseconds";
        else if (std::is_same<_Duration, Microseconds>::value)
            return "microseconds";
        else if (std::is_same<_Duration, Nanoseconds>::value)
            return "nanoseconds";
        else
            return "<unknown unit>";
    }
};

////////////////////////////////////////////////////////////////
// Helper functions

template <typename _Ty>
static inline _Ty DivideRoundUp(const _Ty &a, const _Ty &b)
{
    static_assert(std::is_integral<_Ty>::value, "DivideRoundUp: only integer input is accepted");
    return (a + b - 1) / b;
}

////////////////////////////////////////////////////////////////

template <int _ID>
struct Bailian
{
    static void main() noexcept {}
};

// http://bailian.openjudge.cn/practice/2774/
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
        MyTimer timer;
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

                constexpr const ptrdiff_t simd_step = 8;
                const ptrdiff_t simd_N = N - N % simd_step;

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

                constexpr const ptrdiff_t simd_step = 8;
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
        timer.Print<MyTimer::Microseconds>("1", loops);
#endif

        printf("%d", sLengthResult);
    }
};

// http://bailian.openjudge.cn/practice/2813/
template <>
struct Bailian<2813>
{
    typedef uint32_t T;
    static const int MaxIter = 0x10;

    static void main() noexcept
    {
        // Input
        int n;
        scanf("%d", &n);

        const int height = n;
        const int width = n;
        const int padding = 1;
        const int simd_step = static_cast<int>(sizeof(__m128i) / sizeof(T));
        const int simd_width = DivideRoundUp(width, simd_step) * simd_step;
        const int stride = simd_width + padding * 2;
        const int size = (n + padding * 2) * stride + padding;
        std::vector<T> wallData(size, UINT32_MAX); // padded wall
        auto *wallDataBegin = wallData.data() + stride + 1; // pointer of wall's origin

        for (int j = 0; j < height; ++j)
        {
            auto *datap = wallDataBegin + j * stride;

            for (int i = 0; i < width; ++i)
            {
                while (int c = getchar())
                {
                    if (c == 'w')
                    {
                        *datap = 0;
                        break;
                    }
                    else if (c == 'y')
                    {
                        *datap = UINT32_MAX;
                        break;
                    }
                }
            }
        }

        // Recursion implementation
        int steps = implement1(0, wallData, height, width, stride, simd_width, simd_step);

        if (steps > MaxIter) printf("inf\n");
        else printf("%d\n", steps);
    }

protected:
    // Start from the Largest possible steps
    // Randomly create drawing patterns and verify if they meet the requirement
    // O(Choose(n*n, n*n/2)) = O((n*n)! / (n*n/2)!^2)
    static int implement2(const std::vector<T> &wallData, int height, int width, int stride)
    {
        const int number = height * width;

    }

    // Recursive implementation of exhaustive search
    // Not applicable, and can go repeated steps on the same bricks, which should not happen.
    // O((n*n) ^ MaxIter)
    static int implement1(int recursionTimes, const std::vector<T> &lastWallData,
        int height, int width, int stride, int simd_width, int simd_step)
    {
        // Prepare copyed data
        auto wallData = lastWallData;
        auto *wallDataBegin = wallData.data() + stride + 1;

        // Set outer border to UINT32_MAX
        // (for those should always be UINT32_MAX, but could possibly be inverted)
        { // j = -1
            auto *datap = wallData.data();
            for (const auto *upper = datap + width + 2; datap < upper; ++datap)
            {
                *datap = UINT32_MAX;
            }
        }
        for (int j = 0; j < height; ++j)
        {
            auto *datap = wallData.data() + j * stride;
            datap[0] = UINT32_MAX;
            datap[width + 1] = UINT32_MAX;
        }
        { // j = width
            auto *datap = wallData.data() + (width + 1) * stride;
            for (const auto *upper = datap + width + 2; datap < upper; ++datap)
            {
                *datap = UINT32_MAX;
            }
        }

        // Determine if the requirement is met
        T sign = UINT32_MAX;
#if defined(__SSE2__)
        __m128i sign_v = _mm_set1_epi32(sign);
#endif

        for (int j = 0; j < height; ++j)
        {
            const auto *srcp = wallDataBegin + j * stride;

#if defined(__SSE2__)
            for (const auto *upper = srcp + simd_width; srcp < upper; srcp += simd_step)
            {
                sign_v = _mm_and_si128(sign_v, _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp)));
            }
#endif
            for (const auto *upper = srcp + width; srcp < upper; ++srcp)
            {
                sign &= *srcp;
            }
        }

#if defined(__SSE2__)
        alignas(sizeof(sign_v)) decltype(sign) sign_va[sizeof(sign_v) / sizeof(sign)];
        _mm_store_si128(reinterpret_cast<__m128i *>(sign_va), sign_v);
        sign &= sign_va[0] & sign_va[1] & sign_va[2] & sign_va[3];
#endif

        if (sign) return recursionTimes; // terminate recursion when all the bricks are yellow

        // Terminate recursion when exceeding iteration limitation
        if (++recursionTimes > MaxIter) return INT32_MAX;

        // Apply erosion morphological filter
        std::vector<T> wallEroded(wallData.size());
        auto *wallErodedBegin = wallEroded.data() + stride + 1;

        for (int j = 0; j < height; ++j)
        {
            const auto *srcp = wallDataBegin + j * stride;
            auto *dstp = wallErodedBegin + j * stride;

#if defined(__SSE2__)
            for (const auto *upper = dstp + simd_width; dstp < upper; dstp += simd_step, srcp += simd_step)
            {
                const __m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp - stride));
                const __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp - 1));
                const __m128i s2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp));
                const __m128i s3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp + 1));
                const __m128i s4 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp + stride));

                const __m128i d = _mm_and_si128(s2, _mm_and_si128(_mm_and_si128(s0, s4), _mm_and_si128(s1, s3)));
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dstp), d);
            }
#endif
            for (const auto *upper = dstp + width; dstp < upper; ++dstp, ++srcp)
            {
                *dstp = srcp[0] & (srcp[-stride] & srcp[stride]) & (srcp[-1] & srcp[1]);
            }
        }

        // Recursive searches on possible positions
        constexpr const T inv_mask = UINT32_MAX;
        const __m128i inv_mask_v = _mm_setr_epi32(inv_mask, inv_mask, inv_mask, 0);
        int minRecursionTimes = INT32_MAX;

        for (int j = 0; j < height; ++j)
        {
            const auto *erodep = wallErodedBegin + j * stride;
            auto *data1p = wallDataBegin + j * stride - 1;
            auto *data0p = data1p - stride + 1;
            auto *data2p = data1p + stride + 1;

            for (const auto *upper = erodep + width; erodep < upper && *erodep == 0;
                ++erodep, ++data0p, ++data1p, ++data2p)
            {
                *data0p ^= inv_mask;
                *data2p ^= inv_mask;
                _mm_storeu_si128(reinterpret_cast<__m128i *>(data1p),
                    _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(data1p)), inv_mask_v));

                const int recurs = implement1(recursionTimes, wallData, height, width, stride, simd_width, simd_step);
                if (recurs < minRecursionTimes) minRecursionTimes = recurs;

                *data0p ^= inv_mask;
                *data2p ^= inv_mask;
                _mm_storeu_si128(reinterpret_cast<__m128i *>(data1p),
                    _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(data1p)), inv_mask_v));
            }
        }

        return minRecursionTimes;
    }

    // code by someone others
    static int other1()
    {
        using namespace std;

        int m, n, i, j, a[20], t[20], min, cpy[20];
        char c;
        scanf("%d", &m);
        while (m--)
        {
            scanf("%d", &n);
            memset(a, 0, sizeof(int) * 20);
            memset(t, 0, sizeof(int) * 20);
            memset(cpy, 0, sizeof(int) * 20);
            min = 10000;
            for (i = 0; i<n; i++)
                for (j = 0; j<n; j++)
                    if (cin >> c, c == 'w')
                        a[i] ^= (1 << j);
            memcpy(cpy, a, sizeof(int) * 20);
            for (int k = 0; k<(1 << n); k++)
            {
                memcpy(a, cpy, sizeof(int) * 20);
                memset(t, 0, sizeof(int) * 20);
                for (i = 0; i<n; i++)
                {
                    if (i == 0)
                        t[i] = k;
                    else
                        t[i] = a[i - 1];
                    a[i] ^= t[i];
                    a[i] ^= t[i] >> 1;
                    a[i] ^= (t[i] << 1) & ((1 << n) - 1);
                    a[i + 1] ^= t[i];
                }
                if (a[n - 1] == 0)
                {
                    int tem = 0;
                    for (i = 0; i<n; i++)
                        while (t[i])
                        {
                            t[i] &= (t[i] - 1);
                            tem++;
                        }
                    if (tem < min)
                        min = tem;
                }
            }
            if (min == 10000)
                printf("inf\n");
            else
                printf("%d\n", min);

        }
        system("pause");
        return 0;
    }
};

// http://bailian.openjudge.cn/practice/4144/
template <>
struct Bailian<4144>
{
    typedef std::array<int, 4> MyArray; // { cowNum [0...], A, B, stableNum [1...] }
    typedef std::vector<MyArray> MyVector;
    typedef std::forward_list<MyArray> MyList;

    static const int TIMEPOINT_MIN = 1;
    static const int TIMEPOINT_MAX = 1000000;
    static const int DURATION_MIN = 0;
    static const int DURATION_MAX = 500000;

    static void main()
    {
        // Input
#ifdef DEBUG
        MyTimer timer;

        unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
        std::default_random_engine generator(seed);
        std::uniform_int_distribution<int> distribution1(TIMEPOINT_MIN, TIMEPOINT_MAX);
        std::uniform_int_distribution<int> distribution2(DURATION_MIN, DURATION_MAX);

        int N = 50000;

        MyVector cowVec(N);
        for (int i = 0; i < N; ++i)
        {
            MyArray &cow = cowVec[i];
            cow[0] = i;
            cow[1] = distribution1(generator);
            cow[2] = std::min(TIMEPOINT_MAX, cow[1] + distribution2(generator));
        }

        timer.Start();
#else
        int N;
        scanf("%d", &N);

        MyVector cowVec(N);
        for (int i = 0; i < N; ++i)
        {
            MyArray &cow = cowVec[i];
            cow[0] = i;
            scanf("%d %d", &cow[1], &cow[2]);
        }
#endif
        // Implement
        std::sort(cowVec.begin(), cowVec.end(), [](const MyArray &left, const MyArray &right)
        {
            return left[1] < right[1] || (left[1] == right[1] && left[2] > right[2]);
        });

        MyList cowList1(N);
        MyList cowList2;
        std::copy(cowVec.begin(), cowVec.end(), cowList1.begin());

#ifdef DEBUG
        timer.Print<MyTimer::Milliseconds>("1");
        timer.Start();
#endif

        int stableNum = 0;

        while (!cowList1.empty())
        {
            ++stableNum;
            int last = 0;

            auto cowIter1 = cowList1.begin();
            auto lastCowIter1 = cowList1.before_begin();

            for(; cowIter1 != cowList1.end();)
            {
                MyArray &cow = *cowIter1;

                if (cow[1] > last)
                {
                    last = cow[2];
                    cow[3] = stableNum;
                    cowList2.push_front(std::move(cow));
                    cowIter1 = cowList1.erase_after(lastCowIter1);
                }
                else
                {
                    ++cowIter1;
                    ++lastCowIter1;
                }
            }
        }

#ifdef DEBUG
        timer.Print<MyTimer::Milliseconds>("2");
        timer.Start();
#endif

        // Output
        std::copy(cowList2.begin(), cowList2.end(), cowVec.begin());
        std::sort(cowVec.begin(), cowVec.end(), [](const MyArray &left, const MyArray &right)
        {
            return left[0] < right[0];
        });

#ifdef DEBUG
        timer.Print<MyTimer::Milliseconds>("3");
        timer.Start();
#endif

        std::string outStr;
        outStr.reserve(N * 7);
        outStr.append(std::to_string(stableNum)).append("\n");

        std::for_each(cowVec.begin(), cowVec.end(), [&](const MyArray &cow)
        {
            outStr.append(std::to_string(cow[3])).append("\n");
        });

#ifdef DEBUG
        timer.Print<MyTimer::Milliseconds>("4");
#else
        printf(outStr.c_str());
#endif
    }
};

////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    Bailian<4144>::main();

#ifdef DEBUG
    getchar();
#endif

    return 0;
}

////////////////////////////////////////////////////////////////
