#include <iostream>
#include <algorithm>
#include <cstdint>
#include <ctime>

// 跨平台缓存检测头文件
#ifdef __linux__
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// 分块参数结构体
struct TileSize {
    int ti_inner;  // 内层 i 分块 (L1)
    int ti_mid;    // 中层 i 分块 (L2)
    int ti_outer;  // 外层 i 分块 (L3)
    int tj_inner;  // 内层 j 分块 (L1)
    int tj_mid;    // 中层 j 分块 (L2)
    int tj_outer;  // 外层 j 分块 (L3)
    int tk_mid;    // 中层 k 分块 (L2)
};

// 缓存参数结构体（单位：字节）
struct CacheConfig {
    int64_t l1_size;  // L1 缓存大小
    int64_t l2_size;  // L2 缓存大小
    int64_t l3_size;  // L3 缓存大小

    // 默认构造函数，使用典型值作为回退
    CacheConfig() : l1_size(32 * 1024), l2_size(256 * 1024), l3_size(8 * 1024 * 1024) {
        detect_cache_sizes();  // 尝试检测本地缓存大小
    }

    // 检测本地缓存大小
    void detect_cache_sizes() {
#ifdef __linux__
        // Linux 使用 sysconf 获取缓存大小
        l1_size = sysconf(_SC_LEVEL1_DCACHE_SIZE);
        l2_size = sysconf(_SC_LEVEL2_CACHE_SIZE);
        l3_size = sysconf(_SC_LEVEL3_CACHE_SIZE);
        if (l1_size <= 0 || l2_size <= 0 || l3_size <= 0) {
            std::cerr << "Warning: Failed to detect cache sizes on Linux, using defaults.\n";
            l1_size = 32 * 1024;
            l2_size = 256 * 1024;
            l3_size = 8 * 1024 * 1024;
        }
#elif defined(__APPLE__)
        // macOS 使用 sysctl 获取缓存大小
        size_t len = sizeof(int64_t);
        if (sysctlbyname("hw.l1dcachesize", &l1_size, &len, NULL, 0) != 0) {
            l1_size = 32 * 1024;  // 默认值
        }
        if (sysctlbyname("hw.l2cachesize", &l2_size, &len, NULL, 0) != 0) {
            l2_size = 256 * 1024;
        }
        if (sysctlbyname("hw.l3cachesize", &l3_size, &len, NULL, 0) != 0) {
            l3_size = 8 * 1024 * 1024;
        }
#else
        std::cerr << "Warning: Cache detection not supported on this platform, using defaults.\n";
#endif
    }
};

//分块大小计算类（模板化）
template <typename T>
class TileSizeCalculator {
private:
    const CacheConfig& cache;

     // 根据数据类型动态调整初始值
    int get_initial_inner_size() const {
        int base_size = 16;//基础值
        int type_size = sizeof(T);
        return std::max(1, base_size / type_size); //确保至少为1
    }  
    int get_initial_mid_size() const {
        return 2;  // 中层初始值保持较小
    }

    int get_initial_tk_mid() const {
        int type_size = sizeof(T);
        int l2_elements = cache.l2_size / type_size;
        int target_tk = l2_elements / (get_initial_inner_size() * get_initial_mid_size() * 2);  // 粗略估计
        return std::min(64, std::max(32, target_tk));  // 限制在 32-64 之间
    }
public:
    TileSizeCalculator(const CacheConfig& cache_config) : cache(cache_config){}

    TileSize compute(int M, int N, int K) const {
        // 边界检查
        if (M <= 0 || N <= 0 || K <= 0) {
            throw std::invalid_argument("Matrix dimensions M, N, K must be positive.");
        }

        TileSize ts;
        const int type_size = sizeof(T);

        //L1分块：内层
        int l1_elements = cache.l1_size / type_size;
        ts.ti_inner = get_initial_inner_size();
        ts.tj_inner = get_initial_inner_size();
        int tk_inner = get_initial_tk_mid();
        int l1_usage = ts.ti_inner * tk_inner + tk_inner * ts.tj_inner + ts.ti_inner * ts.tj_inner;
        while (l1_usage > l1_elements && ts.ti_inner > 1) {
            ts.ti_inner /= 2;
            ts.tj_inner /= 2;
            l1_usage = ts.ti_inner * tk_inner + tk_inner * ts.tj_inner + ts.ti_inner * ts.tj_inner;
        }

        // L2 分块：中层
        int l2_elements = cache.l2_size / type_size;
        ts.ti_mid = get_initial_mid_size();
        ts.tj_mid = get_initial_mid_size();
        ts.tk_mid = get_initial_tk_mid();
        int ti = ts.ti_inner * ts.ti_mid;
        int tj = ts.tj_inner * ts.tj_mid;
        int l2_usage = ti * ts.tk_mid + ts.tk_mid * tj + ti * tj;
        while (l2_usage > l2_elements && ts.ti_mid > 1) {
            ts.ti_mid /= 2; // 若超出 L2，减小中层参数
            ts.tj_mid /= 2;
            ti = ts.ti_inner * ts.ti_mid;
            tj = ts.tj_inner * ts.tj_mid;
            l2_usage = ti * ts.tk_mid + ts.tk_mid * tj + ti * tj;
        }
        while (l2_usage > l2_elements && ts.tk_mid > 8) {
            ts.tk_mid /= 2;
            l2_usage = ti * ts.tk_mid + ts.tk_mid * tj + ti * tj;
        }

        // L3 分块：外层
        int l3_elements = cache.l3_size / type_size;
        int inner_mid_i = ts.ti_inner * ts.ti_mid;
        int inner_mid_j = ts.tj_inner * ts.tj_mid;
        ts.ti_outer = (M + inner_mid_i - 1) / inner_mid_i;  // 向上取整避免遗漏
        ts.tj_outer = (N + inner_mid_j - 1) / inner_mid_j;
        int l3_usage = M * K + K * N + M * N;
        if (l3_usage > l3_elements) {
            std::cerr << "Warning: Matrix size exceeds L3 cache capacity for type size " << type_size << " bytes!\n";
            ts.ti_outer = std::min(ts.ti_outer, l3_elements / (K + N + 1));
            ts.tj_outer = std::min(ts.tj_outer, l3_elements / (M + K + 1));
        }

        return ts;

    }
};

