#!/bin/bash

# check_memory_leak.sh - Script to run memory leak detection using xctrace on macOS

# 确保脚本在出错时退出
set -e



# 检查参数
if [ $# -ne 2 ]; then
    echo "Example: $0 /galois/build_debug MemoryLeakTest.AllocateWithoutFree"
    exit 1
fi

BUILD_DIR=build_$1
GTEST_FILTER="$2"

# 检查 build_dir 是否存在
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory $BUILD_DIR does not exist."
    exit 1
fi

# 检查 galois_test 可执行文件是否存在
GALOIS_TEST="$BUILD_DIR/bin/galois_test"
if [ ! -f "$GALOIS_TEST" ]; then
    echo "Error: galois_test executable not found at $GALOIS_TEST."
    exit 1
fi
# 检查 entitlements.plist 文件是否存在（假设在项目根目录）
ENTITLEMENTS_FILE="entitlements.plist"
if [ ! -f "$ENTITLEMENTS_FILE" ]; then
    echo "Error: entitlements.plist not found at $ENTITLEMENTS_FILE."
    exit 1
fi

# 验证 galois_test 的签名是否包含 get-task-allow
echo "Verifying code signature of $GALOIS_TEST..."
SIGNATURE_OUTPUT=$(codesign -d --entitlements - "$GALOIS_TEST" 2>&1)
if echo "$SIGNATURE_OUTPUT" | grep -q "com.apple.security.get-task-allow"; then
    echo "Signature verification passed: get-task-allow entitlement is present."
else
    echo "Signature verification failed: get-task-allow entitlement is missing."
    echo "Attempting to re-sign $GALOIS_TEST with entitlements..."
    codesign -f -s - --entitlements "$ENTITLEMENTS_FILE" "$GALOIS_TEST"
    # 再次验证签名
    SIGNATURE_OUTPUT=$(codesign -d --entitlements - "$GALOIS_TEST" 2>&1)
    if echo "$SIGNATURE_OUTPUT" | grep -q "com.apple.security.get-task-allow"; then
        echo "Re-signing successful: get-task-allow entitlement added."
    else
        echo "Error: Failed to add get-task-allow entitlement after re-signing."
        echo "Signature details:"
        echo "$SIGNATURE_OUTPUT"
        exit 1
    fi
fi
# 切换到构建目录
cd "$BUILD_DIR"

# 生成带有时间戳的 trace 文件名，避免重复
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRACE_FILE="galois_test_leaks_${TIMESTAMP}.trace"
LEAKS_REPORT="leaks_report_${TIMESTAMP}.txt"

# 运行 xctrace 检测内存泄漏
echo "Running xctrace for memory leak detection..."
xctrace record \
    --template "Leaks" \
    --output "$TRACE_FILE" \
    --launch bin/galois_test \
    -- --gtest_filter="$GTEST_FILTER"

# 检查是否生成了 .trace 文件
echo "Checking if trace file exists: $(pwd)/$TRACE_FILE"
if [ -d "$TRACE_FILE" ]; then
    echo "Leak detection completed. Trace file generated: $TRACE_FILE"
else
    echo "Error: No trace file generated."
    exit 1
fi


