#!/bin/bash
# Comprehensive GPU Test Suite Runner
# Runs all GPU tests with detailed output

set -e

echo "🧪 =================================================="
echo "   Hive-GPU Comprehensive Test Suite"
echo "   Running all GPU tests on: $(uname -m) $(uname -s)"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run test suite
run_test_suite() {
    local test_name=$1
    local test_file=$2
    
    echo -e "${BLUE}📦 Running: $test_name${NC}"
    echo "   File: tests/$test_file.rs"
    
    if cargo test --features metal-native --test "$test_file" 2>&1 | tee /tmp/test_output.log | grep -q "test result: ok"; then
        local count=$(grep "test result: ok" /tmp/test_output.log | awk '{print $3}')
        echo -e "${GREEN}   ✅ Passed: $count tests${NC}"
        return 0
    else
        echo -e "${RED}   ❌ Failed${NC}"
        return 1
    fi
    echo ""
}

# Track results
total_tests=0
failed_suites=0

echo "🔍 1. GPU Detection Tests"
if run_test_suite "GPU Detection" "gpu_detection_tests"; then
    ((total_tests+=9))
else
    ((failed_suites++))
fi
echo ""

echo "🔢 2. Vector Operations Tests"
if run_test_suite "Vector Operations" "gpu_vector_ops_tests"; then
    ((total_tests+=11))
else
    ((failed_suites++))
fi
echo ""

echo "💾 3. Memory Management Tests"
if run_test_suite "Memory Management" "gpu_memory_tests"; then
    ((total_tests+=10))
else
    ((failed_suites++))
fi
echo ""

echo "📊 4. VRAM Monitoring Tests"
if run_test_suite "VRAM Monitoring" "gpu_vram_tests"; then
    ((total_tests+=10))
else
    ((failed_suites++))
fi
echo ""

echo "🔗 5. Integration Tests"
if run_test_suite "Integration" "integration_tests"; then
    ((total_tests+=9))
else
    ((failed_suites++))
fi
echo ""

echo "📚 6. Device Info Tests"
if run_test_suite "Device Info" "device_info_tests"; then
    ((total_tests+=4))
else
    ((failed_suites++))
fi
echo ""

echo "⚡ 7. Performance Benchmarks"
if run_test_suite "Performance" "gpu_performance_tests"; then
    ((total_tests+=10))
else
    ((failed_suites++))
fi
echo ""

echo "💪 8. Stress Tests"
if run_test_suite "Stress Tests" "gpu_stress_tests"; then
    ((total_tests+=9))
else
    ((failed_suites++))
fi
echo ""

# Summary
echo "=================================================="
echo "   Test Suite Summary"
echo "=================================================="
if [ $failed_suites -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo "   Total tests run: $total_tests"
    echo ""
    echo "   GPU Test Suites:"
    echo "   • GPU Detection: 9 tests ✓"
    echo "   • Vector Operations: 11 tests ✓"
    echo "   • Memory Management: 10 tests ✓"
    echo "   • VRAM Monitoring: 10 tests ✓"
    echo "   • Integration: 9 tests ✓"
    echo "   • Device Info: 4 tests ✓"
    echo "   • Performance Benchmarks: 10 tests ✓"
    echo "   • Stress Tests: 9 tests ✓"
    echo ""
    echo "🎉 All GPU functionality validated!"
    echo "📊 Total: $total_tests comprehensive GPU tests"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo "   Failed suites: $failed_suites"
    echo ""
    echo "Run with verbose output:"
    echo "  cargo test --features metal-native -- --nocapture"
    exit 1
fi

