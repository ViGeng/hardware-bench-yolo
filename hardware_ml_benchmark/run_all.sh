#!/bin/bash

echo "=============================================================="
echo "Benchmark Test Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Testing all tasks, models, and datasets"
echo "Samples per test: 100"
echo "=============================================================="

PYTHON_CMD="python"
MAIN_SCRIPT="main.py"
OUTPUT_DIR="./results"
SAMPLES=100
DEVICE="auto"
ENABLE_PLOTS="--plot"

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Create log file with timestamp
LOG_FILE="${OUTPUT_DIR}/test_log_$(date '+%Y%m%d_%H%M%S').log"

echo "Configuration:" > "$LOG_FILE"
echo "  Samples: $SAMPLES" >> "$LOG_FILE"
echo "  Device: $DEVICE" >> "$LOG_FILE"

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a single test
run_test() {
    local task=$1
    local model=$2
    local dataset=$3
    local test_name="${task}_${model}_${dataset}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo ""
    echo "[TEST $TOTAL_TESTS] $task | $model | $dataset"
    echo "[TEST $TOTAL_TESTS] $task | $model | $dataset" >> "$LOG_FILE"

    local test_output_dir="${OUTPUT_DIR}/${test_name}"
    local cmd="$PYTHON_CMD $MAIN_SCRIPT --device $DEVICE --task $task --model $model --dataset \"$dataset\" --samples $SAMPLES --output-dir \"$test_output_dir\" $ENABLE_PLOTS --quiet"

    eval $cmd >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        echo "PASSED"
        echo "PASSED" >> "$LOG_FILE"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "FAILED"
        echo "FAILED" >> "$LOG_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

echo ""
echo "========== CLASSIFICATION TESTS =========="

run_test classification resnet18 MNIST
run_test classification resnet18 CIFAR-10
run_test classification resnet18 ImageNet-Sample
run_test classification resnet50 MNIST
run_test classification resnet50 CIFAR-10
run_test classification resnet50 ImageNet-Sample
run_test classification efficientnet_b0 MNIST
run_test classification efficientnet_b0 CIFAR-10
run_test classification efficientnet_b0 ImageNet-Sample
run_test classification efficientnet_b3 MNIST
run_test classification efficientnet_b3 CIFAR-10
run_test classification efficientnet_b3 ImageNet-Sample
run_test classification vit_base_patch16_224 MNIST
run_test classification vit_base_patch16_224 CIFAR-10
run_test classification vit_base_patch16_224 ImageNet-Sample
run_test classification mobilenetv3_large_100 MNIST
run_test classification mobilenetv3_large_100 CIFAR-10
run_test classification mobilenetv3_large_100 ImageNet-Sample

echo ""
echo "========== DETECTION TESTS =========="

run_test detection yolov8n COCO-Sample
run_test detection yolov8n KITTI
run_test detection yolov8n Test-Images
run_test detection yolov8s COCO-Sample
run_test detection yolov8s KITTI
run_test detection yolov8s Test-Images
run_test detection yolov8m COCO-Sample
run_test detection yolov8m KITTI
run_test detection yolov8m Test-Images
run_test detection fasterrcnn_resnet50_fpn COCO-Sample
run_test detection fasterrcnn_resnet50_fpn KITTI
run_test detection fasterrcnn_resnet50_fpn Test-Images
run_test detection fasterrcnn_mobilenet_v3_large_fpn COCO-Sample
run_test detection fasterrcnn_mobilenet_v3_large_fpn KITTI
run_test detection fasterrcnn_mobilenet_v3_large_fpn Test-Images
run_test detection fcos_resnet50_fpn COCO-Sample
run_test detection fcos_resnet50_fpn KITTI
run_test detection fcos_resnet50_fpn Test-Images

echo ""
echo "========== SEGMENTATION TESTS =========="

run_test segmentation deeplabv3plus_resnet50 Cityscapes
run_test segmentation deeplabv3plus_resnet50 Synthetic-Segmentation
run_test segmentation deeplabv3plus_efficientnet_b0 Cityscapes
run_test segmentation deeplabv3plus_efficientnet_b0 Synthetic-Segmentation
run_test segmentation unet_resnet34 Cityscapes
run_test segmentation unet_resnet34 Synthetic-Segmentation
run_test segmentation unetplusplus_resnet50 Cityscapes
run_test segmentation unetplusplus_resnet50 Synthetic-Segmentation
run_test segmentation pspnet_resnet50 Cityscapes
run_test segmentation pspnet_resnet50 Synthetic-Segmentation
run_test segmentation fpn_resnet50 Cityscapes
run_test segmentation fpn_resnet50 Synthetic-Segmentation

# Print summary
echo ""
echo "=============================================================="
echo "Test Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
if [ $TOTAL_TESTS -gt 0 ]; then
    SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "Success Rate: ${SUCCESS_RATE}%"
fi
echo "=============================================================="
echo "Results saved in: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "=============================================================="

# Append summary to log file
echo "" >> "$LOG_FILE"
echo "Test Completed: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "Total Tests: $TOTAL_TESTS" >> "$LOG_FILE"
echo "Passed: $PASSED_TESTS" >> "$LOG_FILE"
echo "Failed: $FAILED_TESTS" >> "$LOG_FILE"
if [ $TOTAL_TESTS -gt 0 ]; then
    echo "Success Rate: ${SUCCESS_RATE}%" >> "$LOG_FILE"
fi

# Exit with appropriate code
if [ $FAILED_TESTS -gt 0 ]; then
    exit 1
else
    exit 0
fi