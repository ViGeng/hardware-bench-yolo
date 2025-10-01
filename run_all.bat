@echo off
setlocal enabledelayedexpansion

echo ==============================================================
echo Benchmark Test Started: %date% %time%
echo Testing all tasks, models, and datasets
echo Samples per test: 100
echo ==============================================================

set PYTHON_CMD=python
set MAIN_SCRIPT=main.py
set OUTPUT_DIR=.\results
set SAMPLES=100
set DEVICE=auto
set ENABLE_PLOTS=--plot

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

set LOG_FILE=%OUTPUT_DIR%\test_log_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

echo Configuration: > "%LOG_FILE%"
echo   Samples: %SAMPLES% >> "%LOG_FILE%"
echo   Device: %DEVICE% >> "%LOG_FILE%"

set TOTAL_TESTS=0
set PASSED_TESTS=0
set FAILED_TESTS=0

echo.
echo ========== CLASSIFICATION TESTS ==========

call :run_test classification resnet18 MNIST
call :run_test classification resnet18 CIFAR-10
call :run_test classification resnet18 ImageNet-Sample
call :run_test classification resnet50 MNIST
call :run_test classification resnet50 CIFAR-10
call :run_test classification resnet50 ImageNet-Sample
call :run_test classification efficientnet_b0 MNIST
call :run_test classification efficientnet_b0 CIFAR-10
call :run_test classification efficientnet_b0 ImageNet-Sample
call :run_test classification efficientnet_b3 MNIST
call :run_test classification efficientnet_b3 CIFAR-10
call :run_test classification efficientnet_b3 ImageNet-Sample
call :run_test classification vit_base_patch16_224 MNIST
call :run_test classification vit_base_patch16_224 CIFAR-10
call :run_test classification vit_base_patch16_224 ImageNet-Sample
call :run_test classification mobilenetv3_large_100 MNIST
call :run_test classification mobilenetv3_large_100 CIFAR-10
call :run_test classification mobilenetv3_large_100 ImageNet-Sample

echo.
echo ========== DETECTION TESTS ==========

call :run_test detection yolov8n COCO-Sample
call :run_test detection yolov8n KITTI
call :run_test detection yolov8n Test-Images
call :run_test detection yolov8s COCO-Sample
call :run_test detection yolov8s KITTI
call :run_test detection yolov8s Test-Images
call :run_test detection yolov8m COCO-Sample
call :run_test detection yolov8m KITTI
call :run_test detection yolov8m Test-Images
call :run_test detection fasterrcnn_resnet50_fpn COCO-Sample
call :run_test detection fasterrcnn_resnet50_fpn KITTI
call :run_test detection fasterrcnn_resnet50_fpn Test-Images
call :run_test detection fasterrcnn_mobilenet_v3_large_fpn COCO-Sample
call :run_test detection fasterrcnn_mobilenet_v3_large_fpn KITTI
call :run_test detection fasterrcnn_mobilenet_v3_large_fpn Test-Images
call :run_test detection fcos_resnet50_fpn COCO-Sample
call :run_test detection fcos_resnet50_fpn KITTI
call :run_test detection fcos_resnet50_fpn Test-Images

echo.
echo ========== SEGMENTATION TESTS ==========

call :run_test segmentation deeplabv3plus_resnet50 Cityscapes
call :run_test segmentation deeplabv3plus_resnet50 Synthetic-Segmentation
call :run_test segmentation deeplabv3plus_efficientnet_b0 Cityscapes
call :run_test segmentation deeplabv3plus_efficientnet_b0 Synthetic-Segmentation
call :run_test segmentation unet_resnet34 Cityscapes
call :run_test segmentation unet_resnet34 Synthetic-Segmentation
call :run_test segmentation unetplusplus_resnet50 Cityscapes
call :run_test segmentation unetplusplus_resnet50 Synthetic-Segmentation
call :run_test segmentation pspnet_resnet50 Cityscapes
call :run_test segmentation pspnet_resnet50 Synthetic-Segmentation
call :run_test segmentation fpn_resnet50 Cityscapes
call :run_test segmentation fpn_resnet50 Synthetic-Segmentation

goto :summary

:run_test
set task=%~1
set model=%~2
set dataset=%~3
set test_name=%task%_%model%_%dataset%
set /a TOTAL_TESTS+=1

echo.
echo [TEST %TOTAL_TESTS%] %task% ^| %model% ^| %dataset%
echo [TEST %TOTAL_TESTS%] %task% ^| %model% ^| %dataset% >> "%LOG_FILE%"

set TEST_OUTPUT_DIR=%OUTPUT_DIR%\%test_name%
set cmd=%PYTHON_CMD% %MAIN_SCRIPT% --device %DEVICE% --task %task% --model %model% --dataset "%dataset%" --samples %SAMPLES% --output-dir "%TEST_OUTPUT_DIR%" %ENABLE_PLOTS% --quiet

%cmd% >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% equ 0 (
    echo PASSED
    echo PASSED >> "%LOG_FILE%"
    set /a PASSED_TESTS+=1
) else (
    echo FAILED
    echo FAILED >> "%LOG_FILE%"
    set /a FAILED_TESTS+=1
)
goto :eof

:summary
echo.
echo ==============================================================
echo Test Completed: %date% %time%
echo ==============================================================
echo Total Tests: %TOTAL_TESTS%
echo Passed: %PASSED_TESTS%
echo Failed: %FAILED_TESTS%
if %TOTAL_TESTS% gtr 0 (
    set /a SUCCESS_RATE=%PASSED_TESTS%*100/%TOTAL_TESTS%
    echo Success Rate: !SUCCESS_RATE!%%
)
echo ==============================================================
echo Results saved in: %OUTPUT_DIR%
echo Log file: %LOG_FILE%
echo ==============================================================

echo. >> "%LOG_FILE%"
echo Test Completed: %date% %time% >> "%LOG_FILE%"
echo Total Tests: %TOTAL_TESTS% >> "%LOG_FILE%"
echo Passed: %PASSED_TESTS% >> "%LOG_FILE%"
echo Failed: %FAILED_TESTS% >> "%LOG_FILE%"
if %TOTAL_TESTS% gtr 0 (
    echo Success Rate: !SUCCESS_RATE!%% >> "%LOG_FILE%"
)

if %FAILED_TESTS% gtr 0 (exit /b 1) else (exit /b 0)