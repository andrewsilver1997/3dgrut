#!/bin/bash

# Check if dataset and stage arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> <stage>"
    echo ""
    echo "Datasets:"
    echo "  castro: Castro simulation dataset"
    echo "  nyx:    Nyx simulation dataset"
    echo ""
    echo "Stages:"
    echo "  stage0: Train first prefix from scratch"
    echo "  stage1: Train all subsequent stages in sequence"
    echo ""
    echo "Examples:"
    echo "  $0 castro stage0"
    echo "  $0 castro stage1"
    echo "  $0 nyx stage0"
    echo "  $0 nyx stage1"
    exit 1
fi

DATASET=$1
STAGE=$2
#MEMBER=$3

# Configure based on dataset
if [ "$DATASET" == "castro" ]; then
    NUMBERS=($(seq -w 1495 -5 50))
    PREFIXES=("090_075" "090_085" "090_090")
    OUT_DIR=results/castro
    MAX_GAUSSIANS=4096
    MAX_INDEX=289
    DATA_PATH_PREFIX="data/castro"

elif [ "$DATASET" == "nyx" ]; then
    NUMBERS=($(seq -w 200 2 1398))
    PREFIXES=("01025_00215_055")
    OUT_DIR=results/nyx
    MAX_GAUSSIANS=16384
    MAX_INDEX=599
    DATA_PATH_PREFIX="data/nyx"

else
    echo "Error: Invalid dataset '$DATASET'"
    echo "Please specify 'castro' or 'nyx'"
    exit 1
fi

export TORCH_EXTENSIONS_DIR=$OUT_DIR/.cache
mkdir -p $OUT_DIR

if [ "$STAGE" == "stage0" ]; then
    PREFIX_INPUT=${PREFIXES[0]}
    PREFIX=${PREFIX_INPUT}_den

    echo "=== Running Stage 0: Training $PREFIX ($DATASET) ==="

    runjob() {
        local id=$1
        local scene=${PREFIX}_0${NUMBERS[id]}

        # skip if "Training Complete." is in the log
        if grep -q "Training Complete." $OUT_DIR/$scene.log; then
            echo "=== Skipping scene: $scene ==="
            return
        fi

        echo "=== Running scene: $scene ==="

        # Check if this is the first job (id=0)
        if [ $id -eq 0 ]; then
            CUDA_LAUNCH_BLOCKING=1 python train.py --config-name apps/scivis_3dgut_mcmc.yaml \
                path=$DATA_PATH_PREFIX/$scene out_dir=$OUT_DIR > $OUT_DIR/$scene.log \
                import_ply.enabled=false strategy.add.max_n_gaussians=$MAX_GAUSSIANS \
                model.progressive_training.max_n_features=0 render.particle_radiance_sph_degree=0
        else
            local pre_scene=${PREFIX}_0${NUMBERS[id-1]}
            python train.py --config-name apps/scivis_3dgut_mcmc.yaml \
                path=$DATA_PATH_PREFIX/$scene out_dir=$OUT_DIR > $OUT_DIR/$scene.log \
                import_ply.enabled=true import_ply.path=$OUT_DIR/$pre_scene/export_last.ply \
                strategy.add.max_n_gaussians=$MAX_GAUSSIANS \
                strategy.relocate.start_iteration=-1 strategy.relocate.end_iteration=-1 \
                strategy.perturb.start_iteration=-1 strategy.perturb.end_iteration=-1 \
                strategy.add.start_iteration=-1 strategy.add.end_iteration=-1 \
                model.progressive_training.max_n_features=0 render.particle_radiance_sph_degree=0
        fi
    }

    for i in $(seq 0 $MAX_INDEX); do
        runjob $i
    done

elif [ "$STAGE" == "stage1" ]; then
    echo "=== Running Stage 1: Training sequence ($DATASET) ==="

    # Calculate number of transitions
    NUM_PREFIXES=${#PREFIXES[@]}
    MAX_IDX=$((NUM_PREFIXES - 1))

    # Loop through all target prefixes (starting from index 1)
    for idx in $(seq 1 $MAX_IDX); do
        INIT_PREFIX_INPUT=${PREFIXES[$((idx-1))]}
        TARGET_PREFIX_INPUT=${PREFIXES[$idx]}

        INIT_PREFIX=${INIT_PREFIX_INPUT}_den
        PREFIX=${TARGET_PREFIX_INPUT}_den

        echo ""
        echo "=========================================="
        echo "Training $PREFIX using $INIT_PREFIX initialization"
        echo "=========================================="
        echo ""

        runjob() {
            local id=$1
            local scene=${PREFIX}_0${NUMBERS[id]}
            local init_scene=${INIT_PREFIX}_0${NUMBERS[id]}

            # skip if "Training Complete." is in the log
            if grep -q "Training Complete." $OUT_DIR/$scene.log; then
                echo "=== Skipping scene: $scene ==="
                return
            fi

            echo "=== Running scene: $scene ==="

            if [ "$DATASET" == "nyx" ]; then
                python train.py --config-name apps/scivis_3dgut_mcmc.yaml \
                      path=$DATA_PATH_PREFIX/$TARGET_PREFIX_INPUT/$scene out_dir=$OUT_DIR > $OUT_DIR/$scene.log \
                      import_ply.enabled=true import_ply.path=$OUT_DIR/$init_scene/export_last.ply \
                      strategy.add.max_n_gaussians=$MAX_GAUSSIANS \
                      strategy.relocate.start_iteration=-1 strategy.relocate.end_iteration=-1 \
                      strategy.perturb.start_iteration=-1 strategy.perturb.end_iteration=-1 \
                      strategy.add.start_iteration=-1 strategy.add.end_iteration=-1 \
                      model.progressive_training.max_n_features=0
            else
                python train.py --config-name apps/scivis_3dgut_mcmc.yaml \
                      path=$DATA_PATH_PREFIX/$scene out_dir=$OUT_DIR > $OUT_DIR/$scene.log \
                      import_ply.enabled=true import_ply.path=$OUT_DIR/$init_scene/export_last.ply \
                      strategy.add.max_n_gaussians=$MAX_GAUSSIANS \
                      strategy.relocate.start_iteration=-1 strategy.relocate.end_iteration=-1 \
                      strategy.perturb.start_iteration=-1 strategy.perturb.end_iteration=-1 \
                      strategy.add.start_iteration=-1 strategy.add.end_iteration=-1 \
                      model.progressive_training.max_n_features=0
            fi
        }

        for i in $(seq 0 $MAX_INDEX); do
            runjob $i
        done
    done

    echo ""
    echo "=========================================="
    echo "Stage 1 Complete: All sequences trained ($DATASET)"
    echo "=========================================="

else
    echo "Error: Invalid stage '$STAGE'"
    echo "Please specify 'stage0' or 'stage1'"
    exit 1
fi
