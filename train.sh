export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=1,3 \
    python3 train.py \
    --outdir=output_path \
    --gpus=4 \
    --resume /raid/student/2021/ai21btech11005/MAT/Places_512_FullData.pkl \
    --batch=32 \
    --metrics=fid_imagenet \
    --mirror=True \
    --cond=False \
    --cfg=places256 \
    --aug=noaug \
    --generator=networks.mat.Generator \
    --discriminator=networks.mat.Discriminator \
    --loss=losses.loss.TwoStageLoss \
    --pr=0.1 \
    --pl=False \
    --truncation=0.5 \
    --style_mix=0.5 \
    --ema=10 \
    --lr=0.001 #Done
    # --data=training_data_path \
    # --data_val=val_data_path \