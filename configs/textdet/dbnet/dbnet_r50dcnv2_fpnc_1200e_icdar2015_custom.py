_base_ = [
    '../../_base_/runtime_10e.py',
   # '../../_base_/schedules/schedule_sgd_160e.py',
    '../../_base_/schedules/schedule_adam_step_6e_custom.py',
    '../../_base_/det_models/dbnet_r50dcnv2_fpnc.py',
    '../../_base_/det_datasets/comics_speech_bubble_dataset.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}

load_from = '/home/gsoykan20/.cache/torch/hub/checkpoints/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_r50dcnv2),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024))

# evaluation = dict(interval=100, metric='hmean-iou')
evaluation = dict(
    interval=1,
    metric=["hmean-iou"],
    save_best="0_hmean-iou:hmean",
    rule="greater"
)  # for best saving
checkpoint_config = dict(interval=100)  # for saving regardless
