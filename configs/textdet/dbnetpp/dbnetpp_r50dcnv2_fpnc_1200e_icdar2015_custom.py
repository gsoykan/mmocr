_base_ = [
    '../../_base_/default_runtime.py',
  #  '../../_base_/schedules/schedule_sgd_1200e.py',
    '../../_base_/schedules/schedule_adam_step_6e_custom.py',
    '../../_base_/det_models/dbnetpp_r50dcnv2_fpnc.py',
    '../../_base_/det_datasets/comics_speech_bubble_dataset.py',
    #   '../../_base_/det_datasets/icdar2015.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}

load_from = '/home/gsoykan20/.cache/torch/checkpoints/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth'

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

evaluation = dict(
    interval=1,
    metric='hmean-iou',
    save_best='0_hmean-iou:hmean',
    rule='greater')
checkpoint_config = dict(interval=100)  # for saving regardless
