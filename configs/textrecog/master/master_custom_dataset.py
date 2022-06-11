_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_models/master.py',
    '../../_base_/schedules/schedule_adam_step_12e.py',
    '../../_base_/recog_pipelines/master_pipeline.py',
    '../../_base_/recog_datasets/comic_speech_bubble_dataset.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

data = dict(
    workers_per_gpu=32,
    samples_per_gpu=4,
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1,
    metric=["acc"],
    save_best="0_char_precision",
    rule="greater"
)  # for best saving
checkpoint_config = dict(interval=100, metric='acc')