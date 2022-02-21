_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_adam_step_6e_custom.py',
    '../../_base_/recog_pipelines/nrtr_pipeline.py',
    '../../_base_/recog_datasets/comic_speech_bubble_dataset.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True)

model = dict(
    type='NRTR',
    backbone=dict(
        type='ResNet31OCR',
        layers=[1, 2, 5, 3],
        channels=[32, 64, 128, 256, 512, 512],
        stage4_pool_cfg=dict(kernel_size=(2, 1), stride=(2, 1)),
        last_stage_pool=False),
    encoder=dict(type='NRTREncoder'),
    decoder=dict(type='NRTRDecoder'),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=40)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
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
