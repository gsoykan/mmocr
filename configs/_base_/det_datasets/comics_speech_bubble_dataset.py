root = 'tests/data/comics_speech_bubble_dataset'

# dataset with type='IcdarDataset'
train = dict(
    type='IcdarDataset',
    ann_file=f'{root}/train/instances_train.json',
    img_prefix=f'{root}/train/imgs',
    pipeline=None)

test = dict(
    type='IcdarDataset',
    ann_file=f'{root}/test/instances_test.json',
    img_prefix=f'{root}/test/imgs',
    pipeline=None,
    test_mode=True)

train_list = [train]

test_list = [test]
