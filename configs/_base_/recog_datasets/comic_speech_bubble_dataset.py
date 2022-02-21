dataset_type = 'OCRDataset'

root = 'tests/data/ocr_comics_speech_bubble_dataset'

train_img_prefix = f'{root}/train/imgs'
train_anno_file1 = f'{root}/train/label.txt'

test_img_prefix = f'{root}/test/imgs'
test_anno_file1 = f'{root}/test/label.txt'

train = dict(
    type=dataset_type,
    img_prefix=train_img_prefix,
    ann_file=train_anno_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

test = dict(
    type=dataset_type,
    img_prefix=test_img_prefix,
    ann_file=test_anno_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

train_list = [train]

test_list = [test]
