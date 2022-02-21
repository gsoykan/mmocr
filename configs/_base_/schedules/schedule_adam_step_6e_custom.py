# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=0.5))
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 6
