base = ['./sam_base.py']

model = dict(
    loss_skel=dict(
        type="SoftCLDice",
        sigmoid=True,
        loss_weight=0.1)
)
