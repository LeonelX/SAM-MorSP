base = ['./sam_base.py']


model = dict(
    MorSP = dict(
        type = 'MorSP',
        iterations = 20,
        entropy_epsilon = 1.0,
        lam = 1.0,
        eta = 1,
        ito = 1e-2,
        delta = 1,
        ker_halfsize = 2,
        skeleton = dict(
            type='SmoothSkeleton',
            half_size=3,
            iter=50,
            alpha=0.05
        )
    ),
    loss_skel=dict(
        type="SoftCLDice",
        sigmoid=True,
        loss_weight=0.1)
)