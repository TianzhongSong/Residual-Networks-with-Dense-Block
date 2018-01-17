from .callbacks import Step

def onetenth_150_200_250(lr):
    steps = [150, 200,250]
    lrs = [lr, lr/10, lr/100,lr/1000]
    return Step(steps, lrs)

def onetenth_20_30(lr):
    steps = [20, 30]
    lrs = [lr, lr/10, lr/100]
    return Step(steps, lrs)

def wideresnet_step(lr):
    steps = [60, 120, 160]
    lrs = [lr, lr/5, lr/25, lr/125]
    return Step(steps, lrs)