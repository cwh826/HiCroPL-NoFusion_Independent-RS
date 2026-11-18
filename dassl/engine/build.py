from dassl.utils import Registry, check_availability

TRAINER_REGISTRY = Registry("TRAINER")


def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()     # 从注册表 TRAINER_REGISTRY 取出所有已注册的 trainer 名字，然后检查 cfg.TRAINER.NAME 是否在列表中。
    check_availability(cfg.TRAINER.NAME, avai_trainers)     # 如果用户传了个不存在的名字（比如 --trainer abc），直接报错退出。
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)      # 从注册表里按名字取出对应的类，把 cfg 传进去实例化，返回 trainer 对象。
