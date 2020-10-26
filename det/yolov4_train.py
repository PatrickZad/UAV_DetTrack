from detectron2.config import get_cfg

def setup(cfg_file):
    cfg=get_cfg()
    cfg.merge_from_file(cfg_file)