import sys
import argparse
import os
import yaml
from pathlib import Path

class GetParam:

    def __init__(self, config=None):
        """
        Get the parameters for the model.
        Load all keys from config.yml as class attributes dynamically.
        """
        self.root = self.get_project_root()
        if os.path.isfile(config):  # 如果传入的是文件路径
            configpath = Path(config)
            if configpath.exists():
                with open(configpath, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise FileNotFoundError(f'Configuration file not found: {configpath}')


        elif os.path.exists(os.path.join(self.root, 'configs', f"{config}.yml")):
            configpath = os.path.join(self.root, 'configs', f"{config}.yml")
            with open(configpath, 'r') as f:
                config = yaml.safe_load(f)
        else:
            print(os.path.join(Path.cwd().parent, 'configs', f"{config}.yml"))
            raise ValueError(f'Invalid configuration path: {config}')

        # 将配置文件中的所有键值对作为属性赋值
        for key, value in config.items():
            setattr(self, key, value)
            if type(value) is dict:
                for subkey, subvalue in value.items():
                    setattr(self, f"{key}_{subkey}", subvalue)


        self.vae_model_path = os.path.join(self.root, config['vae_model_path'], 'vae_{}'.format(config['configname']))

        print(self.vae_model_path)
        # 存一份 config 方便直接访问
        self._config = config

    def __repr__(self):
        return f"GetParam({self._config})"
    def get_project_root(self):
        """获取项目根目录"""
        # 从当前文件向上查找，直到找到标志性文件
        current = Path(__file__).parent
        while current != current.parent:
            # 查找项目标志文件如 .git, pyproject.toml 等
            if any((current / marker).exists() for marker in ['main.py']):
                return current
            current = current.parent

        # 找不到标志文件时回退到配置值或当前工作目录父目录
        return Path.cwd().parent





if __name__ == "__main__":
    # Example usage
    import os
    cfg=GetParam(os.path.join(Path.cwd().parent,'configs/configtest.yml'))
    # cfg=getparament('configtest')

    print(cfg.istest)