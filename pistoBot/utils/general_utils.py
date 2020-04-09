# TODO move everything multi-purpose here
# Text managing utils
def _load_yaml(path: str):
    import yaml
    with open(path, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_dict


def _ml_init():
    from nltk import download
    from tensorflow.random import set_seed
    download("punkt")
    set_seed(42)