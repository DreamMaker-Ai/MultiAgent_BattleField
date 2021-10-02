"""
check filters id and name in each convolutional layer for making animation
*** Need specify model path
"""

from modules.models import MyConv2DModel_v0B_Small_CBAM_1DConv_Share
import ray
from ray.rllib.agents.ppo import ppo
from ray.rllib.models import ModelCatalog
from battle_field_strategy_2D_v0 import BattleFieldStrategy
from settings.initial_settings import *
from tensorflow.keras.utils import plot_model


def main():
    model_path = PROJECT + '/checkpoints/' + TRIAL + '/checkpoint_010001/checkpoint-10001'

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    ModelCatalog.register_custom_model('my_model', MyConv2DModel_v0B_Small_CBAM_1DConv_Share)

    config = {"env": BattleFieldStrategy,
              "model": {"custom_model": "my_model"}
              # "framework": framework
              }  # use tensorflow 2

    # restore model
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(checkpoint_path=model_path)

    model = trainer.get_policy().model.base_model

    model.summary()
    print('\n')
    results_path = PROJECT + '/model.png'
    plot_model(model, to_file=results_path, show_shapes=True)

    # summarize filter shapes
    for i, layer in enumerate(model.layers):
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # get filter weights
        filters = layer.weights[0]
        print(i, layer.name, filters.shape)


if __name__ == '__main__':
    main()
