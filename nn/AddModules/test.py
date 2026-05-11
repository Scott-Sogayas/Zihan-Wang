# test.py（放在任意地方都可以，只要当前解释器能导入 ultralytics）
from ultralytics.nn.AddModules.EfficientNetV2 import efficientnet_v2, MoCAttention

model = efficientnet_v2('efficientnet_v2_s')
print([name for name, m in model.named_modules() if isinstance(m, MoCAttention)])
print("MoCAttention count:", sum(1 for m in model.modules() if isinstance(m, MoCAttention)))