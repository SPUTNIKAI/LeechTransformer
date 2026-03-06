from safetensors.torch import save_file
import torch
import os

# Класс-заглушка для успешной десериализации
class LeechConfig:
    vocab_size: int               # размер словаря
    d_model: int = 192             # размерность модели (кратна 24)
    n_layers: int = 12             # число слоёв
    n_heads: int = 8               # число голов внимания
    block_size: int = 512           # максимальная длина контекста
    dropout: float = 0.05
    bias: bool = False
    tie_weights: bool = True        # разделять веса эмбеддингов и head
    lambda_geo: float = 0.01        # вес геометрической потери
    resonance_threshold: float = 0.95  # порог для детекции «сна»

# Загружаем чекпоинт
checkpoint = torch.load(
    "/Users/anatolii/Проекты/LeechTransformer/data/checkpoint_step_100000_LEECH.pt",
    map_location=torch.device('cpu'),
    weights_only=False
)

# Выводим все ключи верхнего уровня
print("Ключи в чекпоинте:", checkpoint.keys())

# Ищем, где лежат веса модели (обычно это 'model', 'state_dict' или 'model_state_dict')
possible_keys = ['model', 'state_dict', 'model_state_dict', 'module']
weights = None
for key in possible_keys:
    if key in checkpoint and isinstance(checkpoint[key], dict):
        weights = checkpoint[key]
        print(f"Найдены веса под ключом '{key}'")
        break

# Если не нашли по стандартным ключам, пробуем собрать все тензоры из корня
if weights is None:
    print("Стандартные ключи не найдены, собираем все тензоры из корня...")
    weights = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    print(f"Найдено {len(weights)} тензоров")

# Сохраняем только тензоры
if weights:
    save_file(weights, "/Users/anatolii/Проекты/LeechTransformer/data/model.safetensors")
    print("Файл model.safetensors успешно сохранён!")
    print("Количество сохранённых тензоров:", len(weights))
else:
    print("Не удалось найти тензоры в чекпоинте.")