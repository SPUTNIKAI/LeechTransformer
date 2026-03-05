import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# --- 1. ГЕНЕРАЦИЯ ДАННЫХ ---
def visualize_monster_geometry(model, text_input):
    model.eval()
    with torch.no_grad():
        # Получаем эмбеддинги после слоев "Пробуждения"
        logits, h_states = model(text_input) 
        # h_states имеет форму [Batch, Seq, d_model]
        
    # Преобразуем в 2D через PCA для визуализации "упаковки"
    flat_h = h_states.view(-1, model.d_model).cpu().numpy()
    pca = PCA(n_components=2)
    h_2d = pca.fit_transform(flat_h)

    # --- 2. ПОСТРОЕНИЕ ГРАФИКА ---
    plt.figure(figsize=(12, 6))
    
    # Левый график: Плотность распределения (Geometry)
    plt.subplot(1, 2, 1)
    plt.scatter(h_2d[:, 0], h_2d[:, 1], alpha=0.5, c='indigo', s=10)
    plt.title("Leech-Lattice Embedding Space (PCA)")
    plt.grid(True, linestyle='--', alpha=0.6)
    # Здесь ты увидишь "сетку" или выраженные кластеры - это работа Λ24

    # Правый график: Резонанс 1/137 (Monster Constant)
    plt.subplot(1, 2, 2)
    # Посмотрим на активации после MonsterConstantGenerator
    activations = torch.sin(h_states[0] * (1/137.036)).cpu().numpy()
    plt.imshow(activations[:50, :50], cmap='magma')
    plt.title("Monster Resonance Map (1/137)")
    plt.colorbar(label='Phase Shift')

    plt.tight_layout()
    plt.show()

# Запуск (предполагая, что модель инициализирована как `monster_model`)
# test_input = torch.randint(0, 10000, (1, 128))
# visualize_monster_geometry(monster_model, test_input)
