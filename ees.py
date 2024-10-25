import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as mpatches


# Функции для пересчета единиц
def convert_resistance(R, unit):
    units = {"Ом": 1, "кОм": 1e3, "МОм": 1e6}
    return R * units[unit]


def convert_capacitance(C, unit):
    units = {"Ф": 1, "мкФ": 1e-6, "пФ": 1e-12}
    return C * units[unit]


def convert_inductance(L, unit):
    units = {"Гн": 1, "мГн": 1e-3, "мкГн": 1e-6}
    return L * units[unit]


# Функции для вычисления АЧХ и ФЧХ
def rc_response(R, C, f_range):
    omega = 2 * np.pi * f_range
    H_rc = 1 / (1 + 1j * omega * R * C)
    magnitude_rc = 20 * np.log10(np.abs(H_rc))
    phase_rc = np.angle(H_rc, deg=True)
    return magnitude_rc, phase_rc, np.abs(H_rc)


def rl_response(R, L, f_range):
    omega = 2 * np.pi * f_range
    H_rl = 1j * omega * L / (R + 1j * omega * L)
    magnitude_rl = 20 * np.log10(np.abs(H_rl))
    phase_rl = np.angle(H_rl, deg=True)
    return magnitude_rl, phase_rl, np.abs(H_rl)


# Основная функция построения графиков
def plot_signals(chain_type, R, C, L, f_min, f_max, num_points, amplitude, offset, time_scale):
    # Генерация частотного диапазона
    f_range = np.logspace(np.log10(f_min), np.log10(f_max), num_points)

    # Вычисление выходных сигналов в зависимости от типа цепи
    if chain_type == 'RC':
        magnitude, phase, H = rc_response(R, C, f_range)
    elif chain_type == 'RL':
        magnitude, phase, H = rl_response(R, L, f_range)

    # Моделирование входного и выходного сигналов
    input_signal = amplitude + offset
    output_signal = input_signal * H

    # Временной сигнал
    t = np.linspace(0, time_scale, num_points)
    input_time_signal = amplitude * np.sin(2 * np.pi * f_range[-1] * t) + offset
    output_time_signal = input_time_signal * np.abs(H[-1])

    # Построение графиков
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"АЧХ и ФЧХ для {chain_type} цепи", fontsize=16)

    # АЧХ с наложением входного и выходного сигналов
    axs[0, 0].semilogx(f_range, 20 * np.log10(input_signal) * np.ones_like(f_range), 'g--', label="Входной сигнал")
    axs[0, 0].semilogx(f_range, 20 * np.log10(output_signal), 'b', label="Выходной сигнал")
    axs[0, 0].grid(True, which="both", linestyle="--", linewidth=0.5)
    axs[0, 0].set_title("АЧХ", fontsize=12)
    axs[0, 0].set_xlabel("Частота (Гц)")
    axs[0, 0].set_ylabel("Амплитуда (dB)")
    axs[0, 0].legend()

    # ФЧХ
    axs[0, 1].semilogx(f_range, phase, 'r', label="ФЧХ")
    axs[0, 1].grid(True, which="both", linestyle="--", linewidth=0.5)
    axs[0, 1].set_title("ФЧХ", fontsize=12)
    axs[0, 1].set_xlabel("Частота (Гц)")
    axs[0, 1].set_ylabel("Фаза (градусы)")
    axs[0, 1].legend()

    # Временной график входного и выходного сигналов
    axs[1, 0].plot(t, input_time_signal, 'g--', label="Входной сигнал")
    axs[1, 0].plot(t, output_time_signal, 'b', label="Выходной сигнал")
    axs[1, 0].set_title("Временная область", fontsize=12)
    axs[1, 0].set_xlabel("Время (с)")
    axs[1, 0].set_ylabel("Амплитуда")
    axs[1, 0].legend()

    # Входной сигнал для текущей частоты
    axs[1, 1].plot(t, input_time_signal, 'purple', label="Входной сигнал")
    axs[1, 1].set_title("Входной сигнал на частоте генератора", fontsize=12)
    axs[1, 1].set_xlabel("Время (с)")
    axs[1, 1].set_ylabel("Амплитуда")
    axs[1, 1].legend()

    # Добавляем текстовые подписи с введёнными значениями
    textstr = '\n'.join((
        f"Амплитуда: {amplitude} В",
        f"Смещение: {offset} В",
        f"Частота генератора: {f_range[-1]:.2f} Гц",
        f"Временной масштаб: {time_scale} с",
        f"Сопротивление: {R} Ом",
        f"Емкость: {C} Ф",
        f"Индуктивность: {L} Гн"))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.75, 0.5, textstr, transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=props)

    # Кнопка для сохранения графиков
    saveax = plt.axes([0.8, 0.9, 0.1, 0.04])
    button = Button(saveax, 'Сохранить', color='lightgoldenrodyellow', hovercolor='0.975')

    def save_graph(event):
        fig.savefig(f'{chain_type}_response.png')
        print("График сохранен как изображение.")

    button.on_clicked(save_graph)

    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.show()


# Ввод параметров
chain_type = input("Введите тип цепи (RC или RL): ")
R = float(input("Введите значение сопротивления: "))
R_unit = input("Введите единицу сопротивления (Ом, кОм, МОм): ")
C = float(input("Введите значение емкости: "))
C_unit = input("Введите единицу емкости (Ф, мкФ, пФ): ")
L = float(input("Введите значение индуктивности: "))
L_unit = input("Введите единицу индуктивности (Гн, мГн, мкГн): ")
f_min = float(input("Введите минимальную частоту (Гц): "))
f_max = float(input("Введите максимальную частоту (Гц): "))
num_points = int(input("Введите количество точек для графиков: "))
amplitude = float(input("Введите амплитуду входного сигнала: "))
offset = float(input("Введите смещение входного сигнала: "))
time_scale = float(input("Введите временной масштаб для временного графика (в секундах): "))

# Конвертирование
R = convert_resistance(R, R_unit)
C = convert_capacitance(C, C_unit)
L = convert_inductance(L, L_unit)

# Построение
plot_signals(chain_type, R, C, L, f_min, f_max, num_points, amplitude, offset, time_scale)
