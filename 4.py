
# -*- coding: utf-8 -*-
"""
Решение заданий 1 и 2 для варианта 11
Задание 1: уравнение x * 2^x + x^2 - 5 = 0
Задание 2: система уравнений:
    x1^2 + 4*x2^2 - 4 = 0
    2*x2 - exp(x1) - x1 = 0
Методы:
- Для уравнения: дихотомия, простая итерация, Ньютон.
- Для системы: Ньютон, простая итерация, Зейдель.
Точность epsilon = 1e-3.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Задание 1: нелинейное уравнение
# ------------------------------------------------------------

def f1(x):
    """Исходная функция f(x) = x * 2^x + x^2 - 5"""
    return x * (2**x) + x**2 - 5

def f1_prime(x):
    """Производная f'(x) = 2^x (1 + x ln2) + 2x"""
    return (2**x) * (1 + x * np.log(2)) + 2*x

# ------------------- Метод дихотомии -------------------------
def dichotomy(f, a, b, eps=1e-3, max_iter=100):
    """
    Находит корень методом половинного деления.
    Возвращает (корень, количество итераций).
    """
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("Функция не меняет знак на отрезке [a,b]")
    it = 0
    while (b - a) / 2 > eps and it < max_iter:
        c = (a + b) / 2
        fc = f(c)
        if fc == 0:
            return c, it+1
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        it += 1
    return (a + b) / 2, it

# ------------------- Метод простой итерации ------------------
def simple_iteration(phi, x0, eps=1e-3, max_iter=100):
    """
    Метод простой итерации x_{n+1} = phi(x_n).
    Возвращает (корень, количество итераций).
    """
    x_prev = x0
    for it in range(max_iter):
        x_next = phi(x_prev)
        if abs(x_next - x_prev) < eps:
            return x_next, it+1
        x_prev = x_next
    return x_prev, max_iter

# Формы phi для разных корней
def phi_left(x):
    """Для левого корня: x = (5 - x^2) / 2^x"""
    return (5 - x**2) / (2**x)

def phi_right(x):
    """Для правого корня: x = log2((5 - x^2)/x)"""
    return np.log2((5 - x**2) / x)

# Производная phi_left для проверки сходимости
def phi_left_deriv(x):
    return (-2*x - (5 - x**2)*np.log(2)) / (2**x)

def phi_right_deriv(x):
    # phi_right'(x) = (1/ln2) * (-x^2-5)/(x(5-x^2))
    return (1/np.log(2)) * (-x**2 - 5) / (x * (5 - x**2))

# ------------------- Метод Ньютона ---------------------------
def newton(f, f_prime, x0, eps=1e-3, max_iter=100):
    """
    Метод Ньютона (касательных).
    Возвращает (корень, количество итераций).
    """
    x = x0
    for it in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if fpx == 0:
            raise ValueError("Нулевая производная в x = {:.6f}".format(x))
        x_new = x - fx / fpx
        if abs(x_new - x) < eps:
            return x_new, it+1
        x = x_new
    return x, max_iter

# ------------------------------------------------------------
# Задание 2: система нелинейных уравнений
# ------------------------------------------------------------

def F_sys(x):
    """Вектор-функция системы: F(x1,x2) = [F1, F2]"""
    x1, x2 = x
    F1 = x1**2 + 4*x2**2 - 4
    F2 = 2*x2 - np.exp(x1) - x1
    return np.array([F1, F2])

def J_sys(x):
    """Якобиан системы (2x2)"""
    x1, x2 = x
    J = np.array([
        [2*x1, 8*x2],
        [-np.exp(x1)-1, 2]
    ])
    return J

def newton_system(F, J, x0, eps=1e-3, max_iter=50):
    """
    Метод Ньютона для системы.
    Возвращает (решение, количество итераций).
    """
    x = np.array(x0, dtype=float)
    for it in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        try:
            delta = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            raise ValueError("Якобиан вырожден на итерации {}".format(it))
        x_new = x + delta
        if np.linalg.norm(x_new - x) < eps:
            return x_new, it+1
        x = x_new
    return x, max_iter

def phi_system(x):
    """
    Отображение для метода простой итерации:
    x1_new = -sqrt(4 - 4*x2^2)
    x2_new = (exp(x1) + x1)/2
    """
    x1, x2 = x
    x1_new = -np.sqrt(4 - 4*x2**2)
    x2_new = (np.exp(x1) + x1) / 2
    return np.array([x1_new, x2_new])

def simple_iter_system(phi, x0, eps=1e-3, max_iter=100):
    """
    Метод простой итерации для системы.
    Возвращает (решение, количество итераций).
    """
    x = np.array(x0, dtype=float)
    for it in range(max_iter):
        x_next = phi(x)
        if np.linalg.norm(x_next - x) < eps:
            return x_next, it+1
        x = x_next
    return x, max_iter

def seidel_system(x0, eps=1e-3, max_iter=100):
    """
    Метод Зейделя для системы (обновление переменных последовательно):
    x1_new = -sqrt(4 - 4*x2^2)
    x2_new = (exp(x1_new) + x1_new)/2
    """
    x = np.array(x0, dtype=float)
    for it in range(max_iter):
        x_new = x.copy()
        # первое уравнение
        x_new[0] = -np.sqrt(4 - 4*x[1]**2)
        # второе уравнение с использованием уже обновлённого x1
        x_new[1] = (np.exp(x_new[0]) + x_new[0]) / 2
        if np.linalg.norm(x_new - x) < eps:
            return x_new, it+1
        x = x_new
    return x, max_iter

# ------------------------------------------------------------
# Графический поиск начальных приближений
# ------------------------------------------------------------
def plot_functions():
    """Построение графиков для задания 1 и задания 2."""
    # График для уравнения
    x_vals = np.linspace(-3, 3, 500)
    y_vals = f1(x_vals)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.axhline(0, color='black', lw=0.8)
    plt.plot(x_vals, y_vals, 'b', label='$f(x)=x\\cdot2^x+x^2-5$')
    plt.grid(True)
    plt.title('Уравнение: $x\\cdot2^x+x^2-5=0$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    # График для системы
    x1_vals = np.linspace(-2.5, 2.5, 200)
    # Эллипс: x1^2 + 4*x2^2 = 4 -> x2 = ±sqrt((4-x1^2)/4)
    x2_ellipse_pos = np.sqrt((4 - x1_vals**2)/4)
    x2_ellipse_neg = -x2_ellipse_pos
    # Кривая: x2 = (exp(x1)+x1)/2
    x2_curve = (np.exp(x1_vals) + x1_vals) / 2

    plt.subplot(1,2,2)
    plt.plot(x1_vals, x2_ellipse_pos, 'g', label='Эллипс (верх)')
    plt.plot(x1_vals, x2_ellipse_neg, 'g--', label='Эллипс (низ)')
    plt.plot(x1_vals, x2_curve, 'r', label='$x_2 = (e^{x_1}+x_1)/2$')
    plt.grid(True)
    plt.title('Система уравнений')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Основная программа
# ------------------------------------------------------------
def main():
    print("="*60)
    print("ЗАДАНИЕ 1: Решение уравнения x * 2^x + x^2 - 5 = 0")
    print("="*60)

    # Построение графиков (необязательно, но наглядно)
    plot_functions()

    # Начальные отрезки и приближения (по графику)
    # Левый корень ~ -2.2, правый ~ 1.25
    a_left, b_left = -2.5, -2.0
    a_right, b_right = 1.0, 1.5

    print("\n--- 1.1 Метод дихотомии ---")
    root1, it1 = dichotomy(f1, a_left, b_left, eps=1e-3)
    print(f"Левый корень x1 = {root1:.6f}, итераций = {it1}")
    root2, it2 = dichotomy(f1, a_right, b_right, eps=1e-3)
    print(f"Правый корень x2 = {root2:.6f}, итераций = {it2}")

    print("\n--- 1.2 Метод простой итерации ---")
    # Левый корень: phi_left
    x0_left = -2.2
    # Проверка условия сходимости |phi'(x0)| < 1
    deriv_left = phi_left_deriv(x0_left)
    print(f"Для левого корня: φ'(x0) = {deriv_left:.5f} -> |φ'| < 1? {abs(deriv_left) < 1}")
    if abs(deriv_left) >= 1:
        print("   Внимание: условие сходимости в начальной точке не выполняется, но метод может сойтись.")
    root_left_iter, it_left_iter = simple_iteration(phi_left, x0_left, eps=1e-3)
    print(f"Левый корень x1 = {root_left_iter:.6f}, итераций = {it_left_iter}")

    # Правый корень: phi_right
    x0_right = 1.2
    deriv_right = phi_right_deriv(x0_right)
    print(f"Для правого корня: φ'(x0) = {deriv_right:.5f} -> |φ'| < 1? {abs(deriv_right) < 1}")
    if abs(deriv_right) >= 1:
        print("   Внимание: условие сходимости в начальной точке не выполняется, но метод может сойтись.")
    root_right_iter, it_right_iter = simple_iteration(phi_right, x0_right, eps=1e-3)
    print(f"Правый корень x2 = {root_right_iter:.6f}, итераций = {it_right_iter}")

    print("\n--- 1.3 Метод Ньютона ---")
    # Левый корень
    root1_newt, it1_newt = newton(f1, f1_prime, x0_left, eps=1e-3)
    print(f"Левый корень x1 = {root1_newt:.6f}, итераций = {it1_newt}")
    # Правый корень
    root2_newt, it2_newt = newton(f1, f1_prime, x0_right, eps=1e-3)
    print(f"Правый корень x2 = {root2_newt:.6f}, итераций = {it2_newt}")

    print("\n" + "="*60)
    print("ЗАДАНИЕ 2: Решение системы нелинейных уравнений")
    print("="*60)

    # Начальное приближение по графику: левая точка (-1.8, 0.2)
    x0_sys = np.array([-1.8, 0.2])

    print("\n--- 2.1 Метод Ньютона для системы ---")
    try:
        sol_newt, it_newt_sys = newton_system(F_sys, J_sys, x0_sys, eps=1e-3)
        print(f"Решение: x1 = {sol_newt[0]:.6f}, x2 = {sol_newt[1]:.6f}")
        print(f"Итераций: {it_newt_sys}")
        print(f"Невязка: F = {F_sys(sol_newt)}")
    except Exception as e:
        print(f"Ошибка в методе Ньютона: {e}")

    print("\n--- 2.2 Метод простой итерации для системы ---")
    # Проверка условия сходимости (спектральный радиус матрицы J_phi < 1)
    # Для начала вычислим J_phi в точке x0_sys
    x1, x2 = x0_sys
    dphi1_dx2 = 4*x2 / np.sqrt(4 - 4*x2**2)
    dphi2_dx1 = (np.exp(x1) + 1) / 2
    # Собственные значения: ±sqrt(dphi1_dx2 * dphi2_dx1)
    rho = np.sqrt(abs(dphi1_dx2 * dphi2_dx1))
    print(f"Оценка спектрального радиуса в начальной точке: {rho:.5f}")
    print(f"Условие сходимости (ρ<1) выполнено? {rho < 1}")

    try:
        sol_iter, it_iter_sys = simple_iter_system(phi_system, x0_sys, eps=1e-3)
        print(f"Решение: x1 = {sol_iter[0]:.6f}, x2 = {sol_iter[1]:.6f}")
        print(f"Итераций: {it_iter_sys}")
        print(f"Невязка: F = {F_sys(sol_iter)}")
    except Exception as e:
        print(f"Ошибка в методе простой итерации: {e}")

    print("\n--- 2.3 Метод Зейделя для системы (дополнительно) ---")
    try:
        sol_seid, it_seid_sys = seidel_system(x0_sys, eps=1e-3)
        print(f"Решение: x1 = {sol_seid[0]:.6f}, x2 = {sol_seid[1]:.6f}")
        print(f"Итераций: {it_seid_sys}")
        print(f"Невязка: F = {F_sys(sol_seid)}")
    except Exception as e:
        print(f"Ошибка в методе Зейделя: {e}")

    print("\n" + "="*60)
    print("Работа программы завершена.")
    print("="*60)

if __name__ == "__main__":
    main()
