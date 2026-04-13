import numpy as np

# Подынтегральная функция
def f(x):
    return x**3 / (x**2 + 1)**(3/2)

# Параметры интеграла
a, b = 0.0, 1.0
eps = 10**(-3)

# --- Метод средних прямоугольников ---
def rect_mid(f, a, b, n):
    h = (b - a) / n
    x_mid = a + h/2 + np.arange(n) * h
    return h * np.sum(f(x_mid))

# --- Метод трапеций ---
def trapez(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    return h * (0.5*f(a) + np.sum(f(x[1:-1])) + 0.5*f(b))

# --- Метод Симпсона (n чётное) ---
def simpson(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Для метода Симпсона n должно быть чётным")
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    s = f(a) + f(b)
    s += 4 * np.sum(f(x[1:-1:2]))   # нечётные индексы
    s += 2 * np.sum(f(x[2:-2:2]))   # чётные индексы (кроме концов)
    return h/3 * s

# --- Функция уточнения по Рунге-Ромбергу ---
def runge_romberg(method, f, a, b, n_start, p, eps):
    n = n_start
    I_prev = method(f, a, b, n)
    while True:
        n *= 2
        I_curr = method(f, a, b, n)
        delta = (I_curr - I_prev) / (2**p - 1)
        if abs(delta) < eps:
            I_refined = I_curr + delta
            return I_curr, I_refined, n//2, n, abs(delta)
        I_prev = I_curr

# --- Вычисление ---
print("="*60)
print("Вычисление интеграла с точностью ε = 10**(-3)")
print("Вариант 11: ∫₀¹ x³/(x²+1)^(3/2) dx")
print("="*60)

# Прямоугольники (средние)
I_rect, I_rect_ref, n1, n2, err_rect = runge_romberg(rect_mid, f, a, b, 4, 2, eps)
print("\nМетод средних прямоугольников (порядок 2):")
print(f"  Приближение на n={n2}: I = {I_rect:.8f}")
print(f"  Оценка погрешности: {err_rect:.2e}")
print(f"  Уточнённое по Рунге–Ромбергу: I = {I_rect_ref:.8f}")

# Трапеции
I_trap, I_trap_ref, n1, n2, err_trap = runge_romberg(trapez, f, a, b, 4, 2, eps)
print("\nМетод трапеций (порядок 2):")
print(f"  Приближение на n={n2}: I = {I_trap:.8f}")
print(f"  Оценка погрешности: {err_trap:.2e}")
print(f"  Уточнённое по Рунге–Ромбергу: I = {I_trap_ref:.8f}")

# Симпсон
I_simp, I_simp_ref, n1, n2, err_simp = runge_romberg(simpson, f, a, b, 4, 4, eps)
print("\nМетод Симпсона (порядок 4):")
print(f"  Приближение на n={n2}: I = {I_simp:.8f}")
print(f"  Оценка погрешности: {err_simp:.2e}")
print(f"  Уточнённое по Рунге–Ромбергу: I = {I_simp_ref:.8f}")

# Эталонное значение для проверки (очень мелкий шаг)
n_exact = 2**16
I_exact = simpson(f, a, b, n_exact)
print("\nВысокоточное эталонное значение (Симпсон, n=65536):")
print(f"  I_exact = {I_exact:.10f}")

print("\nОтклонения уточнённых значений от эталона:")
print(f"  Прямоугольники: {I_rect_ref - I_exact:.2e}")
print(f"  Трапеции:       {I_trap_ref - I_exact:.2e}")
print(f"  Симпсон:        {I_simp_ref - I_exact:.2e}")