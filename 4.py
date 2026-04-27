import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Задание 1. Нелинейное уравнение: x * 2**x + x**2 - 5 = 0
# ----------------------------
def f(x):
    """Исходная функция."""
    return x * (2 ** x) + x ** 2 - 5

def f_prime(x):
    """Производная: 2^x + x * 2^x * ln2 + 2x."""
    return 2 ** x + x * (2 ** x) * np.log(2) + 2 * x

# ------------------------------------------------------------
# Метод дихотомии
def bisection(a, b, eps, max_iter=100):
    """Возвращает корень, количество итераций и историю."""
    if f(a) * f(b) > 0:
        raise ValueError("На отрезке [a,b] нет корня или чётное количество корней.")
    iter_count = 0
    while (b - a) / 2 > eps and iter_count < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c, iter_count
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iter_count += 1
    return (a + b) / 2, iter_count

# ------------------------------------------------------------
# Метод Ньютона
def newton(x0, eps, max_iter=100):
    """Возвращает корень, количество итераций."""
    x = x0
    iter_count = 0
    while abs(f(x)) > eps and iter_count < max_iter:
        df = f_prime(x)
        if df == 0:
            break
        x = x - f(x) / df
        iter_count += 1
    return x, iter_count

# ------------------------------------------------------------
# Метод простой итерации (релаксация: x_{n+1} = x_n - lambda * f(x_n))
def simple_iteration(x0, lmbda, eps, max_iter=100):
    """
    Параметр lambda выбирается так, чтобы |1 - lambda * f'(x0)| < 1.
    Для корня около 1.5 берём lambda = 0.1, для корня около -2.5 lambda = -0.1.
    """
    x = x0
    iter_count = 0
    while abs(f(x)) > eps and iter_count < max_iter:
        x = x - lmbda * f(x)
        iter_count += 1
    return x, iter_count

# Определение отрезков и начальных приближений для двух корней
# Корень 1: около 1.5, отрезок [1,2]
# Корень 2: около -2.5, отрезок [-3,-2]
roots = [
    {"range": (1, 2), "x0_newton": 1.5, "x0_simple": 1.5, "lambda": 0.1, "desc": "корень x ≈ 1.5"},
    {"range": (-3, -2), "x0_newton": -2.5, "x0_simple": -2.5, "lambda": -0.1, "desc": "корень x ≈ -2.5"}
]

print("="*60)
print("Задание 1. Решение уравнения x * 2^x + x^2 - 5 = 0")
print("="*60)

for i, r in enumerate(roots, 1):
    print(f"\n--- {r['desc']} ---")

    # Дихотомия
    a, b = r["range"]
    root_bis, iter_bis = bisection(a, b, eps=1e-3)
    print(f"Метод дихотомии: корень = {root_bis:.6f}, f(корень) = {f(root_bis):.2e}, итераций = {iter_bis}")

    # Ньютон
    root_newt, iter_newt = newton(r["x0_newton"], eps=1e-3)
    print(f"Метод Ньютона: корень = {root_newt:.6f}, f(корень) = {f(root_newt):.2e}, итераций = {iter_newt}")

    # Простая итерация
    root_simple, iter_simple = simple_iteration(r["x0_simple"], r["lambda"], eps=1e-3)
    print(f"Метод простой итерации: корень = {root_simple:.6f}, f(корень) = {f(root_simple):.2e}, итераций = {iter_simple}")
    # Проверка условия сходимости
    fprime_at_root = f_prime(root_simple)
    mu = 1 - r["lambda"] * fprime_at_root
    print(f"  || Условие сходимости: |1 - λ·f'(x*)| = |{mu:.4f}| < 1? {'Да' if abs(mu) < 1 else 'Нет'}")

# ------------------------------------------------------------
# Задание 2. Система нелинейных уравнений
# { x1^2 + 4 x2^2 - 4 = 0
# { 2 x2 - exp(x1) - x1 = 0
# ------------------------------------------------------------
def F(x):
    """Вектор-функция системы."""
    x1, x2 = x
    return np.array([x1**2 + 4*x2**2 - 4,
                     2*x2 - np.exp(x1) - x1])

def J(x):
    """Матрица Якоби системы."""
    x1, x2 = x
    return np.array([[2*x1, 8*x2],
                     [-np.exp(x1)-1, 2]])

def newton_system(x0, eps, max_iter=50):
    """Метод Ньютона для системы."""
    x = np.array(x0, dtype=float)
    iter_count = 0
    while np.linalg.norm(F(x), np.inf) > eps and iter_count < max_iter:
        J_inv = np.linalg.inv(J(x))
        delta = J_inv @ F(x)
        x = x - delta
        iter_count += 1
    return x, iter_count

# Подбор начальных приближений графически (два решения)
# Первое решение: (x1, x2) ≈ (0.45, 0.98)
# Второе решение: (x1, x2) ≈ (-1.5, -0.65)
initial_guesses = [
    ([0.45, 0.98], "решение в правой полуплоскости (x1>0, x2>0)"),
    ([-1.5, -0.65], "решение в левой полуплоскости (x1<0, x2<0)")
]

print("\n" + "="*60)
print("Задание 2. Решение системы методом Ньютона")
print("="*60)

for i, (x0, desc) in enumerate(initial_guesses, 1):
    sol, it = newton_system(x0, eps=1e-3)
    print(f"\n--- {desc} ---")
    print(f"Начальное приближение: x1 = {x0[0]:.4f}, x2 = {x0[1]:.4f}")
    print(f"Решение: x1 = {sol[0]:.6f}, x2 = {sol[1]:.6f}")
    print(f"Невязка: F1 = {F(sol)[0]:.2e}, F2 = {F(sol)[1]:.2e}")
    print(f"Количество итераций: {it}")
    # Проверка обусловленности якобиана в найденной точке
    Jsol = J(sol)
    cond = np.linalg.cond(Jsol)
    print(f"Число обусловленности Якобиана в решении: {cond:.2f} (чем ближе к 1, тем лучше)")

# Построим графики для наглядного подтверждения (для системы)
plt.figure(figsize=(8,6))
# Кривая из первого уравнения: x2 = ± sqrt((4 - x1^2)/4)
x1_vals = np.linspace(-2.2, 2.2, 400)
x2_ellipse_pos = np.sqrt((4 - x1_vals**2) / 4)
x2_ellipse_neg = -x2_ellipse_pos
plt.plot(x1_vals, x2_ellipse_pos, 'b-', label=r'$x_1^2+4x_2^2=4$')
plt.plot(x1_vals, x2_ellipse_neg, 'b-')
# Кривая из второго уравнения: x2 = (exp(x1)+x1)/2
x2_curve = (np.exp(x1_vals) + x1_vals) / 2
plt.plot(x1_vals, x2_curve, 'r-', label=r'$2x_2 - e^{x_1} - x_1 =0$')
# Нанесём найденные решения
for sol, desc in zip([newton_system(g[0],1e-3)[0] for g in initial_guesses],
                     ["правый корень", "левый корень"]):
    plt.plot(sol[0], sol[1], 'ko', markersize=8)
    plt.annotate(desc, xy=(sol[0], sol[1]), xytext=(5,5), textcoords='offset points')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Графическое определение начальных приближений для системы')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()