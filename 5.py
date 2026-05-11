import numpy as np
import matplotlib.pyplot as plt

# Аналитическое решение
def y_analytical(x):
    return (2/3) * (x**3 + 1) / (x**2 + 1)

# Правая часть ОДУ
def f(x, y):
    return 2 * x * (x - y) / (x**2 + 1)

# Метод Эйлера-Коши (предиктор-корректор)
def euler_cauchy(f, x0, y0, x_end, h):
    xs = np.arange(x0, x_end + h/2, h)   # сетка
    ys = np.zeros(len(xs))
    ys[0] = y0
    for i in range(len(xs)-1):
        x_n = xs[i]
        y_n = ys[i]
        # предиктор
        y_pred = y_n + h * f(x_n, y_n)
        # корректор
        ys[i+1] = y_n + h/2 * (f(x_n, y_n) + f(xs[i+1], y_pred))
    return xs, ys

# Метод Рунге-Кутты 4-го порядка
def runge_kutta4(f, x0, y0, x_end, h):
    xs = np.arange(x0, x_end + h/2, h)
    ys = np.zeros(len(xs))
    ys[0] = y0
    for i in range(len(xs)-1):
        x_n = xs[i]
        y_n = ys[i]
        k1 = f(x_n, y_n)
        k2 = f(x_n + h/2, y_n + h*k1/2)
        k3 = f(x_n + h/2, y_n + h*k2/2)
        k4 = f(x_n + h, y_n + h*k3)
        ys[i+1] = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return xs, ys

# Параметры задачи
x0 = -2.0
y0 = -0.933
x_end = 4.0
h = 0.2

# Вычисление решений
xs_euler, ys_euler = euler_cauchy(f, x0, y0, x_end, h)
xs_rk4, ys_rk4 = runge_kutta4(f, x0, y0, x_end, h)

# Точное решение на той же сетке
y_exact = y_analytical(xs_euler)

# Ошибки
err_euler = np.abs(ys_euler - y_exact)
err_rk4 = np.abs(ys_rk4 - y_exact)

# Вывод таблицы для h=0.2
print("Результаты для шага h = 0.2")
print("-----------------------------------------------------------")
print("    x        y_точн        y_Эйлер-Коши    |ошибка|    y_РК4        |ошибка|")
print("-----------------------------------------------------------")
for i in range(len(xs_euler)):
    print(f"{xs_euler[i]:8.3f}  {y_exact[i]:10.6f}  {ys_euler[i]:12.6f}  {err_euler[i]:10.2e}  {ys_rk4[i]:10.6f}  {err_rk4[i]:10.2e}")

# Графики для h=0.2
plt.figure(figsize=(10,5))
plt.plot(xs_euler, y_exact, 'k-', linewidth=2, label='Аналитическое решение')
plt.plot(xs_euler, ys_euler, 'ro--', markersize=4, label='Эйлер-Коши (h=0.2)')
plt.plot(xs_rk4, ys_rk4, 'bs--', markersize=4, label='Рунге-Кутта 4 (h=0.2)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сравнение численных и аналитического решений')
plt.legend()
plt.grid(True)
plt.show()

# Исследование влияния шага
steps = [0.2, 0.1, 0.05]
errors_euler = []
errors_rk4 = []

for h_test in steps:
    _, ys_euler_test = euler_cauchy(f, x0, y0, x_end, h_test)
    _, ys_rk4_test = runge_kutta4(f, x0, y0, x_end, h_test)
    xs_test = np.arange(x0, x_end + h_test/2, h_test)
    y_exact_test = y_analytical(xs_test)
    err_euler_max = np.max(np.abs(ys_euler_test - y_exact_test))
    err_rk4_max = np.max(np.abs(ys_rk4_test - y_exact_test))
    errors_euler.append(err_euler_max)
    errors_rk4.append(err_rk4_max)
    print(f"\nШаг h = {h_test}:")
    print(f"  Максимальная ошибка Эйлера-Коши = {err_euler_max:.2e}")
    print(f"  Максимальная ошибка Рунге-Кутты 4 = {err_rk4_max:.2e}")

# Построение графика зависимости ошибки от шага
plt.figure(figsize=(8,5))
plt.loglog(steps, errors_euler, 'ro-', label='Эйлер-Коши')
plt.loglog(steps, errors_rk4, 'bs-', label='Рунге-Кутта 4')
plt.xlabel('Шаг h')
plt.ylabel('Максимальная абсолютная ошибка')
plt.title('Влияние шага интегрирования на погрешность')
plt.legend()
plt.grid(True)
plt.show()