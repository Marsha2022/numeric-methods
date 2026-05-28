import numpy as np
import matplotlib.pyplot as plt

# Аналитическое решение
def y_analytical(x):
    return (2/3) * (x**3 + 1) / (x**2 + 1)

# Правая часть ОДУ
def f(x, y):
    return 2 * x * (x - y) / (x**2 + 1)

# Метод Эйлера-Коши
def euler_cauchy(f, x0, y0, x_end, h):
    xs = np.arange(x0, x_end + h/2, h)
    ys = np.zeros_like(xs)
    ys[0] = y0
    for i in range(len(xs)-1):
        x_n = xs[i]
        y_n = ys[i]
        y_pred = y_n + h * f(x_n, y_n)
        ys[i+1] = y_n + h/2 * (f(x_n, y_n) + f(xs[i+1], y_pred))
    return xs, ys

# Метод Рунге-Кутты 4
def runge_kutta4(f, x0, y0, x_end, h):
    xs = np.arange(x0, x_end + h/2, h)
    ys = np.zeros_like(xs)
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

# Параметры
x0, y0, x_end = -2.0, -0.933, 4.0
steps = [0.2, 0.1, 0.05]

#----------------------------------------------------------------------
# 1. ДЕТАЛЬНЫЕ ТАБЛИЦЫ ДЛЯ КАЖДОГО ШАГА (сравнение методов)
# ----------------------------------------------------------------------
print("\n" + "="*100)
print("ПОДРОБНЫЕ ТАБЛИЦЫ ДЛЯ КАЖДОГО ШАГА")
print("="*100)

for h in steps:
    xs_euler, ys_euler = euler_cauchy(f, x0, y0, x_end, h)
    xs_rk4,   ys_rk4   = runge_kutta4(f, x0, y0, x_end, h)
    y_exact = y_analytical(xs_euler)   # сетка совпадает с сеткой Эйлера
    err_euler = np.abs(ys_euler - y_exact)
    err_rk4   = np.abs(ys_rk4   - y_exact)

    print(f"\nРезультаты для шага h = {h}")
    print("-----------------------------------------------------------")
    print("    x        y_точн        y_Эйлер-Коши    |ошибка|    y_РК4        |ошибка|")
    print("-----------------------------------------------------------")
    for i in range(len(xs_euler)):
        print(f"{xs_euler[i]:8.3f}  {y_exact[i]:10.6f}  {ys_euler[i]:12.6f}  {err_euler[i]:10.2e}  {ys_rk4[i]:10.6f}  {err_rk4[i]:10.2e}")
    print("-----------------------------------------------------------\n")

# ----------------------------------------------------------------------
# 2. ТАБЛИЦЫ СРАВНЕНИЯ ШАГОВ ДЛЯ КАЖДОГО МЕТОДА (выборочные x)
# ----------------------------------------------------------------------
x_display = np.arange(x0, x_end + 0.001, 0.4)   # от -2.0 до 4.0 с шагом 0.4

print("="*100)
print("ТАБЛИЦА 1. МЕТОД ЭЙЛЕРА-КОШИ – сравнение шагов")
print("="*100)
print(f"{'x':>8} | {'y_точн':>12} | {'h=0.2':>12} {'ошибка':>12} | {'h=0.1':>12} {'ошибка':>12} | {'h=0.05':>12} {'ошибка':>12}")
print("-"*100)

for x_val in x_display:
    y_ex = y_analytical(x_val)
    row = f"{x_val:8.2f} | {y_ex:12.6f}"
    for h in steps:
        xs, ys = euler_cauchy(f, x0, y0, x_end, h)
        idx = np.argmin(np.abs(xs - x_val))
        y_num = ys[idx]
        err = abs(y_num - y_ex)
        row += f" | {y_num:12.6f} {err:12.2e}"
    print(row)

print("\n" + "="*100)
print("ТАБЛИЦА 2. МЕТОД РУНГЕ-КУТТЫ 4 – сравнение шагов")
print("="*100)
print(f"{'x':>8} | {'y_точн':>12} | {'h=0.2':>12} {'ошибка':>12} | {'h=0.1':>12} {'ошибка':>12} | {'h=0.05':>12} {'ошибка':>12}")
print("-"*100)

for x_val in x_display:
    y_ex = y_analytical(x_val)
    row = f"{x_val:8.2f} | {y_ex:12.6f}"
    for h in steps:
        xs, ys = runge_kutta4(f, x0, y0, x_end, h)
        idx = np.argmin(np.abs(xs - x_val))
        y_num = ys[idx]
        err = abs(y_num - y_ex)
        row += f" | {y_num:12.6f} {err:12.2e}"
    print(row)

# ----------------------------------------------------------------------
# 1. Графики сравнения решений для каждого шага
# ----------------------------------------------------------------------
fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, h in enumerate(steps):
    xs, ys_euler = euler_cauchy(f, x0, y0, x_end, h)
    _, ys_rk4 = runge_kutta4(f, x0, y0, x_end, h)
    y_ex = y_analytical(xs)

    ax = axes[idx]
    ax.plot(xs, y_ex, 'k-', linewidth=2, label='Аналитическое')
    ax.plot(xs, ys_euler, 'ro--', markersize=3, label='Эйлер-Коши')
    ax.plot(xs, ys_rk4, 'bs--', markersize=3, label='РК4')
    ax.set_title(f'Сравнение решений, h = {h}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 2. Графики абсолютной погрешности по x для каждого шага
# ----------------------------------------------------------------------
fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, h in enumerate(steps):
    xs, ys_euler = euler_cauchy(f, x0, y0, x_end, h)
    _, ys_rk4 = runge_kutta4(f, x0, y0, x_end, h)
    y_ex = y_analytical(xs)

    err_euler = np.abs(ys_euler - y_ex)
    err_rk4   = np.abs(ys_rk4 - y_ex)

    ax = axes[idx]
    ax.semilogy(xs, err_euler, 'ro-', label='Эйлер-Коши')
    ax.semilogy(xs, err_rk4, 'bs-', label='РК4')
    ax.set_title(f'Погрешность, h = {h}')
    ax.set_xlabel('x')
    ax.set_ylabel('Абсолютная ошибка (лог. шкала)')
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 3. Зависимость максимальной ошибки от шага (сходимость)
# ----------------------------------------------------------------------
errors_euler_max = []
errors_rk4_max = []
for h in steps:
    _, ys_euler = euler_cauchy(f, x0, y0, x_end, h)
    _, ys_rk4 = runge_kutta4(f, x0, y0, x_end, h)
    xs = np.arange(x0, x_end + h/2, h)
    y_ex = y_analytical(xs)
    err_euler_max = np.max(np.abs(ys_euler - y_ex))
    err_rk4_max   = np.max(np.abs(ys_rk4 - y_ex))
    errors_euler_max.append(err_euler_max)
    errors_rk4_max.append(err_rk4_max)

    print(f"Шаг h = {h}:")
    print(f"  Макс. ошибка Эйлера-Коши = {err_euler_max:.2e}")
    print(f"  Макс. ошибка РК4         = {err_rk4_max:.2e}")

# График сходимости в лог-логарифмических осях
plt.figure(figsize=(8,5))
plt.loglog(steps, errors_euler_max, 'ro-', linewidth=2, markersize=8, label='Эйлер-Коши')
plt.loglog(steps, errors_rk4_max, 'bs-', linewidth=2, markersize=8, label='Рунге-Кутта 4')
# Теоретические порядки: для сравнения проведём линии с наклоном 1 и 4
steps_theor = np.array([0.05, 0.2])
plt.loglog(steps_theor, errors_euler_max[0] * (steps_theor/steps[0])**1, 'r--', alpha=0.5, label='Порядок 1 (теор.)')
plt.loglog(steps_theor, errors_rk4_max[0] * (steps_theor/steps[0])**4, 'b--', alpha=0.5, label='Порядок 4 (теор.)')
plt.xlabel('Шаг h')
plt.ylabel('Максимальная абсолютная ошибка')
plt.title('Сходимость численных методов')
plt.legend()
plt.grid(True)
plt.show()
