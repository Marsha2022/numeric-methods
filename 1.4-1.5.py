import numpy as np

# ============================================================================
# Задание 7.1. Собственные значения симметричной матрицы (вариант 11)
# ============================================================================

eps_eig = 1e-4   # точность для методов

# Симметричная матрица 5x5 из таблицы 4, вариант 11
A_sym = np.array([
    [-3, -5, -4,  0, -3],
    [-5,  7,  1,  2,  2],
    [-4,  1, -1,  6,  5],
    [ 0,  2,  6,  1,  0],
    [-3,  2,  5,  0, -2]
], dtype=float)

print("Задание 7.1. Собственные значения симметричной матрицы (вариант 11)")
print("Входная матрица A (симметричная):")
print(A_sym)
print()

# ---------- 1. Метод вращения Якоби ----------
def jacobi_eigen(A, eps=1e-4, max_iter=1000):
    n = A.shape[0]
    for k in range(max_iter):
        max_val = 0
        p, q = 0, 1
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j
        if max_val < eps:
            return np.diag(A), k+1
        if A[p, p] == A[q, q]:
            theta = np.pi/4
        else:
            theta = 0.5 * np.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))
        c = np.cos(theta)
        s = np.sin(theta)
        new_pp = c**2 * A[p, p] + s**2 * A[q, q] - 2 * c * s * A[p, q]
        new_qq = s**2 * A[p, p] + c**2 * A[q, q] + 2 * c * s * A[p, q]
        old_p = A[p, :].copy()
        old_q = A[q, :].copy()
        A[p, p] = new_pp
        A[q, q] = new_qq
        A[p, q] = 0.0
        A[q, p] = 0.0
        for i in range(n):
            if i != p and i != q:
                A[i, p] = c * old_p[i] - s * old_q[i]
                A[p, i] = A[i, p]
                A[i, q] = s * old_p[i] + c * old_q[i]
                A[q, i] = A[i, q]
    return np.diag(A), max_iter

print("Метод вращения Якоби:")
eig_jacobi, iter_jacobi = jacobi_eigen(A_sym.copy(), eps_eig)
print("Собственные значения (диагональ после преобразований):")
print(eig_jacobi)
print("Количество итераций:", iter_jacobi)
print()

# ---------- 2. QR-алгоритм ----------
def householder_qr(A):
    """Возвращает Q и R для матрицы A методом Хаусхолдера."""
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)

    for k in range(min(m, n)):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)

        if norm_x == 0 or len(x) == 1:
            continue

        if x[0] >= 0:
            u = x + norm_x * np.eye(len(x))[:, 0]
        else:
            u = x - norm_x * np.eye(len(x))[:, 0]

        u_norm = np.linalg.norm(u)
        if u_norm > 1e-12:
            v = u / u_norm
        else:
            v = u

        for j in range(k, n):
            dot = 2 * np.dot(v, R[k:, j])
            R[k:, j] -= dot * v

        for j in range(m):
            dot = 2 * np.dot(v, Q[k:, j])
            Q[k:, j] -= dot * v

    return Q.T, R   # Q - ортогональная, R - верхнетреугольная

def qr_algorithm_symmetric(A, epsilon=1e-4, max_iter=1000):
    """QR-алгоритм для симметричной матрицы."""
    n = A.shape[0]
    A_k = A.copy().astype(float)

    print("=" * 70)
    print("QR-АЛГОРИТМ ДЛЯ ПОИСКА СОБСТВЕННЫХ ЗНАЧЕНИЙ (СИММЕТРИЧНАЯ МАТРИЦА)")
    print("=" * 70)

    trace_A = np.trace(A)
    det_A = np.linalg.det(A)
    print(f"\nАРИФМЕТИЧЕСКАЯ ПРОВЕРКА ИСХОДНОЙ МАТРИЦЫ:")
    print(f"След матрицы A: {trace_A:.6f}")
    print(f"Определитель матрицы A: {det_A:.6f}")

    print(f"\nТИП МАТРИЦЫ: СИММЕТРИЧНАЯ (только вещественные корни)")
    print(f"Критерий сходимости: максимальный поддиагональный элемент < {epsilon}")

    for iteration in range(max_iter):
        Q, R = householder_qr(A_k)
        A_next = R @ Q

        max_subdiag = 0.0
        for i in range(n):
            for j in range(i):
                max_subdiag = max(max_subdiag, abs(A_next[i, j]))

        if iteration % 10 == 0:
            print(f"Итерация {iteration:4d}, max поддиаг = {max_subdiag:.6f}")

        if max_subdiag < epsilon:
            print(f"\nСОШЛОСЬ на итерации {iteration} (диагональная форма)")
            final_matrix = A_next
            break

        A_k = A_next
    else:
        print(f"\nДостигнуто максимальное число итераций ({max_iter})")
        final_matrix = A_k

    return iteration, final_matrix

# Запуск QR-алгоритма
iterations, final_matrix = qr_algorithm_symmetric(A_sym.copy(), epsilon=eps_eig)

print(f"\nКоличество итераций: {iterations}")
print("\nФинальная матрица (почти диагональная):")
for row in final_matrix:
    print(" ".join(f"{val:10.6f}" for val in row))

eigenvalues_qr = np.diag(final_matrix)
print("\nСОБСТВЕННЫЕ ЗНАЧЕНИЯ (диагональ финальной матрицы):")
for i, val in enumerate(eigenvalues_qr):
    print(f"λ{i+1} = {val:.6f}")

# Проверка через след и определитель
print("\n" + "=" * 70)
print("ПРОВЕРКА СОБСТВЕННЫХ ЗНАЧЕНИЙ (QR-алгоритм)")
print("=" * 70)

trace_A = np.trace(A_sym)
sum_eigenvals = np.sum(eigenvalues_qr)
print(f"\n1. ПРОВЕРКА ЧЕРЕЗ СЛЕД МАТРИЦЫ:")
print(f"   След матрицы A: {trace_A:.6f}")
print(f"   Сумма собственных значений: {sum_eigenvals:.6f}")
print(f"   Разница: {abs(trace_A - sum_eigenvals):.2e}")

det_A = np.linalg.det(A_sym)
prod_eigenvals = np.prod(eigenvalues_qr)
print(f"\n2. ПРОВЕРКА ЧЕРЕЗ ОПРЕДЕЛИТЕЛЬ:")
print(f"   Определитель матрицы A: {det_A:.6f}")
print(f"   Произведение собственных значений: {prod_eigenvals:.6f}")
print(f"   Разница: {abs(det_A - prod_eigenvals):.2e}")

# ========== ДОБАВЛЕННЫЙ БЛОК: ВЫВОД ОРТОГОНАЛЬНОЙ МАТРИЦЫ QR-РАЗЛОЖЕНИЯ ==========
print("\n" + "=" * 70)
print("QR-РАЗЛОЖЕНИЕ ИСХОДНОЙ МАТРИЦЫ (МЕТОД ХАУСХОЛДЕРА)")
print("=" * 70)

Q, R = householder_qr(A_sym.copy())

print("\nОРТОГОНАЛЬНАЯ МАТРИЦА Q:")
print(Q)
print("\nВЕРХНЕТРЕУГОЛЬНАЯ МАТРИЦА R:")
print(R)

# Проверка: A = Q * R
A_reconstructed = Q @ R
print("\nПРОВЕРКА: Q * R (должно быть равно исходной A):")
print(A_reconstructed)
print("\nМаксимальная разница |A - Q*R|:", np.max(np.abs(A_sym - A_reconstructed)))

# Проверка ортогональности Q: Q^T * Q = I
QTQ = Q.T @ Q
print("\nПРОВЕРКА ОРТОГОНАЛЬНОСТИ Q (Q^T * Q):")
print(QTQ)
print("Максимальное отклонение от единичной матрицы:", np.max(np.abs(QTQ - np.eye(Q.shape[0]))))
# ============================================================================

# Сравнение с numpy
numpy_vals = np.linalg.eigvalsh(A_sym)
print("\n" + "=" * 70)
print("СРАВНЕНИЕ С NUMPY (ДЛЯ ПРОВЕРКИ):")
print("=" * 70)
print("Собственные значения от numpy (eigvalsh):")
for i, val in enumerate(numpy_vals):
    print(f"λ{i+1} = {val:.6f}")

print("\nМаксимальная разница (QR vs numpy):",
      np.max(np.abs(np.sort(eigenvalues_qr) - np.sort(numpy_vals))))
