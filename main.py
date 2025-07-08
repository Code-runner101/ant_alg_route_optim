import numpy as np
import matplotlib.pyplot as plt
import time
import openrouteservice
import mplcursors
from geopy.distance import geodesic
import os
from collections import defaultdict

API_KEY = '<///>'
client = openrouteservice.Client(key=API_KEY)

# --- Настройки алгоритма ---
FILENAME = "cities_test.txt"  # Имя файла с координатами городов
MATRIX_FILE = "full_distance_matrix.npy"
DURATION_MATRIX = "full_duration_matrix.npy"
N_ANTS = 200  # Количество муравьев
N_ITERATIONS = 50  # Количество поколений
ALPHA = 1.0
BETA  = 2.0
EVAPORATION_RATE = 0.1
PHEROMONE_CONSTANT = 100.0


# --- Шаг 1: Загрузка данных ---
def load_coordinates(filename):
    names = []
    coords = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                names.append(parts[0])
                coords.append([float(parts[1]), float(parts[2])])
    return names, np.array(coords)


# --- Шаг 2: Вычисление матрицы расстояний большого кол-ва городов (с кэшированием)---
def calculate_distance_matrix(coords, chunk_size=50, cache_dir="matrix_cache4", return_durations=True):
    n = len(coords)
    all_coords = [(lon, lat) for lat, lon in coords]

    os.makedirs(cache_dir, exist_ok=True)

    full_distance_matrix = np.full((n, n), np.inf)
    full_duration_matrix = np.full((n, n), np.inf) if return_durations else None

    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            from_idx = list(range(i, min(i + chunk_size, n)))
            to_idx = list(range(j, min(j + chunk_size, n)))
            block_filename = os.path.join(cache_dir, f"block_{i}_{j}.npy")

            if os.path.exists(block_filename):
                distances = np.load(block_filename)
                full_distance_matrix[i:i + len(from_idx), j:j + len(to_idx)] = distances
                if return_durations:
                    duration_filename = block_filename.replace("block_", "duration_")
                    if os.path.exists(duration_filename):
                        durations = np.load(duration_filename)
                        full_duration_matrix[i:i + len(from_idx), j:j + len(to_idx)] = durations
                continue

            unique_indices = sorted(set(from_idx + to_idx))
            sub_locations = [all_coords[k] for k in unique_indices]
            index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(unique_indices)}
            sources = [index_map[k] for k in from_idx]
            destinations = [index_map[k] for k in to_idx]

            try:
                matrix = client.distance_matrix(
                    locations=sub_locations,
                    profile='driving-car',
                    metrics=['distance', 'duration'] if return_durations else ['distance'],
                    units='km',
                    sources=sources,
                    destinations=destinations
                )
                distances = np.array(matrix['distances'])
                np.save(block_filename, distances)
                full_distance_matrix[i:i + len(from_idx), j:j + len(to_idx)] = distances

                if return_durations:
                    durations = np.array(matrix['durations'])
                    duration_filename = block_filename.replace("block_", "duration_")
                    np.save(duration_filename, durations)
                    full_duration_matrix[i:i + len(from_idx), j:j + len(to_idx)] = durations

                print(f"Блок ({i}:{i + len(from_idx)}, {j}:{j + len(to_idx)}) — успешно сохранён.")
            except Exception as e:
                print(f"❌ Ошибка при обработке блока ({i}:{i+chunk_size}, {j}:{j+chunk_size}): {e}")
                time.sleep(5)

            time.sleep(3)

    if return_durations:
        return full_distance_matrix, full_duration_matrix
    return full_distance_matrix

# --- Шаг 3: Реализация муравьиного алгоритма ---
def ant_colony_optimization(
    dist_matrix, n_ants, n_iterations, alpha, beta, evaporation_rate, pheromone_constant, start_city, end_city
):
    n_cities = len(dist_matrix)
    pheromone = np.ones((n_cities, n_cities))
    best_route = None
    best_distance = float('inf')
    all_best_routes = []

    for iteration in range(n_iterations):
        routes = []
        route_lengths = []

        for ant in range(n_ants):
            visited = np.zeros(n_cities, dtype=bool)
            visited[start_city] = True
            current_city = start_city
            route = [current_city]
            total_distance = 0

            while current_city != end_city:
                probabilities = calculate_transition_probabilities(
                    current_city, visited, pheromone, dist_matrix, alpha, beta
                )

                # Если нет доступных переходов, прерываем маршрут
                if np.all(probabilities == 0):
                    route = []
                    total_distance = float('inf')
                    break

                next_city = np.random.choice(range(n_cities), p=probabilities)
                route.append(next_city)
                total_distance += dist_matrix[current_city, next_city]
                current_city = next_city
                visited[current_city] = True

            if route:  # только если маршрут завершился
                route_lengths.append(total_distance)
                routes.append(route)

        # Обновление феромонов
        pheromone *= (1 - evaporation_rate)
        for i, route in enumerate(routes):
            for j in range(len(route) - 1):
                pheromone[route[j], route[j + 1]] += pheromone_constant / route_lengths[i]

        # Обновление лучшего маршрута
        if route_lengths:
            min_length = min(route_lengths)
            if min_length < best_distance:
                best_distance = min_length
                best_route = routes[route_lengths.index(min_length)]

        print(f"Поколение {iteration + 1}: собрано {len(routes)} завершённых маршрутов")
        all_best_routes.append((best_route, best_distance))

    return best_route, best_distance, all_best_routes


# --- Шаг 4: Вероятности переходов между городами ---
def calculate_transition_probabilities(current_city, visited, pheromone, dist_matrix, alpha, beta):
    n = len(visited)
    probs = np.zeros(n, dtype=float)
    for j in range(n):
        if not visited[j] and not np.isinf(dist_matrix[current_city, j]) and dist_matrix[current_city, j] > 0:
            pher = pheromone[current_city, j] ** alpha
            eta = (1.0 / dist_matrix[current_city, j]) ** beta
            probs[j] = pher * eta

    total = probs.sum()
    if total <= 0 or np.isnan(total):
        # ни одного доступного перехода
        return probs  # всё равно нули
    return probs / total


def visualize_routes(coords, all_best_routes, final_best_route, city_names):
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))

    for i, (route, distance) in enumerate(all_best_routes):
        route_coords = coords[route]
        axes[0].plot(route_coords[:, 1], route_coords[:, 0], marker='o', linestyle='-', label=f"Gen {i + 1}")

    axes[0].set_title("Лучшие маршруты по поколениям")
    axes[0].set_xlabel("Широта")
    axes[0].set_ylabel("Долгота")

    scatter_left = axes[0].scatter(coords[:, 1], coords[:, 0], color='blue', alpha=0)
    cursor_left = mplcursors.cursor(scatter_left, hover=True)
    cursor_left.connect("add", lambda sel: sel.annotation.set_text(city_names[sel.index]))

    final_route_coords = coords[final_best_route]
    axes[1].plot(final_route_coords[:, 1], final_route_coords[:, 0], marker='o', color='red', linestyle='-')

    axes[1].set_title("Лучший маршрут среди всех поколений")
    axes[1].set_xlabel("Широта")
    axes[1].set_ylabel("Долгота")

    scatter_right = axes[1].scatter(coords[:, 1], coords[:, 0], color='red', alpha=0)
    cursor_right = mplcursors.cursor(scatter_right, hover=True)
    cursor_right.connect("add", lambda sel: sel.annotation.set_text(city_names[sel.index]))

    plt.tight_layout()
    plt.show()


def get_full_route_polyline(route_indices, coordinates, client):
    coords = [(coordinates[i][1], coordinates[i][0]) for i in route_indices]
    try:
        route = client.directions(
            coordinates=coords,
            profile='driving-car',
            format='geojson'
        )
        return route['features'][0]['geometry']['coordinates']
    except Exception as e:
        print("Ошибка при получении полного маршрута:", e)
        return []


def get_cities_along_polyline(polyline_coords, coordinates, city_names, threshold_km=10):
    passed = []
    seen = set()
    for lon, lat in polyline_coords[::10]:
        for i, (city_lat, city_lon) in enumerate(coordinates):
            if i in seen:
                continue
            if geodesic((lat, lon), (city_lat, city_lon)).km <= threshold_km:
                passed.append(i)
                seen.add(i)
    return passed


def plot_real_route_polyline(polyline_coords, coordinates, city_names):
    lats = [lat for lon, lat in polyline_coords]
    lons = [lon for lon, lat in polyline_coords]
    plt.figure(figsize=(12, 8))
    plt.plot(lons, lats, color='purple', linewidth=2)

    # Отображение только городов, попавших в маршрут
    passed_city_indices = get_cities_along_polyline(polyline_coords, coordinates, city_names)
    for i in passed_city_indices:
        lat, lon = coordinates[i]
        plt.scatter(lon, lat, color='green')
        plt.text(lon + 0.1, lat + 0.1, city_names[i], fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    plt.title("Реальный маршрут по данным")
    plt.xlabel("Долгота")
    plt.ylabel("Широта")
    plt.grid(True)
    plt.show()


# Загрузка данных и запуск алгоритма
city_names, coordinates = load_coordinates(FILENAME)

coord_map = defaultdict(list)
for idx, coord in enumerate(coordinates):
    coord_key = (round(coord[0], 6), round(coord[1], 6))
    coord_map[coord_key].append(idx)

duplicates = {k: v for k, v in coord_map.items() if len(v) > 1}
if duplicates:
    print("Найдены дубли координат:")
    for coord, idxs in duplicates.items():
        names = [city_names[i] for i in idxs]
        print(f"Координаты {coord} — города {names} (индексы {idxs})")
else:
    print("Дубликатов координат не найдено.")

# Для кэшированной матрицы
# Загрузка или расчёт матрицы расстояний
if os.path.exists(MATRIX_FILE) and os.path.exists(DURATION_MATRIX):
    print("✅ Загружаем кэшированную матрицу расстояний и времени...")
    distance_matrix = np.load(MATRIX_FILE)
    duration_matrix = np.load(DURATION_MATRIX)
else:
    print("🧮 Матрица не найдена — запускаем расчёт...")
    distance_matrix, duration_matrix = calculate_distance_matrix(coordinates)
    np.save(MATRIX_FILE, distance_matrix)
    np.save(DURATION_MATRIX, duration_matrix)
    print("💾 Матрица сохранена в", MATRIX_FILE)

start_city = city_names.index("Санкт-Петербург")
end_city = city_names.index("Яраг-Казмаляр")

print("Общее число недостижимых ячеек:", np.sum(np.isinf(distance_matrix)))

start_time = time.time()  # Начало замера времени

# Для кэшированной матрицы
best_route, best_distance, all_best_routes = ant_colony_optimization(
    distance_matrix,
    N_ANTS,
    N_ITERATIONS,
    ALPHA,
    BETA,
    EVAPORATION_RATE,
    PHEROMONE_CONSTANT,
    start_city,
    end_city
)

end_time = time.time()  # Конец замера времени

# Вычисление затраченного времени
elapsed_time = end_time - start_time


print(f"\n\nДлина найденного маршрута: {best_distance:.2f} км\n")

# Преобразуем маршрут в индексы для ORS
expanded_route_indices = best_route
real_polyline = get_full_route_polyline(expanded_route_indices, coordinates, client)

if real_polyline:
    cities_along_polyline = get_cities_along_polyline(real_polyline, coordinates, city_names)
    print("🚗 Реальный маршрут по дорогам с промежуточными городами:")
    print(" → ".join([city_names[i] for i in cities_along_polyline]))

# Время в пути
total_duration = sum(duration_matrix[best_route[i], best_route[i + 1]] for i in range(len(best_route) - 1))
duration_hours = int(total_duration // 3600)
duration_minutes = int((total_duration % 3600) // 60)
print(f"⏱️ Примерное время в пути: {duration_hours} ч {duration_minutes} мин")

# Визуализация поколений
visualize_routes(coordinates, all_best_routes, best_route, city_names)
plot_real_route_polyline(real_polyline, coordinates, city_names)

