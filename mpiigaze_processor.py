import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class MPIIGazeProcessor:
    """
    Класс для обработки данных MPIIGaze по требованиям конкурсного задания.
    """
    
    def __init__(self, base_path='MPIIGaze'):
        self.base_path = base_path
        self.processed_data = None
        self.report_content = []
        
    def load_and_clean_data(self, participant='p00', day='day01'):
        """Загрузка и очистка данных (критерий А1)"""
        print("[Шаг 1/6] Загрузка данных...")
        
        annotation_path = os.path.join(
            self.base_path, 'Data', 'Original', participant, day, 'annotation.txt'
        )
        
        if not os.path.exists(annotation_path):
            print("Ошибка: файл аннотаций не найден")
            return None
        
        # Загрузка данных
        annotations = np.loadtxt(annotation_path)
        
        # Создание DataFrame с понятными названиями колонок
        columns = [
            *[f'eye_landmark_{i}' for i in range(24)],  # 24 координаты ориентиров глаз
            'screen_gaze_x', 'screen_gaze_y',           # Координаты взгляда на экране
            *[f'gaze_3d_{i}' for i in range(3)],        # 3D направление взгляда
            *[f'head_pose_{i}' for i in range(6)],      # Поза головы (6 параметров)
            *[f'right_eye_center_{i}' for i in range(3)],  # 3D центр правого глаза
            *[f'left_eye_center_{i}' for i in range(3)]    # 3D центр левого глаза
        ]
        
        df = pd.DataFrame(annotations, columns=columns)
        
        # Очистка данных (критерий А1: обработка пустых значений, дубликатов)
        initial_count = len(df)
        
        # Проверка и удаление дубликатов
        df = df.drop_duplicates()
        
        # Проверка на пропущенные значения
        missing_counts = df.isnull().sum().sum()
        if missing_counts > 0:
            df = df.fillna(df.mean(numeric_only=True))
        
        print(f"Загружено записей: {initial_count}")
        print(f"После очистки: {len(df)} записей")
        
        self.report_content.append(f"Загружено записей: {initial_count}")
        self.report_content.append(f"После очистки: {len(df)} записей")
        
        return df
    
    def analyze_correlations(self, df):
        """Анализ корреляций (критерий А1: построение корреляционной матрицы)"""
        print("[Шаг 2/6] Анализ корреляций...")
        
        # Выбираем ключевые признаки для анализа
        key_features = [
            'screen_gaze_x', 'screen_gaze_y',
            'head_pose_0', 'head_pose_1', 'head_pose_2',
            'head_pose_3', 'head_pose_4', 'head_pose_5'
        ]
        
        # Фильтруем существующие колонки
        available_features = [f for f in key_features if f in df.columns]
        
        if len(available_features) < 2:
            print("Недостаточно признаков для анализа корреляций")
            return None
        
        # Создаем матрицу корреляций
        correlation_matrix = df[available_features].corr()
        
        # Визуализация
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', linewidths=0.5, square=True)
        plt.title('Матрица корреляций ключевых признаков')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Анализ сильных корреляций
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > 0.7:
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    strong_correlations.append((col1, col2, correlation_matrix.iloc[i, j]))
        
        self.report_content.append("\nКорреляционный анализ:")
        self.report_content.append(f"Проанализировано признаков: {len(available_features)}")
        
        if strong_correlations:
            self.report_content.append("Обнаружены сильные корреляции (|r| > 0.7):")
            for col1, col2, corr in strong_correlations:
                self.report_content.append(f"  {col1} ↔ {col2}: {corr:.3f}")
        else:
            self.report_content.append("Сильных корреляций (|r| > 0.7) не обнаружено")
        
        print("Матрица корреляций сохранена в 'correlation_matrix.png'")
        return correlation_matrix
    
    def create_scatter_plots(self, df):
        """Создание диаграмм рассеяния (критерий А1)"""
        print("[Шаг 3/6] Построение диаграмм рассеяния...")
        
        # Выбираем пары признаков для визуализации
        pairs = [
            ('screen_gaze_x', 'screen_gaze_y'),
            ('head_pose_0', 'head_pose_1'),
            ('head_pose_2', 'head_pose_3'),
            ('screen_gaze_x', 'head_pose_0')
        ]
        
        # Фильтруем существующие пары
        valid_pairs = []
        for x_col, y_col in pairs:
            if x_col in df.columns and y_col in df.columns:
                valid_pairs.append((x_col, y_col))
        
        if len(valid_pairs) == 0:
            print("Нет подходящих признаков для диаграмм рассеяния")
            return
        
        # Создаем сетку графиков
        n_pairs = len(valid_pairs)
        n_cols = 2
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        
        if n_pairs > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for idx, (x_col, y_col) in enumerate(valid_pairs):
            if idx < len(axes):
                axes[idx].scatter(df[x_col], df[y_col], alpha=0.5, s=20)
                axes[idx].set_xlabel(x_col)
                axes[idx].set_ylabel(y_col)
                axes[idx].set_title(f'{x_col} vs {y_col}')
                axes[idx].grid(True, alpha=0.3)
        
        # Скрываем неиспользуемые оси
        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('scatter_plots.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        self.report_content.append("\nДиаграммы рассеяния:")
        self.report_content.append(f"Построено диаграмм: {len(valid_pairs)}")
        
        # Анализ зависимостей
        insights = []
        for x_col, y_col in valid_pairs:
            if x_col == 'screen_gaze_x' and y_col == 'screen_gaze_y':
                correlation = df[x_col].corr(df[y_col])
                insights.append(f"Координаты взгляда X и Y коррелируют: r={correlation:.3f}")
        
        if insights:
            self.report_content.append("Выводы по диаграммам рассеяния:")
            for insight in insights:
                self.report_content.append(f"  • {insight}")
        
        print("Диаграммы рассеяния сохранены в 'scatter_plots.png'")
    
    def perform_clustering(self, df):
        """Кластеризация данных (критерий А1)"""
        print("[Шаг 4/6] Выполнение кластеризации...")
        
        # Выбираем признаки для кластеризации
        cluster_features = ['screen_gaze_x', 'screen_gaze_y']
        available_features = [f for f in cluster_features if f in df.columns]
        
        if len(available_features) < 2:
            print("Недостаточно признаков для кластеризации")
            return None
        
        X_cluster = df[available_features].values
        
        # Нормализация данных для кластеризации
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Применение K-means кластеризации
        n_clusters = 3  # Можно адаптировать
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Визуализация результатов кластеризации
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df[available_features[0]], df[available_features[1]], 
                             c=clusters, cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Кластер')
        plt.xlabel(available_features[0])
        plt.ylabel(available_features[1])
        plt.title(f'Кластеризация данных (K-means, k={n_clusters})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('clustering_result.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Анализ кластеров
        df['cluster'] = clusters
        cluster_stats = df.groupby('cluster')[available_features].mean()
        
        self.report_content.append("\nКластеризация данных:")
        self.report_content.append(f"Метод: K-means с {n_clusters} кластерами")
        self.report_content.append("Статистика по кластерам:")
        
        for cluster_id in range(n_clusters):
            cluster_size = (df['cluster'] == cluster_id).sum()
            self.report_content.append(f"  Кластер {cluster_id}: {cluster_size} записей")
            for feature in available_features:
                mean_val = df[df['cluster'] == cluster_id][feature].mean()
                self.report_content.append(f"    {feature}: {mean_val:.2f}")
        
        print(f"Кластеризация завершена. Создано {n_clusters} кластеров")
        print("Результаты сохранены в 'clustering_result.png'")
        
        return clusters
    
    def analyze_distributions(self, df):
        """Анализ распределений данных (критерий А1, А2)"""
        print("[Шаг 5/6] Анализ распределений...")
        
        # Выбираем ключевые признаки для анализа
        key_features = [
            'screen_gaze_x', 'screen_gaze_y',
            'head_pose_0', 'head_pose_1', 'head_pose_2'
        ]
        
        available_features = [f for f in key_features if f in df.columns]
        
        if len(available_features) == 0:
            print("Нет признаков для анализа распределений")
            return
        
        # Создаем гистограммы распределений
        n_features = min(6, len(available_features))
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        
        if n_features > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        distribution_types = {}
        
        for idx, feature in enumerate(available_features[:n_features]):
            if idx < len(axes):
                data = df[feature].dropna()
                
                # Построение гистограммы
                axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_xlabel('Значение')
                axes[idx].set_ylabel('Частота')
                axes[idx].set_title(f'Распределение: {feature}')
                axes[idx].grid(True, alpha=0.3)
                
                # Определение типа распределения
                skewness = data.skew()
                kurtosis = data.kurtosis()
                
                if abs(skewness) < 0.5:
                    dist_type = "приблизительно нормальное"
                elif skewness > 0:
                    dist_type = "правостороннее асимметричное"
                else:
                    dist_type = "левостороннее асимметричное"
                
                distribution_types[feature] = {
                    'type': dist_type,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'mean': data.mean(),
                    'std': data.std()
                }
        
        # Скрываем неиспользуемые оси
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('distributions.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Добавляем информацию в отчет
        self.report_content.append("\nАнализ распределений:")
        for feature, stats in distribution_types.items():
            self.report_content.append(f"  {feature}:")
            self.report_content.append(f"    Тип: {stats['type']}")
            self.report_content.append(f"    Скошенность: {stats['skewness']:.3f}")
            self.report_content.append(f"    Среднее: {stats['mean']:.3f}, Ст.отклонение: {stats['std']:.3f}")
        
        print(f"Проанализировано распределений: {len(distribution_types)}")
        print("Графики распределений сохранены в 'distributions.png'")
        
        return distribution_types
    
    def remove_outliers(self, df):
        """Удаление выбросов (критерий А1)"""
        print("[Шаг 6/6] Удаление выбросов...")
        
        initial_count = len(df)
        
        # Выбираем числовые колонки для анализа выбросов
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ограничиваемся ключевыми признаками для избежания излишнего удаления
        key_cols = [col for col in ['screen_gaze_x', 'screen_gaze_y', 
                                   'head_pose_0', 'head_pose_1', 'head_pose_2'] 
                   if col in numeric_cols]
        
        if not key_cols:
            print("Нет ключевых признаков для анализа выбросов")
            return df
        
        # Метод межквартильного размаха (IQR) для удаления выбросов
        for col in key_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Фильтрация выбросов
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df = df[mask]
        
        removed_count = initial_count - len(df)
        removal_percentage = (removed_count / initial_count) * 100 if initial_count > 0 else 0
        
        self.report_content.append("\nОбработка выбросов:")
        self.report_content.append(f"Изначально записей: {initial_count}")
        self.report_content.append(f"Удалено выбросов: {removed_count} ({removal_percentage:.1f}%)")
        self.report_content.append(f"Осталось записей: {len(df)}")
        
        print(f"Удалено выбросов: {removed_count} записей")
        
        return df
    
    def prepare_for_modeling(self, df):
        """Подготовка данных для обучения модели"""
        # Выбираем признаки и целевую переменную
        feature_cols = [f'head_pose_{i}' for i in range(6)]
        target_cols = ['screen_gaze_x', 'screen_gaze_y']
        
        # Фильтруем существующие колонки
        available_features = [col for col in feature_cols if col in df.columns]
        available_targets = [col for col in target_cols if col in df.columns]
        
        if not available_features or not available_targets:
            print("Недостаточно данных для подготовки к моделированию")
            return None, None, None, None
        
        X = df[available_features].values
        y = df[available_targets].values
        
        # Нормализация
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        self.report_content.append("\nПодготовка данных для моделирования:")
        self.report_content.append(f"Признаки: {len(available_features)}")
        self.report_content.append(f"Целевые переменные: {len(available_targets)}")
        self.report_content.append(f"Обучающая выборка: {X_train.shape[0]} записей")
        self.report_content.append(f"Тестовая выборка: {X_test.shape[0]} записей")
        
        return X_train, X_test, y_train, y_test
    
    def save_report(self):
        """Сохранение отчета о проделанной работе (критерий А3)"""
        report_path = 'report_A.md'
        
        # Создаем заголовок отчета
        full_report = ["# Отчет по модулю А: Анализ и предобработка данных", ""]
        full_report.append("## Выполненные работы")
        full_report.extend(self.report_content)
        
        # Добавляем рекомендации для обучения модели
        full_report.append("\n## Рекомендации для обучения модели")
        full_report.append("1. Используйте подготовленные признаки позы головы для предсказания координат взгляда")
        full_report.append("2. Рассмотрите использование нормализованных изображений из папки 'Normalized' для CNN")
        full_report.append("3. Учтите индивидуальные различия между участниками (разделение по participant)")
        full_report.append("4. Для классификации преобразуйте координаты взгляда в зоны экрана (например, 3x3)")
        
        # Сохраняем отчет
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_report))
        
        print(f"\nОтчет сохранен в '{report_path}'")
        
        return report_path
    
    def process_complete_analysis(self, participant='p00', day='day01'):
        """Полный цикл обработки и анализа данных"""
        print("=" * 60)
        print("НАЧАЛО ОБРАБОТКИ ДАННЫХ MPIIGAZE")
        print("=" * 60)
        
        # 1. Загрузка и очистка данных
        df = self.load_and_clean_data(participant, day)
        if df is None:
            return None
        
        # 2. Анализ корреляций
        self.analyze_correlations(df)
        
        # 3. Построение диаграмм рассеяния
        self.create_scatter_plots(df)
        
        # 4. Кластеризация данных
        self.perform_clustering(df)
        
        # 5. Анализ распределений
        self.analyze_distributions(df)
        
        # 6. Удаление выбросов
        df_cleaned = self.remove_outliers(df)
        
        # 7. Подготовка данных для моделирования
        training_data = self.prepare_for_modeling(df_cleaned)
        
        # 8. Сохранение отчета
        report_file = self.save_report()
        
        print("\n" + "=" * 60)
        print("ОБРАБОТКА ЗАВЕРШЕНА")
        print("=" * 60)
        print("\nСозданные файлы:")
        print("1. correlation_matrix.png - матрица корреляций")
        print("2. scatter_plots.png - диаграммы рассеяния")
        print("3. clustering_result.png - результаты кластеризации")
        print("4. distributions.png - распределения признаков")
        print(f"5. {report_file} - отчет о проделанной работе")
        
        return df_cleaned

# Запуск обработки
if __name__ == "__main__":
    processor = MPIIGazeProcessor('MPIIGaze')
    processed_data = processor.process_complete_analysis('p00', 'day01')