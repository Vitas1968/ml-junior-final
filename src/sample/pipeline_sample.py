"""
Production training pipeline.

Назначение:
- запуск preprocessing;
- обучение финальной модели;
- оценка качества;
- сохранение артефактов.

Ноутбуки НЕ используются.
"""



from src.sample.preprocessing_sample import run_preprocessing


def main():
    print("▶ Running SAMPLE preprocessing pipeline...")
    run_preprocessing(with_target=True)
    print("✔ SAMPLE pipeline completed")


if __name__ == "__main__":
    main()

