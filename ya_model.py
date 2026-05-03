import os
os.environ["HF_HOME"] = "C:\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import re
import torch
import numpy as np
import argparse
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer


MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-instruct"

def load_model():
    """Загрузка модели и токенизатора"""
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Загрузка модели с 4-битным квантованием...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print("Модель загружена!")
    return model, tokenizer


class ThemeRAGSystem:
    def __init__(self):
        self.themes_data = self._load_themes_database()
        print("Загрузка эмбеддера для RAG...")
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self._build_vector_index()
        print("RAG система инициализирована!")

    def _load_themes_database(self):
        """Загружает базу данных тем с контекстными метками"""
        return [
            {
                "id": 0,
                "name": "нет рисков",
                "description": "Отсутствие каких-либо рисковых тем",
                "age_rating": "0+",
                "examples": [],
                "context_labels": ["уровень: нулевой"]
            },
            {
                "id": 1,
                "name": "Нецензурная лексика",
                "description": "Мат, обсценная речь",
                "age_rating": "16+ / 18+",
                "examples": ["мат", "обсценная лексика", "брань"],
                "context_labels": ["частота: низкая/высокая", "эмоциональный_контекст"]
            },
            {
                "id": 2,
                "name": "Оскорбления, унижения",
                "description": "Издевательства, дискриминация",
                "age_rating": "12+ / 16+",
                "examples": ["оскорбления", "унижения", "издевательства"],
                "context_labels": ["высказывания_в_шутку", "по_признаку"]
            },
            {
                "id": 3,
                "name": "Физическое насилие",
                "description": "Драки, убийства",
                "age_rating": "16+ / 18+",
                "examples": ["драка", "убийство", "избиение"],
                "context_labels": ["натуралистичность", "одобрение"]
            },
            {
                "id": 4,
                "name": "Психологическое давление",
                "description": "Манипуляции, угрозы",
                "age_rating": "12+ / 16+",
                "examples": ["манипуляции", "угрозы", "шантаж"],
                "context_labels": ["эмоциональная_нагрузка"]
            },
            {
                "id": 5,
                "name": "Сексуальный контент",
                "description": "Половой акт, возбуждение",
                "age_rating": "18+",
                "examples": ["секс", "половой акт", "интим"],
                "context_labels": ["явность", "контекст: искусство/эротика"]
            },
            {
                "id": 6,
                "name": "Флирт / намёки",
                "description": "Поцелуи, заигрывания",
                "age_rating": "12+ / 16+",
                "examples": ["флирт", "поцелуи", "заигрывания"],
                "context_labels": ["характер: невинный/эротичный"]
            },
            {
                "id": 7,
                "name": "Алкоголь",
                "description": "Упоминание конкретных алкогольных напитков или состояния опьянения",
                "age_rating": "16+ / 18+",
                "examples": ["водка", "пиво", "вино", "коньяк", "шампанское", "опьянение"],
                "context_labels": ["возраст_персонажа", "пропаганда"]
            },
            {
                "id": 8,
                "name": "Курение",
                "description": "Курение, бренды табачных изделий",
                "age_rating": "12+ / 16+",
                "examples": ["сигареты", "папиросы", "табак", "вейп"],
                "context_labels": ["пропаганда", "возраст_персонажа"]
            },
            {
                "id": 9,
                "name": "Наркотики / ПАВ",
                "description": "Употребление, продажа наркотических веществ",
                "age_rating": "18+",
                "examples": ["наркотики", "героин", "кокаин", "марихуана"],
                "context_labels": ["инструкции", "романтизация"]
            },
            {
                "id": 10,
                "name": "Самоубийство / суицидальные мотивы",
                "description": "Попытки, размышления о самоубийстве",
                "age_rating": "18+",
                "examples": ["суицид", "самоубийство", "повеситься", "выброситься", "суицидальные мысли"],
                "context_labels": ["детализация", "оправдание"]
            },
            {
                "id": 11,
                "name": "Смерть / утрата",
                "description": "Траур, похороны, переживание потери",
                "age_rating": "12+ / 16+",
                "examples": ["смерть", "похороны", "траур", "горе", "потеря близкого"],
                "context_labels": ["эмоциональная_нагрузка"]
            },
            {
                "id": 12,
                "name": "Ужасы / страх / тревога",
                "description": "Саспенс, чудовища, создание атмосферы страха",
                "age_rating": "12+ / 16+",
                "examples": ["ужасы", "страх", "тревога", "саспенс", "чудовища", "кошмары"],
                "context_labels": ["детализация", "визуальный_эффект"]
            },
            {
                "id": 13,
                "name": "Мистика / демоны / потустороннее",
                "description": "Магия, ритуалы, сверхъестественные явления",
                "age_rating": "6+ / 12+ / 16+",
                "examples": ["магия", "демоны", "призраки", "ритуалы", "потустороннее", "одержимость"],
                "context_labels": ["тип: сказка/оккультизм", "жестокость"]
            },
            {
                "id": 14,
                "name": "Криминал / преступность",
                "description": "Воровство, мафия, организованная преступность",
                "age_rating": "12+ / 16+",
                "examples": ["воровство", "мафия", "банда", "ограбление", "шантаж", "вымогательство"],
                "context_labels": ["романтизация", "осуждение"]
            },
            {
            "id": 15,
            "name": "Ставка на деньги",
            "description": "Казино, ставки на деньги, игровые автоматы на реальные деньги. НЕ ВКЛЮЧАЕТ: обычные видеоигры без денежных ставок",
            "age_rating": "18+",
            "examples": ["казино", "ставки", "покер на деньги", "рулетка", "игровые автоматы", "выигрыш денег", "проигрыш денег", "букмекер"],
            "context_labels": ["реклама", "возраст_игроков", "денежные_ставки"]
        },
        {
            "id": 16,
            "name": "Оружие / стрельба",
            "description": "Пистолеты, ножи, использование оружия",
            "age_rating": "12+ / 16+",
            "examples": ["пистолет", "револьвер", "нож", "стрельба", "выстрел", "оружейный клуб"],
            "context_labels": ["использование", "возраст_персонажа"]
        },
        {
            "id": 17,
            "name": "Кровь / расчленение / трупы",
            "description": "Натуралистические сцены с кровью и насилием",
            "age_rating": "18+",
            "examples": ["кровь", "расчленение", "трупы", "органы", "внутренности", "рана"],
            "context_labels": ["детализация", "продолжительность"]
        },
        {
            "id": 18,
            "name": "Травмы / страдания",
            "description": "Раны, физическая боль, мучения",
            "age_rating": "16+ / 18+",
            "examples": ["травмы", "раны", "боль", "страдания", "пытки", "ожоги", "переломы"],
            "context_labels": ["эмоциональный_уровень", "показ_деталей"]
        },
            {
                "id": 19,
                "name": "Насилие над животными",
                "description": "Охота, жестокое обращение с животными",
                "age_rating": "16+ / 18+",
                "examples": ["жестокость к животным", "охота", "избиение животных", "эксперименты", "бойни"],
                "context_labels": ["оправдание", "реализм"]
            },
            {
                "id": 20,
                "name": "Пропаганда насилия",
                "description": "Одобрение мести, войны, насильственных действий",
                "age_rating": "18+",
                "examples": ["пропаганда насилия", "одобрение мести", "призывы к войне", "оправдание агрессии"],
                "context_labels": ["осуждение", "контекст: исторический/реальный"]
            },
            {
                "id": 21,
                "name": "Оружие у детей",
                "description": "Игры с оружием, использование оружия несовершеннолетними",
                "age_rating": "12+ / 16+",
                "examples": ["оружие у детей", "ребенок с пистолетом", "игрушечное оружие", "подростки со стрелковым оружием"],
                "context_labels": ["контекст: игра/опасность"]
            },
            {
                "id": 22,
                "name": "Семейные конфликты",
                "description": "Развод, агрессия в семье, межпоколенческие конфликты",
                "age_rating": "12+ / 16+",
                "examples": ["семейные ссоры", "развод", "конфликты родителей и детей", "домашние скандалы"],
                "context_labels": ["эмоциональный_контекст"]
            },
            {
                "id": 23,
                "name": "Бытовое насилие",
                "description": "Домашние побои, насилие в близких отношениях",
                "age_rating": "16+ / 18+",
                "examples": ["домашнее насилие", "побои", "избиение супруга", "насилие в семье"],
                "context_labels": ["осуждение", "натуралистичность"]
            },
            {
                "id": 24,
                "name": "Дискриминация / расизм / сексизм",
                "description": "Пренебрежительные фразы по расовому, гендерному или другим признакам",
                "age_rating": "16+ / 18+",
                "examples": ["расизм", "сексизм", "дискриминация", "национализм", "ксенофобия", "гомофобия"],
                "context_labels": ["осуждение", "повторяемость"]
            },
            {
                "id": 25,
                "name": "Пропаганда вредных привычек",
                "description": "Романтизация употребления алкоголя, табака, наркотиков",
                "age_rating": "18+",
                "examples": ["пропаганда алкоголя", "реклама курения", "романтизация наркотиков", "герой пьет/курит"],
                "context_labels": ["контекст", "возраст_персонажа"]
            },
            {
                "id": 26,
                "name": "Вульгарность / непристойные шутки",
                "description": "Эротические метафоры, неприличные шутки",
                "age_rating": "16+",
                "examples": ["вульгарные шутки", "непристойные анекдоты", "эротические намеки", "пошлости"],
                "context_labels": ["частота", "намерение"]
            },
            {
                "id": 27,
                "name": "Саморазрушение / вред себе",
                "description": "Резка, членовредительство, осознанное причинение вреда себе",
                "age_rating": "18+",
                "examples": ["селфхарм", "членовредительство", "резка", "ожоги себе", "голодовка"],
                "context_labels": ["детализация", "романтизация"]
            },
            {
                "id": 28,
                "name": "Экстремизм / терроризм",
                "description": "Призывы к насилию, террористическая деятельность",
                "age_rating": "18+",
                "examples": ["экстремизм", "терроризм", "призывы к насилию", "радикализация", "вербовка"],
                "context_labels": ["пропаганда", "реальные_события"]
            },
            {
                "id": 29,
                "name": "Политическая пропаганда",
                "description": "Агитация, политические лозунги, пропаганда",
                "age_rating": "16+ / 18+",
                "examples": ["политическая агитация", "лозунги", "пропаганда", "митинги", "политические призывы"],
                "context_labels": ["одобрение", "призывы"]
            },
            {
                "id": 30,
                "name": "Религиозная чувствительность",
                "description": "Богохульство, религиозная сатира, критика веры",
                "age_rating": "16+ / 18+",
                "examples": ["богохульство", "религиозная сатира", "критика религии", "насмешка над обрядами"],
                "context_labels": ["ирония", "насмешка"]
            },
            {
                "id": 31,
                "name": "Эротика без обнажения",
                "description": "Силуэты, страстные поцелуи, эротические сцены без показа обнаженного тела",
                "age_rating": "16+",
                "examples": ["эротические сцены", "страстные поцелуи", "силуэты", "интимные объятия", "сексуальное напряжение"],
                "context_labels": ["характер", "длительность"]
            },
            {
                "id": 32,
                "name": "Обнажённое тело (нейтрально)",
                "description": "Медицинские сцены, художественные изображения без сексуального контекста",
                "age_rating": "12+ / 16+",
                "examples": ["медицинский осмотр", "художественная нагота", "пляжные сцены", "переодевание", "академический рисунок"],
                "context_labels": ["контекст: учебный/эротический"]
            },
            {
                "id": 33,
                "name": "Обнажённое тело (сексуализировано)",
                "description": "Интимные сцены с обнаженным телом в сексуальном контексте",
                "age_rating": "18+",
                "examples": ["сексуализированная нагота", "эротическое обнажение", "интимные сцены", "постельные сцены с обнажением"],
                "context_labels": ["детализация", "время_кадра"]
            },
            {
                "id": 34,
                "name": "Моральный выбор",
                "description": "Ложь, предательство, сложные моральные дилеммы",
                "age_rating": "12+ / 16+",
                "examples": ["измена", "предательство", "моральный выбор", "ложь", "обман", "нравственные дилеммы"],
                "context_labels": ["моральное_осуждение"]
            },
            {
                "id": 35,
                "name": "Алчность / коррупция",
                "description": "Взятки, хищения, коррупционные схемы",
                "age_rating": "12+ / 16+",
                "examples": ["взятки", "коррупция", "хищения", "вымогательство", "злоупотребление должностью", "откаты"],
                "context_labels": ["осуждение", "пропаганда"]
            },
            {
                "id": 36,
                "name": "Психические расстройства",
                "description": "Фобии, психозы, депрессия и другие психические заболевания",
                "age_rating": "12+ / 16+ / 18+",
                "examples": ["депрессия", "психоз", "шизофрения", "фобии", "панические атаки", "биполярное расстройство", "ОКР"],
                "context_labels": ["тип", "деструктивность"]
            },
            {
                "id": 37,
                "name": "Философские / экзистенциальные темы",
                "description": "Смерть, смысл жизни, экзистенциальные вопросы",
                "age_rating": "12+ / 16+",
                "examples": ["смысл жизни", "экзистенциальные вопросы", "философские размышления", "смерть", "бытие", "свобода воли"],
                "context_labels": ["тяжесть", "возраст_аудитории"]
            },
            {
                "id": 38,
                "name": "Война / разрушения",
                "description": "Сражения, взрывы, военные действия",
                "age_rating": "12+ / 16+",
                "examples": ["война", "сражения", "взрывы", "бомбежки", "военные действия", "разрушения", "блокада"],
                "context_labels": ["реализм", "насилие"]
            },
            {
                "id": 39,
                "name": "Психотропные вещества / лекарства",
                "description": "Таблетки, антидепрессанты, психоактивные препараты",
                "age_rating": "16+ / 18+",
                "examples": ["антидепрессанты", "транквилизаторы", "психотропные препараты", "лекарства", "таблетки", "рецептурные препараты"],
                "context_labels": ["контекст: лечение/злоупотребление"]
            },
            {
                "id": 40,
                "name": "Зависимости (игры, соцсети)",
                "description": "Потеря контроля над играми, социальными сетями, интернетом",
                "age_rating": "12+ / 16+",
                "examples": ["игровая зависимость", "зависимость от соцсетей", "интернет-зависимость", "ломка", "потеря контроля", "компульсивное использование"],
                "context_labels": ["негативная_оценка", "возраст_персонажа"]
            },
            {
                "id": 41,
                "name": "Эксплуатация детей",
                "description": "Использование детей в труде, порнографии, других формах эксплуатации",
                "age_rating": "18+",
                "examples": ["детский труд", "эксплуатация детей", "использование детей", "наем детей", "детская порнография"],
                "context_labels": ["насилие", "возраст_жертвы"]
            },
            {
                "id": 42,
                "name": "Торговля людьми / рабство",
                "description": "Похищения, работорговля, принудительный труд",
                "age_rating": "18+",
                "examples": ["торговля людьми", "рабство", "похищения", "принудительный труд", "сексуальное рабство", "контрабанда людьми"],
                "context_labels": ["детализация", "эмоциональный_контекст"]
            },
            {
                "id": 43,
                "name": "Пропаганда антисоциальных действий",
                "description": "Воровство без наказания, нарушение социальных норм",
                "age_rating": "12+ / 16+",
                "examples": ["пропаганда воровства", "антисоциальное поведение", "нарушение норм", "противоправные действия", "социальный протест"],
                "context_labels": ["моральная_оценка"]
            },
            {
                "id": 44,
                "name": "Разрушение норм / аморальность",
                "description": "Отказ от традиционных ценностей, аморальное поведение",
                "age_rating": "16+ / 18+",
                "examples": ["разрушение норм", "аморальность", "отказ от ценностей", "нигилизм", "антимораль", "деконструкция"],
                "context_labels": ["контекст: сатира/идеология"]
            },
            {
                "id": 45,
                "name": "Нарушение закона",
                "description": "Укрывательство преступлений, противоправные действия",
                "age_rating": "12+ / 16+",
                "examples": ["нарушение закона", "укрывательство", "соучастие", "пособничество", "незаконные действия", "преступный сговор"],
                "context_labels": ["мотивировка", "осуждение"]
            },
            {
                "id": 46,
                "name": "Медицинские процедуры",
                "description": "Уколы, операции, медицинские вмешательства",
                "age_rating": "12+ / 16+ / 18+",
                "examples": ["операция", "уколы", "переливание крови", "хирургия", "медицинские манипуляции", "стоматология", "швы"],
                "context_labels": ["детализация", "тип_процедуры"]
            },
            {
                "id": 47,
                "name": "Катастрофы / аварии / пожары",
                "description": "Трагедии, техногенные и природные катастрофы",
                "age_rating": "12+ / 16+",
                "examples": ["катастрофа", "авария", "пожар", "землетрясение", "наводнение", "цунами", "крушение"],
                "context_labels": ["реализм", "пострадавшие"]
            },
            {
                "id": 48,
                "name": "Агрессия животных / монстров",
                "description": "Нападения животных, монстров, кровь и жестокость",
                "age_rating": "12+ / 16+",
                "examples": ["нападение животных", "агрессивные звери", "монстры", "чудовища", "хищники", "злые собаки"],
                "context_labels": ["реализм", "страшный_эффект"]
            },
            {
                "id": 49,
                "name": "Магия / колдовство",
                "description": "Заклинания, ведьмы, магические ритуалы",
                "age_rating": "6+ / 12+ / 16+",
                "examples": ["магия", "колдовство", "заклинания", "ведьмы", "волшебство", "магические ритуалы", "чары"],
                "context_labels": ["уровень_страха", "вера"]
            },
            {
                "id": 50,
                "name": "Манипуляции сознанием / секты",
                "description": "Контроль сознания, гипноз, сектантство",
                "age_rating": "16+ / 18+",
                "examples": ["манипуляции сознанием", "секты", "гипноз", "промывка мозгов", "контроль мышления", "вербовка"],
                "context_labels": ["пропаганда", "контекст"]
            },
            {
                "id": 51,
                "name": "Несовершеннолетние жертвы",
                "description": "Личные данные, фото, описания несовершеннолетних жертв",
                "age_rating": "18+",
                "examples": ["несовершеннолетние жертвы", "дети-жертвы", "подростки-жертвы", "эксплуатация детей", "насилие над детьми"],
                "context_labels": ["реальность", "жертва_ребёнок"]
            },
            {
                "id": 52,
                "name": "Иностранные агенты / политическая маркировка",
                "description": "Упоминания, цитаты иностранных агентов, требующие маркировки",
                "age_rating": "Требует маркировки",
                "examples": ["иностранные агенты", "политическая маркировка", "иноагент", "резиновые формулировки", "политические цитаты"],
                "context_labels": ["наличие_реестра", "цитирование"]
            },
            {
                "id": 53,
                "name": "Инструкции по преступлениям / оружию / наркотикам",
                "description": "Пошаговые описания совершения преступлений, изготовления оружия или наркотиков",
                "age_rating": "18+",
                "examples": ["инструкции по преступлениям", "изготовление оружия", "рецепты наркотиков", "взрывчатка", "взлом", "кража"],
                "context_labels": ["детализация", "практическая_ценность"]
            },
            {
                "id": 54,
                "name": "Отрицание семейных ценностей",
                "description": "Насмешка над семьёй, отказ от родительства, отрицание традиционных ценностей",
                "age_rating": "16+ / 18+",
                "examples": ["отрицание семьи", "насмешка над браком", "отказ от родительства", "антисемейные высказывания", "критика семейных ценностей"],
                "context_labels": ["ирония", "осуждение"]
            },
            {
                "id": 55,
                "name": "Нарушение неприкосновенности частной жизни",
                "description": "Публикация личных данных, переписок без согласия",
                "age_rating": "18+",
                "examples": ["нарушение приватности", "личные данные", "переписки", "фотографии без согласия", "шантаж личной информацией", "слежка"],
                "context_labels": ["персональные_данные", "согласие: нет"]
            },
            {
                "id": 56,
                "name": "Дискредитация гос. институтов",
                "description": "Осмеяние армии, суда, власти без осуждения",
                "age_rating": "18+",
                "examples": ["дискредитация армии", "осмеяние суда", "критика власти", "сатира на госорганы", "высмеивание полиции"],
                "context_labels": ["ирония", "оценка: положительная"]
            },
            {
                "id": 57,
                "name": "Использование мата в шутке",
                "description": "Нецензурная лексика в ироничном контексте",
                "age_rating": "12+ / 16+",
                "examples": ["мат в шутку", "ироничный мат"],
                "context_labels": ["оценка: нейтральная", "эмоция: ирония"]
            },
            {
                "id": 58,
                "name": "Эксплуатация животных в шоу",
                "description": "Использование животных без насилия",
                "age_rating": "12+ / 16+",
                "examples": ["цирковые животные", "дрессировка"],
                "context_labels": ["оценка: нейтральная", "уровень: низкий"]
            }
        ]

    def _build_vector_index(self):
        """Строит векторный индекс для семантического поиска"""
        theme_texts = []
        for theme in self.themes_data:
            text_representation = f"{theme['name']} {theme['description']} {' '.join(theme['examples'])} {' '.join(theme['context_labels'])}"
            theme_texts.append(text_representation)

        self.theme_embeddings = self.embedder.encode(theme_texts)
        self.theme_names = [theme['name'] for theme in self.themes_data]

    def find_relevant_themes(self, scene_text: str, top_k: int = 5) -> list:
        """Находит наиболее релевантные темы для текста сцены"""
        scene_embedding = self.embedder.encode([scene_text])
        similarities = np.dot(self.theme_embeddings, scene_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        relevant_themes = []
        for idx in top_indices:
            theme = self.themes_data[idx]
            relevant_themes.append({
                "name": theme["name"],
                "similarity": float(similarities[idx]),
                "age_rating": theme["age_rating"],
                "context_labels": theme["context_labels"]
            })

        return relevant_themes

    def get_theme_by_name(self, theme_name: str) -> dict:
        """Возвращает полную информацию о теме по названию"""
        for theme in self.themes_data:
            if theme["name"] == theme_name:
                return theme
        return None


def clean_scene_text(text: str) -> str:
    """Удаляет символы, ломающие JSON, и ограничивает длину"""
    text = re.sub(r"```", " ", text)
    text = re.sub(r"`", "'", text)
    text = re.sub(r'""+', '"', text)
    return text[:1200].strip()

def generate_response(prompt: str, model, tokenizer) -> str:
    """Генерация ответа от модели"""
    device = model.device
    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None

    if has_chat_template:
        try:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(device)
            prompt_len = input_ids.shape[1]
        except Exception:
            has_chat_template = False

    if not has_chat_template:
        full_prompt = f"### Инструкция:\n{prompt}\n\n### Ответ:\n"
        encoded = tokenizer(full_prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        prompt_len = encoded.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=800,
            do_sample=False,
            temperature=0.0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    gen_tokens = outputs[0][prompt_len:]
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return response.strip()

def build_prompt(scene_text: str, theme_rag) -> str:
    """Создание промпта для модели"""
    relevant_themes = theme_rag.find_relevant_themes(scene_text, top_k=8)

    rag_hints = "РЕЛЕВАНТНЫЕ ТЕМЫ ДЛЯ АНАЛИЗА (используй при наличии в тексте):\n"
    for i, theme in enumerate(relevant_themes, 1):
        rag_hints += f"- {theme['name']} ({theme['age_rating']}) - {theme['context_labels']}\n"

    return f"""Ты — эксперт по возрастному рейтингу фильмов в России согласно Федеральному закону №436-ФЗ.

{rag_hints}

Твоя задача — проанализировать фрагмент сценария и:
1) определить минимальный возрастной рейтинг: 0+, 6+, 12+, 16+ или 18+;
2) выявить ВСЕ темы риска из списка (1–58), которые явно присутствуют в тексте.

КРИТЕРИИ РЕЙТИНГА:
- 18+: мат, порнография, самоубийство/селфхарм, наркотики/алкоголь без осуждения, оправдание насилия.
- 16+: отдельные бранные слова, жестокость без натурализма, наркотики с осуждением.
- 12+: эпизодическое упоминание вредных привычек, насилие с осуждением.
- 6+: антиобщественные действия с осуждением.
- 0+: только если рисков нет.

СТРОГОЕ ТРЕБОВАНИЕ:
- ВЕРНИ ТОЛЬКО ОДИН ВАЛИДНЫЙ JSON-ОБЪЕКТ, начинающийся с {{ и заканчивающийся }}.
- НИКАКИХ: ```json, ```, //, /* */, пояснений, новых строк до {{ или после }}.
- Если не хватает места — сократи reason/evidence, но СОХРАНИ ВАЛИДНОСТЬ.
- НЕ выдумывай контент! Анализируй ТОЛЬКО то, что явно указано в тексте.
- Пример:
{{"rating":"16+","themes":["Алкоголь"],"reason":"Упоминание водки.","evidence":"— Пей!"}}

ТЕКСТ СЦЕНЫ:
{scene_text}

ОТВЕТ (ТОЛЬКО ЧИСТЫЙ JSON):"""

def parse_response(response: str, theme_rag) -> dict:
    """Парсинг ответа от модели"""
    response = re.split(r'^ОТВЕТ[:\s]*', response, flags=re.MULTILINE | re.IGNORECASE)[-1].strip()

    best_parsed = None
    best_length = 0

    stack = []
    for i, char in enumerate(response):
        if char == '{':
            stack.append(i)
        elif char == '}' and stack:
            start = stack.pop()
            candidate = response[start:i+1]
            try:
                parsed = json.loads(candidate)
                if all(k in parsed for k in ["rating", "themes", "reason", "evidence"]):
                    if len(candidate) > best_length:
                        best_parsed = parsed
                        best_length = len(candidate)
            except (json.JSONDecodeError, TypeError):
                continue

    if best_parsed is not None:
        rating = best_parsed["rating"]
        if rating not in {"0+", "6+", "12+", "16+", "18+"}:
            rating = "18+"
        themes = best_parsed["themes"]
        if not isinstance(themes, list):
            themes = ["нет рисков"]
        else:
            valid_themes = []
            for theme_name in themes:
                theme_info = theme_rag.get_theme_by_name(str(theme_name).strip())
                if theme_info is not None:
                    valid_themes.append(theme_info["name"])
            themes = valid_themes if valid_themes else ["нет рисков"]
        return {
            "rating": rating,
            "themes": themes,
            "reason": str(best_parsed.get("reason", "OK")).strip(),
            "evidence": str(best_parsed.get("evidence", ""))[:200].strip()
        }

    return {
        "rating": "18+",
        "themes": ["нет рисков"],
        "reason": "Не найден валидный JSON.",
        "evidence": response[:150].replace('\n', ' ')
    }


def process_json_file(input_file: str, output_dir: str = "results") -> dict:
    """Основная функция обработки JSON файла"""
    
    # Создаем директорию для результатов
    Path(output_dir).mkdir(exist_ok=True)
    
    # Загружаем модель и RAG систему
    model, tokenizer = load_model()
    theme_rag = ThemeRAGSystem()
    
    print(f"Загрузка файла: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            scenes = json.load(f)
        print(f"Загружено {len(scenes)} сцен.")
    except Exception as e:
        print(f"Ошибка чтения: {e}")
        return None

    results = []
    theme_counter = {}
    rag_insights = []

    for i, sc in enumerate(scenes, 1):
        num = sc.get("scene_number", f"#{i}")
        print(f"Сцена {i}/{len(scenes)}: {num}")
        
        try:
            clean_text = clean_scene_text(sc["text"])

            # RAG предварительный анализ
            rag_themes = theme_rag.find_relevant_themes(clean_text, top_k=3)
            rag_insights.append({
                "scene_number": num,
                "rag_suggestions": rag_themes
            })

            prompt = build_prompt(clean_text, theme_rag)
            resp = generate_response(prompt, model, tokenizer)
            anal = parse_response(resp, theme_rag)
            
        except Exception as e:
            print(f"Ошибка: {e}")
            anal = {
                "rating": "18+",
                "themes": ["нет рисков"],
                "reason": f"Сбой: {e}",
                "evidence": ""
            }

        res = {
            "scene_number": num,
            "scene_title": sc.get("scene_title", ""),
            "rating": anal["rating"],
            "themes": anal["themes"],
            "reason": anal["reason"],
            "evidence": anal["evidence"],
            "full_text": sc["text"]
        }
        results.append(res)

        for t in anal["themes"]:
            theme_counter[t] = theme_counter.get(t, 0) + 1

    # Итоговый рейтинг
    order = ["0+", "6+", "12+", "16+", "18+"]
    final = max(
        (r["rating"] for r in results if r["rating"] in order),
        key=order.index,
        default="18+"
    )

    # Формируем отчет
    report = {
        "meta": {
            "total_scenes": len(scenes),
            "final_rating": final,
            "theme_stats": dict(sorted(theme_counter.items(), key=lambda x: -x[1])),
            "rag_analysis": {
                "total_suggestions": sum(len(insight["rag_suggestions"]) for insight in rag_insights),
                "top_rag_themes": dict(sorted(
                    {theme: sum(1 for insight in rag_insights for t in insight["rag_suggestions"] if t["name"] == theme)
                     for theme in set(t["name"] for insight in rag_insights for t in insight["rag_suggestions"])}.items(),
                    key=lambda x: -x[1]
                )[:10])
            }
        },
        "scenes": results,
        "rag_insights": rag_insights
    }

    # Сохранение результатов
    input_path = Path(input_file)
    output_file = Path(output_dir) / f"{input_path.stem}_RATING_RESULT.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nГотово!")
    print(f"Итоговый рейтинг: **{final}**")
    print(f"RAG-анализ: {report['meta']['rag_analysis']['total_suggestions']} предложений тем")
    print(f"Результаты сохранены в: {output_file}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Анализ возрастного рейтинга сценария')
    parser.add_argument('input_file', help='Путь к JSON файлу со сценами')
    parser.add_argument('-o', '--output', default='results', help='Директория для результатов (по умолчанию: results)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Файл {args.input_file} не найден!")
        sys.exit(1)
    
    print(f"GPU: {'Да' if torch.cuda.is_available() else 'Нет'}")
    if torch.cuda.is_available():
        print(f"   Устройство: {torch.cuda.get_device_name(0)}")
    
    print("Запуск анализа...")
    report = process_json_file(args.input_file, args.output)
    
    if report:
        print(f"\nАнализ завершен! Итоговый рейтинг: {report['meta']['final_rating']}")
        print("\nТоп-5 тем:")
        for i, (theme, count) in enumerate(list(report['meta']['theme_stats'].items())[:5], 1):
            print(f"  {i}. {theme} — {count} сцен")
    else:
        print("\nОшибка при обработке файла")
        sys.exit(1)

if __name__ == "__main__":
    main()
