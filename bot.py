from telebot import *
import json
import os
import numpy as np
from tensorflow.keras.models import load_model
import cv2

from random import *

MODEL_PATH = "model/NeyronchikBeter.h5" # локальный путь к модели
userdirectory = "." # текущая директория


unique_labels = ["paper", "rock", "scissors", "unknown"]
model = load_model(MODEL_PATH)


bot = telebot.TeleBot('7932251208:AAHaoaBKRcVQQKxXAA2on9DfYeiKFJI-KC0')

def process_image(image):
    # Проверяем среднюю яркость изображения
    brightness = np.mean(image)
    print(f"Средняя яркость: {brightness}")
    
    # Если изображение слишком темное (средняя яркость меньше 100)
    if brightness < 100:
        # Увеличиваем яркость
        alpha = 1.5  # коэффициент контрастности
        beta = 30    # коэффициент яркости
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        print("Яркость увеличена")
    
    return image

def roll_rps(vvod, message):
    rps = ["rock", "paper", "scissors"]
    if rps.count(vvod) != 0:
        vibor = choice(rps)
        bot.send_message(message.chat.id, "Выбор оппонента: " + vibor, parse_mode='html')
        if vibor == vvod:
            return "tie"

        elif vibor == "rock":
            if vvod == "scissors":
                return "you lose"
            elif vvod == "paper":
                return "you win"

        elif vibor == "paper":
            if vvod == "scissors":
                return "you win"
            elif vvod == "rock":
                return "you lose"

        elif vibor == "scissors":
            if vvod == "rock":
                return "you win"
            elif vvod == "paper":
                return "you lose"

    else:
        warning_message = "⚠️ Внимание! Низкая точность определения. Рекомендации:\n"
        warning_message += "• Убедитесь, что фото четкое\n"
        warning_message += "• Сделайте фото при хорошем освещении\n"
        warning_message += "• Покажите жест на фоне нейтральной поверхности\n"
        warning_message += "• Убедитесь, что на фото нет других предметов\n"
        warning_message += "• Убедитесь, что жест хорошо виден в кадре\n\n"
        return warning_message

def writeStats(message, statlist):
    with open(f'{userdirectory}/users/{message.from_user.username}.json', 'w+') as file:
        json.dump(statlist, file)


def getStats(message):
    with open(f'{userdirectory}/users/{message.from_user.username}.json', 'r') as file:
        return json.load(file)


@bot.message_handler(commands=['start'])
def startBot(message):
    bot.send_message(message.chat.id, "Отправьте фото вашего выбора.", parse_mode='html')
    writeStats(message, [0, 0, 0])

def resize_with_pad(image, target_size):
    """Изменяет размер изображения с сохранением пропорций и добавляет черные полосы при необходимости"""
    target_w, target_h = target_size
    h, w = image.shape[:2]
    
    # Вычисляем масштаб
    scale = min(target_w/w, target_h/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Изменяем размер изображения
    resized = cv2.resize(image, (new_w, new_h))
    
    # Создаем новое изображение с черным фоном
    new_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Вычисляем позицию для центрирования изображения
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Размещаем изображение по центру
    new_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return new_image

@bot.message_handler(content_types=['photo'])
def checkImage(message):
    if message.photo:
        # Получение самого большого по размеру изображения
        photo = message.photo[-1]

        # Загрузка файла
        file_info = bot.get_file(photo.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        save_path = 'photo.jpg'
        with open(save_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        # Предобработка изображения
        image = cv2.imread(save_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Изменяем размер с сохранением пропорций
        image = resize_with_pad(image, (128, 128))
        
        # Проверяем и корректируем яркость
        image = process_image(image)
        
        # Нормализуем значения пикселей
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Предсказание
        predictions = model.predict(image)
        
        # Получаем индекс класса unknown (3)
        unknown_index = unique_labels.index("unknown")
        unknown_probability = predictions[0][unknown_index] * 100
        
        # Если вероятность unknown не 100%, выбираем второе максимальное значение
        if unknown_probability < 99:
            # Создаем копию предсказаний без класса unknown
            predictions_without_unknown = predictions[0].copy()
            predictions_without_unknown[unknown_index] = 0
            
            # Получаем индекс второго максимального значения
            predicted_label = unique_labels[np.argmax(predictions_without_unknown)]
            confidence = np.max(predictions_without_unknown) * 100
        else:
            predicted_label = "unknown"
            confidence = unknown_probability
            
        for prediction in predictions:
            print(prediction * 100)

            warning_message = ""
            roll = roll_rps(predicted_label, message)
            w, l, t = getStats(message)[0], getStats(message)[1], getStats(message)[2]
            if roll == "you win":
                writeStats(message, [int(w) + 1, l, t])
                roll = "Вы выиграли!"
            if roll == "you lose":
                writeStats(message, [w, int(l) + 1, t])
                roll = "Вы проиграли!"
            if roll == "tie":
                writeStats(message, [w, l, int(t) + 1])
                roll = "Ничья."
            bot.send_message(message.chat.id, f'Распознанный жест: {predicted_label}', parse_mode='html')
            bot.send_message(message.chat.id, roll, parse_mode='html')
            bot.send_message(message.chat.id, "Отправьте фото вашего выбора.", parse_mode='html')
           
        # Удаление сохранённого изображения
        os.remove(save_path)
        
    else:
        bot.send_message(message.chat.id, "Отправьте фото вашего выбора.", parse_mode='html')


@bot.message_handler(commands=['leaderboard'])
def showLeaderboard(message):
    directory = f"{userdirectory}/users"
    leaderboard = {}
    text = ""
    files = os.listdir(directory)
    for user in files:
        with open(f'{userdirectory}/users/{user}', 'r') as file:
            leaderboard[user] = json.load(file)[0]
    sortedLeaderboard = sorted(leaderboard.items(), key=lambda leader: leader[1], reverse=True)
    for user in sortedLeaderboard:
        text += f"{user[0].replace('.json', '')}: {user[1]}\n"
    bot.send_message(message.chat.id, text, parse_mode='html')

@bot.message_handler(commands=['stats'])
def showStats(message):
    bot.send_message(message.chat.id, f"w {getStats(message)[0]}, l {getStats(message)[1]}, t {getStats(message)[2]}", parse_mode='html')

bot.polling(none_stop=True)