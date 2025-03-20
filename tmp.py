import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Получение разрешения экрана
screen_width, screen_height = pyautogui.size()

# Настройки для стабилизации клика
click_delay = 1  # минимальное время между кликами в секундах
last_click_time = 0

# Переменные для сглаживания движения
prev_screen_x, prev_screen_y = 0, 0
smooth_factor = 0.6  # Коэффициент сглаживания (0.0 - без сглаживания, 1.0 - мгновенное движение)

# Функция для вычисления расстояния между двумя точками
def calc_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Переворачиваем кадр для зеркального эффекта
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Получение координат большого пальца (кончик) - id 4
            thumb_tip = handLms.landmark[4]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Отображение точки большого пальца
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)

            # Маппинг координат большого пальца с камеры на экран
            screen_x = int(thumb_tip.x * screen_width)
            screen_y = int(thumb_tip.y * screen_height)

            # Сглаживание движения курсора
            screen_x = int(prev_screen_x + (screen_x - prev_screen_x) * smooth_factor)
            screen_y = int(prev_screen_y + (screen_y - prev_screen_y) * smooth_factor)

            # Обновляем предыдущие координаты
            prev_screen_x, prev_screen_y = screen_x, screen_y

            # Перемещение курсора
            pyautogui.moveTo(screen_x, screen_y)

            # Получение координат указательного пальца (кончик) - id 8
            index_tip = handLms.landmark[8]
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)

            # Вычисление расстояния между большим и указательным пальцами
            distance = calc_distance(thumb_x, thumb_y, index_x, index_y)
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 0, 255), 2)

            # Если расстояние меньше порога, эмулируем клик
            if distance < 40:
                current_time = time.time()
                if current_time - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = current_time
                    cv2.putText(frame, "CLICK", (thumb_x, thumb_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Отрисовка скелета руки
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()