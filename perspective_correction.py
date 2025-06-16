import cv2 as cv
import numpy as np

from Gaussian_method.lin_algebra import inverse_matrix, dot_mv


# Расчёт матрицы гомографии в предположении h_33 = 1
def compute_homography(src_pts, dst_pts):
    if src_pts.shape != (4, 2) or dst_pts.shape != (4, 2):
        raise ValueError('src_pts and dst_pts must be 4x2 arrays')

    A = []
    b = []
    # Составление матрицы A и вектора b
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        # u: [x, y, 1, 0,0,0, -u*x, -u*y] * h' = u
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        # v: [0,0,0, x, y, 1, -v*x, -v*y] * h' = v
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)


    # Непосредственно нахождение матрицы гомографии
    h_prime = np.linalg.solve(A, b)

    # Дополнение h_33 = 1
    h = np.concatenate([h_prime, [1]])

    # Преобразование вектора в матрицу 3x3
    H = h.reshape((3, 3))

    return H
def warp_perspective(src, H, dsize):
    width, height = dsize # Размер выходного изображения
    dst = np.zeros((height, width, 3), dtype=np.uint8) # Выходное изображение

    H_inv = inverse_matrix(H) # Обратная матрица гомографии (для обратного преобразования) (x, y, 1)^T ~ H^-1 @ (u, v, 1)^T

    for y_dst in range(height):
        for x_dst in range(width):
            # Преобразование точки обратно в координаты исходного изображения
            vec = np.array([x_dst, y_dst, 1.0])
            src_cords = dot_mv(H_inv, vec) # Получение координат с исходного изображения в итоговом
            src_cords /= src_cords[2] # Деление на масштабный коэффициент (в данной реализации всегда 1)

            x_src, y_src = src_cords[0], src_cords[1]

            # Проверка, что точка в пределах итогового изображения
            if 0 <= int(y_src) < src.shape[0] and 0 <= int(x_src) < src.shape[1]:
                dst[y_dst, x_dst] = src[int(y_src), int(x_src)]

    return dst

def warp_image(height, width, scale_factor_for_out):
    global pts_src, im_src, scale_factor

    np_pts_src = np.array(pts_src) # Преобразование в np.array что бы можно было домножить на scale_factor_for_out
    pts_dst = np.array([[0, 0],
                        [width*scale_factor_for_out - 1, 0],
                        [width*scale_factor_for_out - 1, height*scale_factor_for_out - 1],
                        [0, height*scale_factor_for_out - 1]]) # Точки соответствия на итоговом изображении

    h = compute_homography(np_pts_src*scale_factor, pts_dst) # Вычисление матрицы гомографии
    print(h)
    im_out = warp_perspective(im_src, h, (width*scale_factor_for_out, height*scale_factor_for_out)) # Формирование итогового изображения

    # Отображение
    cv.imshow('warped src image', im_out)
    cv.moveWindow('warped src image', 1200, 10)

    return im_out

# Метод для работы с мышкой
def mouse_callback(event, x, y, flags, param):
    global pts_src

    # event нажатия левой кнопки мыши
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(scaled_im_src, (x, y), 10, (0, 0, 255), 2)
        cv.putText(img=scaled_im_src, text=f'({x};{y})', org=(x-70,y-30), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv.LINE_AA)
        pts_src.append([x,y]) # Добавление отмеченных координат в список исходных точек соответствия
def open_editor(img):
    cv.namedWindow('Editor')
    cv.moveWindow('Editor', 10, 10)
    cv.setMouseCallback('Editor', mouse_callback)

    while True:
        cv.imshow('Editor', img)

        if cv.waitKey(1) & 0xFF == 27:
            break

pts_src = []
im_src = cv.imread('IMG_0950.JPG')
scale_factor = im_src.shape[0]//(1080*0.7) # Множитель, что бы изображение поместилось на экран

if __name__ == '__main__':
    # Уменьшение изображения для того что бы открыть его в редакторе
    scaled_im_src = cv.resize(im_src, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv.INTER_LINEAR)
    open_editor(scaled_im_src)

    warp_image(28, 34, 30) # 29, 21, 30 (для IMG0967.PNG) (соотношение сторон)

    cv.waitKey(0)
