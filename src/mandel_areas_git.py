from PIL import Image, ImageDraw
import cv2
import numpy as np

MAX_ITER = 80
WIDTH = 800
HEIGHT = 600


def get_mandelbrot_iterations(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n


def generate_colors(n):
    palette = [(0, 0, 0)]
    img = np.zeros([1, 1, 3], dtype=np.uint8)
    for i in range(n):
        img[:] = ((i / n) * 255, 255 * 0.85, 255)
        rgb = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        palette.append((rgb[0, 0][0], rgb[0, 0][1], rgb[0, 0][2]))

    return palette


def process(x, y, radius_x, radius_y):
    real_start = x - radius_x
    real_end = x + radius_x
    imaginary_start = y - radius_y
    imaginary_end = y + radius_y

    im = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    iterations = 500
    palette = generate_colors(3000)
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            # Convert pixel coordinate to complex number
            c = complex(real_start + (x / WIDTH) * (real_end - real_start),
                        imaginary_start + (y / HEIGHT) * (imaginary_end - imaginary_start))
            # Compute the number of iterations
            m = get_mandelbrot_iterations(c, iterations)

            if m >= iterations:
                m = 0
            index = (m * len(palette)) // iterations
            draw.point([x, y], (palette[index][0], palette[index][1], palette[index][2]))
    res = np.asarray(im)
    cv2.imshow('Result', res)
    cv2.waitKey(1)

    return res


if __name__ == '__main__':
    areas = [(-0.16, 1.0405, 0.026, 0.026), (-0.722, 0.246, 0.019, 0.019), (-0.235125, 0.827215, 4.0E-5, 4.0E-5), (-0.748, 0.1, 0.0014, 0.0014),
             (-0.7463, 0.1102, 0.005, 0.005), (-1.25066, 0.02012, 2.4E-4, 1.7E-4), (-0.745428, 0.113009, 3.9E-5, 3.0E-5),
             (-0.748, 0.1, 0.0014, 0.0014), (-0.7453, 0.1127, 6.5E-4, 6.5E-4)]
    for i, area in enumerate(areas):
        if i > 5:
            res = process(area[0], area[1], area[2], area[3])