import argparse
import numpy as np
import cv2
from tqdm import tqdm


def generate_colors(n):
    palette = [(0, 0, 0)]
    img = np.zeros([1, 1, 3], dtype=np.uint8)
    for i in range(n):
        img[:] = ((i / n) * 255, 255 * 0.85, 255)
        rgb = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        palette.append((rgb[0, 0][0], rgb[0, 0][1], rgb[0, 0][2]))

    return palette


def calc_mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1

    return n


def iteration_animation(args):
    iterations = args.resulting_video
    resulting_video = args.output
    real_init = -2
    real_end = 1
    imaginary_init = -1
    imaginary_end = 1

    width = 800
    height = 600
    img = np.zeros([height, width, 3], dtype=np.uint8)

    pbar = tqdm(total=iterations)

    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    out_cutter_name = resulting_video
    out = cv2.VideoWriter(out_cutter_name, fourcc, 24, (width, height))

    delta_real = real_end - real_init
    delta_imaginary = imaginary_end - imaginary_init
    palette = generate_colors(3000)
    for i in range(1, iterations + 1):
        max_iterations = i
        for x in range(width):
            for y in range(height):
                # Convert pixel coordinate to complex number
                c = complex(real_init + (x / width) * delta_real, imaginary_init + (y / height) * delta_imaginary)
                # Compute the number of iterations
                m = calc_mandelbrot(c, max_iterations)
                if m >= max_iterations:
                    m = 0

                color = 255 - int(m * 255 / max_iterations)
                index = (m * len(palette)) // max_iterations
                img[y, x] = (palette[index][0], palette[index][1], palette[index][2])

        cv2.imshow('', img)
        cv2.waitKey(1)
        out.write(img)
        pbar.update(1)

    out.release()


def palette_animation(args):
    iterations = args.iterations
    step = args.step
    resulting_video = args.output

    real_init = -2
    real_end = 1
    imaginary_init = -1
    imaginary_end = 1

    width = 800
    height = 600
    img = np.zeros([height, width, 3], dtype=np.uint8)
    iters = np.zeros([height, width, 1], dtype=np.uint8)

    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    out_cutter_name = resulting_video
    out = cv2.VideoWriter(out_cutter_name, fourcc, 24, (width, height))

    delta_real = real_end - real_init
    delta_imaginary = imaginary_end - imaginary_init
    max_iterations = 150
    palette = generate_colors(3000)
    for x in range(width):
        for y in range(height):
            # Convert pixel coordinate to complex number
            c = complex(real_init + (x / width) * delta_real, imaginary_init + (y / height) * delta_imaginary)
            # Compute the number of iterations
            m = calc_mandelbrot(c, max_iterations)
            if m >= max_iterations:
                m = 0

            index = (m * len(palette)) // max_iterations
            iters[y, x] = index

    init = min(10, iterations)
    end = max(2 * init, iterations)
    pbar = tqdm(total=(end-init)//step)
    for i in range(init, end, step):
        for x in range(width):
            for y in range(height):
                index = iters[y, x][0]
                if index > 0:
                    if index + i < 3000:
                        index += i
                    else:
                        index = 3000 - (index + i)
                img[y, x] = (palette[index][0], palette[index][1], palette[index][2])
        cv2.imshow('', img)
        cv2.waitKey(1)
        out.write(img)
        pbar.update(1)
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default='pal', type=str, help='animation type')
    parser.add_argument('-o', '--output', default='result.mp4', type=str, help='resulting video')
    parser.add_argument('-i', '--iterations', default=100, type=int, help='number of iterations')
    parser.add_argument('-s', '--step', default=1, type=int, help='how fast colours change')
    args = parser.parse_args()
    print(args)
    if args.type in ['iter']:
        iteration_animation(args)
    elif args.type in ['pal']:
        palette_animation(args)