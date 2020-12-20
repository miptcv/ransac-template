import sys, os.path, json, random, math, cv2, numpy as np


def generate_data(
        img_size: tuple, line_params: tuple,
        n_points: int, sigma: float, inlier_ratio: float
) -> np.ndarray:
    img = np.zeros((img_size[0], img_size[1]), dtype=int)
    data = np.zeros((n_points,3),dtype= int)
    inp = int(round(n_points * inlier_ratio))
    A = line_params[0]
    B = -line_params[1]
    C = -line_params[2]
    Pepe = 0

    x0,y0,x1,y1 = 0,0,0,0
    if C * A < 0:
        x0 = 0
        y0 = C/B
    elif C * A > 0:
        y0 = 0
        x0 = -C/A
    else:
        x0 = 0
        y0 = 0


    if (B * img_size[0] + C) / A < img_size[1]:
        y1 = img_size[0]
        x1 = (B * img_size[0] + C) / A

    elif (A * img_size[1] + C) / B < img_size[0]:
        x1 = img_size[1]
        y1 = (A * img_size[1] + C) / B
    else:
        x1 = img_size[1] - 1
        y1 = img_size[0] - 1
    if (A < 0 and B > 0) or (A > 0 and B < 0):
        x1 = C/B

    for i in range(inp):
        rhek = int(round(random.random() * abs(x1 - x0)))
        sig = int(round(random.random() * abs(2 * sigma) - sigma))
        X = rhek + x0
        Y = X*(A/B) + C
        X = X + sig
        if X >= x1:
            X = x1 - 1
        if X <= x0:
            X = x0 + 1
        if Y >= img_size[0]:
            Y = img_size[0] - 1
        if Y <= 0:
            Y = 1
        data[Pepe][0] = int(Y)
        data[Pepe][1] = int(X)
        Pepe += 1

        img[int(Y)][int(X)] = 255
    G = ((img_size[0] * img_size[1]) / (n_points - inp))
    for i in range((n_points - inp)):
        G1 = int((random.random() * G))
        img[round(i * G + G1) // img_size[1]][round(i * G + G1) % img_size[1]] = 255
        data[Pepe][0] = round(i * G + G1) // img_size[1]
        data[Pepe][1] = round(i * G + G1) % img_size[1]
        Pepe += 1
    cv2.imwrite('out.png', img)
    return data




def compute_ransac_threshold(
        alpha: float, sigma: float
) -> float:
    return abs(math.sqrt(alpha * sigma**2))


def compute_ransac_iter_count(
        conv_prob: float, inlier_ratio: float
) -> int:
    N = math.log(0.05)/math.log(1 - inlier_ratio ** conv_prob)
    return N


def compute_line_ransac(
        data: np.ndarray, threshold: float, iter_count: int, N, N2
) -> tuple:

    NN = N
    NNN = int(round(NN * N2))
    max, MAX = 0, 0
    A, AA = 0, 0
    B, BB = 0, 0
    C, CC = 0, 0
    for i in range(iter_count):
        preflop = np.zeros(NNN)
        tt = 0
        while tt < NNN:
            RR = int(round(random.random() * NN)) - 1
            if data[RR][2] != i + 1:
                preflop[tt] = RR
                data[RR][2] = i + 1
                tt += 1
        MAX = 0
        for h in range(iter_count):
            RR0 = int(round(random.random() * NNN))
            RR1 = RR0
            while (RR1 == RR0):
                RR1 = int(round(random.random() * NNN))
            A0 = data[RR1][0] - data[RR0][0]
            B0 = -(data[RR1][1] - data[RR0][1])
            C0 = -data[RR0][0] * A0 - data[RR0][1] * B0
            counter = 0
            for k in range(NNN):
                KKona = int(preflop[k])

                if (abs(A0 * data[KKona][1] + B0 * data[KKona][0] + C0) / math.sqrt(A0 ** 2 + B0 ** 2)) < threshold:
                    counter += 1

            if (counter > MAX):
                MAX = counter
                AA = A0
                BB = B0
                CC = C0
        counter = 0
        for k in range(NN):
            if (abs(AA * data[k][1] + BB * data[k][0] + CC) / math.sqrt(AA ** 2 + BB ** 2)) < threshold:
                counter += 1
        if counter > max:
            A = AA
            B = BB
            C = CC
    return[A, B, C]


def detect_line(params: dict) -> tuple:
    data = generate_data(
        (params['w'], params['h']),
        (params['a'], params['b'], params['c']),
        params['n_points'], params['sigma'], params['inlier_ratio']
    )
    threshold = compute_ransac_threshold(
        params['alpha'], params['sigma']
    )
    iter_count = round(compute_ransac_iter_count(
        params['conv_prob'], params['inlier_ratio']
    ))


    detected_line = compute_line_ransac(data, threshold, iter_count, params['n_points'], params['inlier_ratio'])
    return detected_line


def main():
    assert len(sys.argv) == 2
    params_path = sys.argv[1]
    assert os.path.exists(params_path)
    with open(params_path) as fin:
        params = json.load(fin)
    assert params is not None

    """
    params:
    line_params: (a,b,c) - line params (ax+by+c=0) 
    img_size: (w, h) - size of the image
    n_points: count of points to be used
    sigma - Gaussian noise
    alpha - probability of point is an inlier
    inlier_ratio - ratio of inliers in the data
    conv_prob - probability of convergence
    """

    detected_line = detect_line(params)
    print(detected_line)


if __name__ == '__main__':
    main()